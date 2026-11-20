#define EAHMMF_TESTING
#define QWEN3MOE_NO_MAIN
#include "../qwen3moe.cpp"
#include "../pwea_placement.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <limits>

static int failures = 0;
static int checks = 0;

#define CHECK(condition, message) do { \
    checks++; \
    if (!(condition)) { \
        failures++; \
        std::fprintf(stderr, "FAIL: %s\n", message); \
    } \
} while (0)

static pwea::Prediction make_prediction() {
    pwea::Prediction prediction;

    for (size_t layer = 0; layer < pwea::N_LAYER; ++layer) {
        for (size_t expert = 0; expert < pwea::N_EXPERT; ++expert)
            prediction.values[layer * pwea::N_EXPERT + expert] = 0.25 / 126.0;

        prediction.values[layer * pwea::N_EXPERT] = 0.50;
        prediction.values[layer * pwea::N_EXPERT + 1] = 0.25;
    }

    return prediction;
}

static const eahmmf::PWEAPlacementRecommendation* find_rec(
    const eahmmf::PWEAPlacementReport& report, int layer, int expert) {
    for (const auto& rec : report.placements)
        if (rec.layer == layer && rec.expert_id == expert)
            return &rec;
    return nullptr;
}

int main(int argc, char** argv) {
    init_types();

    const char* model = argc > 1 ? argv[1] : "tiny.gguf";

    Engine engine;
    engine.load(model, 128, false);

    const size_t footprint = engine.store->descriptor(0, 0).footprint;
    const auto prediction = make_prediction();

    std::fprintf(stderr, "[PWEA placement] model dimensions: %d x %d\n",
                 engine.n_layer, engine.n_expert);

    // Protected VRAM -> SSD releases VRAM directly, without RAM staging.
    {
        eahmmf::PlacementCapacity no_ram{
            0, 0, 0, false
        };

        CHECK(
            no_ram.can_reserve(
                eahmmf::Residency::VRAM,
                eahmmf::Residency::SSD,
                footprint
            ),
            "VRAM to SSD direct release does not require RAM"
        );

        CHECK(
            no_ram.reserve(
                eahmmf::Residency::VRAM,
                eahmmf::Residency::SSD,
                footprint
            ) && no_ram.ram_remaining == 0,
            "VRAM to SSD direct release consumes no RAM capacity"
        );
    }

    // Test 1: classification, fallback without CUDA, and read-only behavior.
    {
        eahmmf::Config store_config;
        store_config.ram_budget =
            (static_cast<size_t>(engine.n_layer) * 2 + 2) * footprint;
        store_config.vram_budget =
            (static_cast<size_t>(engine.n_layer) * 2 + 2) * footprint;

        CHECK(engine.init_experts(store_config),
              "initialize test ExpertStore");

        const auto before = engine.store->stats();

        const auto report = eahmmf::plan_pwea_placement(
            *engine.store, prediction,
            engine.n_layer, engine.n_expert, 0);

        CHECK(report.valid, "valid prediction produces a valid plan");
        CHECK(report.evaluated ==
                  static_cast<size_t>(engine.n_layer) * engine.n_expert,
              "all store experts are evaluated");

        const auto* hot = find_rec(report, 0, 0);
        const auto* warm = find_rec(report, 0, 1);
        const auto* cold = find_rec(report, 0, 2);

        CHECK(hot && hot->classification == eahmmf::DemandClass::HOT,
              "highest predicted expert is HOT");
        CHECK(warm && warm->classification == eahmmf::DemandClass::WARM,
              "medium predicted expert is WARM");
        CHECK(cold && cold->classification == eahmmf::DemandClass::COLD,
              "low predicted expert is COLD");

        CHECK(hot && hot->target == eahmmf::Residency::RAM,
              "HOT falls back to RAM without CUDA");
        CHECK(warm && warm->target == eahmmf::Residency::RAM,
              "WARM targets RAM");
        CHECK(cold && cold->target == eahmmf::Residency::SSD,
              "COLD targets SSD");

        CHECK(report.dry_run_feasible ==
                  static_cast<size_t>(engine.n_layer) * 2,
              "HOT and WARM candidates fit the configured RAM capacity");

        CHECK(engine.store->count_in(eahmmf::Residency::SSD) ==
                  static_cast<size_t>(engine.n_layer) * engine.n_expert,
              "planning does not change expert residency");

        const auto after = engine.store->stats();
        uint64_t migrations_before = 0, migrations_after = 0;

        for (int i = 0; i < 5; ++i) {
            migrations_before += before.mig_n[i] + before.mig_fail[i];
            migrations_after += after.mig_n[i] + after.mig_fail[i];
        }

        CHECK(migrations_before == migrations_after,
              "planning performs no migrations");
        CHECK(after.pinned_now == before.pinned_now,
              "planning does not change pin counts");
    }

    // Test 2: virtual capacity competition.
    {
        eahmmf::Config store_config;
        store_config.ram_budget = footprint;
        store_config.vram_budget = 0;

        CHECK(engine.init_experts(store_config),
              "initialize capacity competition store");

        const auto report = eahmmf::plan_pwea_placement(
            *engine.store, prediction,
            engine.n_layer, engine.n_expert, 0);

        CHECK(report.valid, "capacity-limited plan remains valid");
        CHECK(report.dry_run_feasible == 1,
              "only one equal-footprint HOT candidate fits");
        CHECK(report.rejected > 0,
              "remaining candidates are rejected after capacity is reserved");

        if (!report.placements.empty()) {
            CHECK(report.placements.front().normalized_score >= 1.0 - 1e-12,
                  "highest normalized prediction is considered first");
        }
    }

    // Test 3: active pins prevent proposed moves.
    {
        eahmmf::Config store_config;
        store_config.ram_budget =
            (static_cast<size_t>(engine.n_layer) * 2 + 2) * footprint;

        CHECK(engine.init_experts(store_config),
              "initialize pinned-store test");

        {
            auto lease = engine.store->acquire_lease(0, 0);
            CHECK(static_cast<bool>(lease), "acquire test expert lease");

            const auto report = eahmmf::plan_pwea_placement(
                *engine.store, prediction,
                engine.n_layer, engine.n_expert, 0);

            const auto* hot = find_rec(report, 0, 0);
            CHECK(report.valid, "pinned-store plan remains valid");
            CHECK(hot && hot->status == eahmmf::PWEAPlanStatus::REJECTED,
                  "pinned expert is rejected");
            CHECK(hot && hot->reason == "Expert currently pinned",
                  "individual pin rejection is explicit");
        }

        CHECK(engine.store->stats().pinned_now == 0,
              "test lease is released");
    }

    // Test 4: malformed prediction is rejected before planning.
    {
        auto invalid = prediction;
        invalid.values[0] = -0.1;

        const auto report = eahmmf::plan_pwea_placement(
            *engine.store, invalid,
            engine.n_layer, engine.n_expert, 0);

        CHECK(!report.valid, "negative prediction is rejected");
        CHECK(report.placements.empty(),
              "invalid prediction produces no placement candidates");
    }

    {
        auto invalid = prediction;
        invalid.values[0] += 0.1;

        const auto report = eahmmf::plan_pwea_placement(
            *engine.store, invalid,
            engine.n_layer, engine.n_expert, 0);

        CHECK(!report.valid, "non-simplex prediction is rejected");
    }



    // Adaptive planner tests: combine PWEA prediction with ETF history.
    {
        CHECK(engine.n_expert > 3,
              "adaptive ranking fixture has at least four experts");

        if (engine.n_expert > 3) {
            eahmmf::Config store_config;
            store_config.ram_budget =
                (static_cast<size_t>(engine.n_layer) * 3 + 2) * footprint;
            store_config.vram_budget = 0;

            CHECK(engine.init_experts(store_config),
                  "initialize adaptive placement test store");

            auto adaptive_prediction = prediction;

            // PWEA validates all 128 outputs per layer, even when the
            // test ExpertStore exposes only a subset of those experts.
            const double residual =
                (1.0 - 0.30 - 0.25 - 0.29) /
                static_cast<double>(pwea::N_EXPERT - 3);

            for (size_t expert = 0; expert < pwea::N_EXPERT; ++expert) {
                adaptive_prediction.values[expert] =
                    expert == 0 ? 0.30 :
                    expert == 1 ? 0.25 :
                    expert == 2 ? 0.29 : residual;
            }

            const size_t history_size =
                static_cast<size_t>(engine.n_layer) * engine.n_expert;

            std::vector<double> zero_history(history_size, 0.0);
            std::vector<double> history(history_size, 0.0);

            // Layer 0, expert 2 has the strongest historical activation.
            history[2] = 1.0;

            const auto pwea_only = eahmmf::plan_pwea_placement(
                *engine.store, adaptive_prediction,
                engine.n_layer, engine.n_expert, 0);

            const auto zero_history_report = eahmmf::plan_pwea_placement(
                *engine.store, adaptive_prediction,
                engine.n_layer, engine.n_expert, 0,
                eahmmf::PWEAPlacementConfig{}, &zero_history);

            const auto adaptive_report = eahmmf::plan_pwea_placement(
                *engine.store, adaptive_prediction,
                engine.n_layer, engine.n_expert, 0,
                eahmmf::PWEAPlacementConfig{}, &history);

            CHECK(pwea_only.valid,
                  "PWEA-only reference plan is valid");
            CHECK(zero_history_report.valid,
                  "zero-history adaptive plan is valid");
            CHECK(adaptive_report.valid,
                  "PWEA plus ETF history plan is valid");

            const auto* pwea_expert0 = find_rec(pwea_only, 0, 0);
            const auto* zero_expert0 = find_rec(zero_history_report, 0, 0);
            const auto* zero_expert2 = find_rec(zero_history_report, 0, 2);
            const auto* adaptive_expert0 = find_rec(adaptive_report, 0, 0);
            const auto* adaptive_expert2 = find_rec(adaptive_report, 0, 2);

            CHECK(pwea_expert0 && zero_expert0 &&
                      std::fabs(pwea_expert0->normalized_score -
                                zero_expert0->normalized_score) < 1e-12,
                  "zero ETF history preserves PWEA-only score");

            CHECK(zero_expert2 &&
                      std::fabs(zero_expert2->normalized_score -
                                adaptive_prediction.values[2] / 0.30) < 1e-12,
                  "zero ETF history preserves PWEA-only normalization");

            CHECK(adaptive_expert0 && adaptive_expert2,
                  "adaptive report exposes both comparison experts");

            CHECK(adaptive_expert2 &&
                      std::fabs(adaptive_expert2->normalized_history_score -
                                1.0) < 1e-12,
                  "ETF history is normalized within its layer");

            CHECK(adaptive_expert2 &&
                      adaptive_expert2->historical_activation_score == 1.0,
                  "recommendation exposes raw historical activation");

            CHECK(adaptive_expert2 && adaptive_expert0 &&
                      adaptive_expert2->normalized_score >
                          adaptive_expert0->normalized_score,
                  "strong ETF history can change expert ranking");

            CHECK(adaptive_expert2 &&
                      std::fabs(adaptive_expert2->normalized_score - 1.0) < 1e-12,
                  "highest combined expert score normalizes to one");

            // Invalid history dimensions must be rejected.
            std::vector<double> wrong_history(history_size - 1, 0.0);

            const auto wrong_dimensions = eahmmf::plan_pwea_placement(
                *engine.store, adaptive_prediction,
                engine.n_layer, engine.n_expert, 0,
                eahmmf::PWEAPlacementConfig{}, &wrong_history);

            CHECK(!wrong_dimensions.valid,
                  "history with incorrect dimensions is rejected");

            // Negative and non-finite history values must be rejected.
            auto negative_history = history;
            negative_history[0] = -0.1;

            const auto negative_report = eahmmf::plan_pwea_placement(
                *engine.store, adaptive_prediction,
                engine.n_layer, engine.n_expert, 0,
                eahmmf::PWEAPlacementConfig{}, &negative_history);

            CHECK(!negative_report.valid,
                  "negative ETF history is rejected");

            auto nonfinite_history = history;
            nonfinite_history[0] =
                std::numeric_limits<double>::quiet_NaN();

            const auto nonfinite_report = eahmmf::plan_pwea_placement(
                *engine.store, adaptive_prediction,
                engine.n_layer, engine.n_expert, 0,
                eahmmf::PWEAPlacementConfig{}, &nonfinite_history);

            CHECK(!nonfinite_report.valid,
                  "non-finite ETF history is rejected");

            // Blend weights must sum to one.
            eahmmf::PWEAPlacementConfig invalid_weights;
            invalid_weights.prediction_weight = 0.7;
            invalid_weights.history_weight = 0.2;

            const auto invalid_weights_report =
                eahmmf::plan_pwea_placement(
                    *engine.store, adaptive_prediction,
                    engine.n_layer, engine.n_expert, 0,
                    invalid_weights, &history);

            CHECK(!invalid_weights_report.valid,
                  "blend weights that do not sum to one are rejected");
        }
    }

    // Test 5: protected RAM placement never evicts an existing resident.
    {
        eahmmf::Config config;
        config.exec_tier = eahmmf::Residency::SSD;
        config.ram_budget = footprint;
        config.vram_budget = 0;

        CHECK(engine.init_experts(config),
              "initialize protected RAM migration store");

        CHECK(engine.store->migrate_no_eviction(
                  0, 0, eahmmf::Residency::RAM),
              "protected SSD to RAM migration succeeds");

        CHECK(!engine.store->migrate_no_eviction(
                  0, 1, eahmmf::Residency::RAM),
              "protected RAM migration rejects a full tier");

        CHECK(engine.store->residency(0, 0) == eahmmf::Residency::RAM &&
              engine.store->residency(0, 1) == eahmmf::Residency::SSD,
              "failed protected RAM migration does not evict the resident");

        {
            auto lease = engine.store->acquire_lease(0, 2);
            CHECK(static_cast<bool>(lease),
                  "acquire lease for protected migration pin test");

            CHECK(!engine.store->migrate_no_eviction(
                      0, 2, eahmmf::Residency::RAM),
                  "protected migration rejects while a lease is active");
        }

        CHECK(engine.store->stats().pinned_now == 0,
              "protected migration pin test releases its lease");

        CHECK(engine.store->check_invariants().empty(),
              "protected RAM migration preserves store invariants");
    }

    // Test 6: protected VRAM placement cannot evict, and failed SSD->VRAM
    // staging rolls the incoming expert back to SSD.
    {
        eahmmf::Config config;
        config.exec_tier = eahmmf::Residency::SSD;
        config.ram_budget = footprint;
        config.vram_budget = footprint;

        CHECK(engine.init_experts(config),
              "initialize protected VRAM migration store");

        CHECK(engine.store->migrate_no_eviction(
                  0, 0, eahmmf::Residency::VRAM),
              "protected SSD to VRAM migration succeeds");

        CHECK(!engine.store->migrate_no_eviction(
                  0, 1, eahmmf::Residency::VRAM),
              "protected VRAM migration rejects a full tier");

        CHECK(engine.store->residency(0, 0) == eahmmf::Residency::VRAM,
              "failed VRAM placement preserves the original resident");

        CHECK(engine.store->residency(0, 1) == eahmmf::Residency::SSD,
              "failed SSD to VRAM placement rolls back to SSD");

        CHECK(engine.store->ram_used() == 0,
              "failed VRAM placement releases its temporary RAM staging");

        CHECK(engine.store->migrate_no_eviction(
                  0, 0, eahmmf::Residency::SSD),
              "protected VRAM to SSD uses direct backing release");

        CHECK(engine.store->ram_used() == 0,
              "direct VRAM to SSD release requires no temporary RAM");

        CHECK(engine.store->residency(0, 0) == eahmmf::Residency::SSD,
              "direct VRAM drop ends in SSD residency");

        CHECK(engine.store->check_invariants().empty(),
              "protected VRAM migration preserves store invariants");
    }

    std::fprintf(stderr,
                 "\nPWEA placement tests: %d checks, %d failures\n",
                 checks, failures);

    return failures == 0 ? 0 : 1;
}
