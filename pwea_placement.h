#pragma once

#include "eahmmf_optimizer.h"
#include "pwea_client.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace eahmmf {

enum class PWEAPlanStatus {
    ALREADY_PLACED,
    DRY_RUN_FEASIBLE,
    REJECTED
};

struct PWEAPlacementConfig {
    double hot_threshold = 0.75;
    double warm_threshold = 0.25;
    double simplex_tolerance = 1e-8;

    // Used only when ETF history is supplied.
    double prediction_weight = 0.8;
    double history_weight = 0.2;

    // HOT experts are placed in VRAM only when a CUDA device is available
    // AND GPU expert compute is enabled; otherwise HOT experts use RAM.
    // Defaults to false so callers must opt in explicitly.
    bool gpu_compute = false;
};

struct PWEAPlacementRecommendation {
    int layer = -1;
    int expert_id = -1;

    double predicted_probability = 0.0;
    double historical_activation_score = 0.0;
    double normalized_prediction_score = 0.0;
    double normalized_history_score = 0.0;
    double combined_score = 0.0;
    double normalized_score = 0.0;

    DemandClass classification = DemandClass::COLD;
    Residency current = Residency::SSD;
    Residency target = Residency::SSD;

    size_t footprint = 0;
    PWEAPlanStatus status = PWEAPlanStatus::REJECTED;
    std::string reason;
};

struct PWEAPlacementReport {
    bool valid = false;
    std::string error;

    size_t evaluated = 0;
    size_t hot = 0;
    size_t warm = 0;
    size_t cold = 0;

    size_t already_placed = 0;
    size_t dry_run_feasible = 0;
    size_t rejected = 0;

    std::vector<PWEAPlacementRecommendation> placements;
};

inline bool pwea_descriptor_is_stable(const ExpertDescriptor& d) {
    if (d.pending_mask != 0) return false;

    switch (d.state) {
        case Residency::SSD:
            return !d.ram_ptr && !d.vram_ptr &&
                   d.ram_mask == 0 && d.vram_mask == 0;

        case Residency::RAM:
            return d.ram_ptr && !d.vram_ptr &&
                   d.ram_mask == 7 && d.vram_mask == 0;

        case Residency::VRAM:
            return d.vram_ptr && !d.ram_ptr &&
                   d.vram_mask == 7 && d.ram_mask == 0;

        default:
            return false;
    }
}

inline std::string pwea_capacity_rejection(
    const PlacementCapacity& capacity,
    Residency current,
    Residency target,
    size_t footprint) {

    if (footprint == 0) return "Zero expert footprint";

    const bool needs_ram =
        target == Residency::RAM ||
        (current == Residency::SSD && target == Residency::VRAM);

    if (needs_ram && capacity.ram_remaining < footprint)
        return "Insufficient RAM capacity";

    if (target == Residency::VRAM) {
        if (!capacity.cuda_available)
            return "CUDA device unavailable";
        if (capacity.vram_remaining < footprint)
            return "Insufficient logical VRAM capacity";
        if (capacity.physical_vram_remaining < footprint)
            return "Insufficient physical VRAM after reserve";
    }

    return "Capacity reservation failed";
}

// Read-only prediction-based planner.
// It does not call promote(), demote(), acquire(), or modify ETF telemetry.
//
// n_layer/n_expert describe the actual ExpertStore dimensions.
// The prediction itself retains the fixed PWEA 48 x 128 output shape.
inline PWEAPlacementReport plan_pwea_placement(
    const ExpertStore& store,
    const pwea::Prediction& prediction,
    int n_layer,
    int n_expert,
    size_t vram_reserve = 0,
    const PWEAPlacementConfig& config = {},
    const std::vector<double>* historical_activation = nullptr) {

    PWEAPlacementReport report;

    if (n_layer <= 0 || n_layer > static_cast<int>(pwea::N_LAYER) ||
        n_expert <= 0 || n_expert > static_cast<int>(pwea::N_EXPERT)) {
        report.error = "ExpertStore dimensions are outside the PWEA model dimensions";
        return report;
    }

    if (!std::isfinite(config.hot_threshold) ||
        !std::isfinite(config.warm_threshold) ||
        config.warm_threshold < 0.0 ||
        config.hot_threshold > 1.0 ||
        config.warm_threshold > config.hot_threshold ||
        !std::isfinite(config.simplex_tolerance) ||
        config.simplex_tolerance <= 0.0 ||
        !std::isfinite(config.prediction_weight) ||
        !std::isfinite(config.history_weight) ||
        config.prediction_weight < 0.0 ||
        config.history_weight < 0.0 ||
        std::fabs(config.prediction_weight +
                  config.history_weight - 1.0) > 1e-12) {
        report.error = "Invalid PWEA placement thresholds or tolerance";
        return report;
    }

    // Validate every model output layer, not just the dimensions currently
    // exposed by the store.
    for (size_t layer = 0; layer < pwea::N_LAYER; ++layer) {
        double sum = 0.0;

        for (size_t expert = 0; expert < pwea::N_EXPERT; ++expert) {
            const double value = prediction.at(layer, expert);

            if (!std::isfinite(value) || value < 0.0) {
                report.error = "Prediction contains a negative or non-finite value";
                return report;
            }

            sum += value;
        }

        if (!std::isfinite(sum) ||
            std::fabs(sum - 1.0) > config.simplex_tolerance) {
            report.error = "Prediction layer does not satisfy the simplex constraint";
            return report;
        }
    }

    if (historical_activation) {
        const size_t expected_history_size =
            static_cast<size_t>(n_layer) *
            static_cast<size_t>(n_expert);

        if (historical_activation->size() != expected_history_size) {
            report.error = "ETF history dimensions do not match ExpertStore";
            return report;
        }

        for (double value : *historical_activation) {
            if (!std::isfinite(value) || value < 0.0) {
                report.error = "ETF history contains a negative or non-finite value";
                return report;
            }
        }
    }

    const bool cuda_available = store.has_cuda_device();
    const size_t physical_free = cuda_available
        ? store.device_free_memory()
        : 0;

    const size_t physical_available =
        physical_free > vram_reserve
            ? physical_free - vram_reserve
            : 0;

    PlacementCapacity capacity{
        store.ram_total() >= store.ram_used()
            ? store.ram_total() - store.ram_used() : 0,
        store.vram_total() >= store.vram_used()
            ? store.vram_total() - store.vram_used() : 0,
        physical_available,
        cuda_available
    };

    const Stats initial_stats = store.stats();
    const bool store_has_active_pins = initial_stats.pinned_now != 0;

    report.placements.reserve(
        static_cast<size_t>(n_layer) * static_cast<size_t>(n_expert));

    // Build candidates using PWEA alone or PWEA + normalized ETF history.
    for (int layer = 0; layer < n_layer; ++layer) {
        double layer_max_prediction = 0.0;
        double layer_max_history = 0.0;

        for (int expert = 0; expert < n_expert; ++expert) {
            const size_t index =
                static_cast<size_t>(layer) * n_expert + expert;

            layer_max_prediction = std::max(
                layer_max_prediction,
                prediction.at(static_cast<size_t>(layer),
                              static_cast<size_t>(expert)));

            if (historical_activation) {
                layer_max_history = std::max(
                    layer_max_history, (*historical_activation)[index]);
            }
        }

        if (!(layer_max_prediction > 0.0) ||
            !std::isfinite(layer_max_prediction)) {
            report.error = "Invalid prediction maximum for an ExpertStore layer";
            report.placements.clear();
            return report;
        }

        std::vector<double> combined_scores(
            static_cast<size_t>(n_expert), 0.0);

        double combined_max = 0.0;

        for (int expert = 0; expert < n_expert; ++expert) {
            const size_t index =
                static_cast<size_t>(layer) * n_expert + expert;

            const double prediction_score =
                prediction.at(static_cast<size_t>(layer),
                              static_cast<size_t>(expert)) /
                layer_max_prediction;

            double history_score = 0.0;

            if (historical_activation && layer_max_history > 0.0) {
                history_score =
                    (*historical_activation)[index] / layer_max_history;
            }

            const double combined =
                historical_activation
                    ? config.prediction_weight * prediction_score +
                      config.history_weight * history_score
                    : prediction_score;

            combined_scores[static_cast<size_t>(expert)] = combined;
            combined_max = std::max(combined_max, combined);
        }

        if (!(combined_max > 0.0) || !std::isfinite(combined_max)) {
            report.error = "Invalid combined placement score maximum";
            report.placements.clear();
            return report;
        }

        for (int expert = 0; expert < n_expert; ++expert) {
            PWEAPlacementRecommendation rec;

            rec.layer = layer;
            rec.expert_id = expert;

            rec.predicted_probability = prediction.at(
                static_cast<size_t>(layer),
                static_cast<size_t>(expert));

            rec.normalized_prediction_score =
                rec.predicted_probability / layer_max_prediction;

            if (historical_activation) {
                const size_t index =
                    static_cast<size_t>(layer) * n_expert + expert;

                rec.historical_activation_score =
                    (*historical_activation)[index];

                rec.normalized_history_score =
                    layer_max_history > 0.0
                        ? rec.historical_activation_score / layer_max_history
                        : 0.0;
            }

            rec.combined_score =
                combined_scores[static_cast<size_t>(expert)] /
                combined_max;

            // Keep the existing ranking and threshold interface.
            rec.normalized_score = rec.combined_score;

            if (rec.normalized_score >= config.hot_threshold) {
                rec.classification = DemandClass::HOT;
                rec.target = (cuda_available && config.gpu_compute)
                    ? Residency::VRAM : Residency::RAM;
                report.hot++;
            } else if (rec.normalized_score >= config.warm_threshold) {
                rec.classification = DemandClass::WARM;
                rec.target = Residency::RAM;
                report.warm++;
            } else {
                rec.classification = DemandClass::COLD;
                rec.target = Residency::SSD;
                report.cold++;
            }

            const ExpertDescriptor d = store.descriptor(layer, expert);

            rec.current = d.state;
            rec.footprint = d.footprint;

            if (d.layer != layer || d.expert_id != expert) {
                rec.reason = "Expert descriptor identity mismatch";
            } else if (!pwea_descriptor_is_stable(d)) {
                rec.reason = "Unstable, partial, or inconsistent residency";
            } else if (d.pin_count != 0) {
                rec.reason = "Expert currently pinned";
            } else if (d.footprint == 0) {
                rec.reason = "Zero expert footprint";
            } else {
                rec.reason.clear();
            }

            report.placements.push_back(std::move(rec));
        }
    }

    if (report.placements.size() !=
        static_cast<size_t>(n_layer) * static_cast<size_t>(n_expert)) {
        report.error = "Internal candidate count mismatch";
        report.placements.clear();
        return report;
    }

    std::stable_sort(
        report.placements.begin(),
        report.placements.end(),
        [](const PWEAPlacementRecommendation& a,
           const PWEAPlacementRecommendation& b) {
            if (a.normalized_score != b.normalized_score)
                return a.normalized_score > b.normalized_score;
            if (a.layer != b.layer)
                return a.layer < b.layer;
            return a.expert_id < b.expert_id;
        });

    for (auto& rec : report.placements) {
        if (!rec.reason.empty()) {
            rec.status = PWEAPlanStatus::REJECTED;
            report.rejected++;
            continue;
        }

        if (rec.current == rec.target) {
            rec.status = PWEAPlanStatus::ALREADY_PLACED;
            rec.reason = "Current residency already matches predicted target";
            report.already_placed++;
            continue;
        }

        if (store_has_active_pins) {
            rec.status = PWEAPlanStatus::REJECTED;
            rec.reason = "Store has outstanding expert pins";
            report.rejected++;
            continue;
        }

        if (!capacity.can_reserve(
                rec.current, rec.target, rec.footprint)) {
            rec.status = PWEAPlanStatus::REJECTED;
            rec.reason = pwea_capacity_rejection(
                capacity, rec.current, rec.target, rec.footprint);
            report.rejected++;
            continue;
        }

        if (!capacity.reserve(
                rec.current, rec.target, rec.footprint)) {
            rec.status = PWEAPlanStatus::REJECTED;
            rec.reason = "Capacity reservation failed";
            report.rejected++;
            continue;
        }

        rec.status = PWEAPlanStatus::DRY_RUN_FEASIBLE;
        rec.reason = "Passed validation and virtual capacity checks; no migration performed";
        report.dry_run_feasible++;
    }

    report.evaluated = report.placements.size();
    report.valid = true;
    return report;
}


struct PWEAExecutionResult {
    int layer = -1;
    int expert_id = -1;
    Residency from = Residency::SSD;
    Residency to = Residency::SSD;
    bool migrated = false;
    std::string reason;
};

struct PWEAExecutionReport {
    bool valid = false;
    std::string error;
    size_t attempted = 0;
    size_t migrated = 0;
    size_t failed = 0;
    size_t skipped = 0;
    PWEAPlacementReport plan;
    std::vector<PWEAExecutionResult> results;
};

// Executes only candidates accepted by the read-only planner.
// Every move uses migrate_no_eviction(); this function never calls
// promote(), demote(), or the observed-demand optimizer.
inline PWEAExecutionReport execute_pwea_placement(
    ExpertStore& store,
    const pwea::Prediction& prediction,
    int n_layer,
    int n_expert,
    size_t vram_reserve,
    size_t max_migrations,
    const PWEAPlacementConfig& config = {},
    const std::vector<double>* historical_activation = nullptr) {

    PWEAExecutionReport report;

    if (max_migrations == 0) {
        report.error = "Maximum PWEA migrations must be greater than zero";
        return report;
    }

    report.plan = plan_pwea_placement(
        store, prediction, n_layer, n_expert, vram_reserve,
        config, historical_activation);

    if (!report.plan.valid) {
        report.error = report.plan.error;
        return report;
    }

    report.valid = true;

    for (const auto& rec : report.plan.placements) {
        if (rec.status != PWEAPlanStatus::DRY_RUN_FEASIBLE)
            continue;

        if (report.attempted >= max_migrations) {
            report.skipped++;
            continue;
        }

        ++report.attempted;

        PWEAExecutionResult result;
        result.layer = rec.layer;
        result.expert_id = rec.expert_id;
        result.from = rec.current;
        result.to = rec.target;

        const ExpertDescriptor before =
            store.descriptor(rec.layer, rec.expert_id);

        if (before.layer != rec.layer ||
            before.expert_id != rec.expert_id) {
            result.reason = "Descriptor identity changed after planning";
        } else if (!pwea_descriptor_is_stable(before)) {
            result.reason = "Descriptor is no longer in stable residency";
        } else if (before.state != rec.current ||
                   before.footprint != rec.footprint) {
            result.reason = "Descriptor residency or footprint changed after planning";
        } else if (before.pin_count != 0 ||
                   store.stats().pinned_now != 0) {
            result.reason = "Store has active expert pins";
        } else if (!store.migrate_no_eviction(
                       rec.layer, rec.expert_id, rec.target)) {
            result.reason = "Protected migration failed; no eviction was permitted";
        } else {
            const ExpertDescriptor after =
                store.descriptor(rec.layer, rec.expert_id);

            if (after.state == rec.target &&
                pwea_descriptor_is_stable(after)) {
                result.migrated = true;
                result.reason = "Migration completed";
            } else {
                result.reason = "Migration returned success but final residency validation failed";
            }
        }

        if (result.migrated) {
            ++report.migrated;
        } else {
            ++report.failed;
        }

        report.results.push_back(result);
    }

    const std::string invariant_error = store.check_invariants();
    if (!invariant_error.empty()) {
        report.valid = false;
        report.error = "ExpertStore invariant failure after PWEA placement: " +
                       invariant_error;
    }

    return report;
}


} // namespace eahmmf
