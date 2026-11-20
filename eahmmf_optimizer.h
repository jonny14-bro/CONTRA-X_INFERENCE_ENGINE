#pragma once

#include "eahmmf.h"
#include "etf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <utility>

namespace eahmmf {

enum class DemandClass : uint8_t {
    COLD = 0,
    WARM = 1,
    HOT = 2
};

struct ExpertDemand {
    int layer = -1;
    DemandClass classification = DemandClass::COLD;
    int expert_id = -1;

    Residency current_residency = Residency::SSD;
    size_t footprint = 0;
    uint32_t pin_count = 0;

    uint64_t activation_count = 0;
    double routing_weight_sum = 0.0;
    double historical_activation_score = 0.0;

    uint64_t access_delta = 0;
    uint64_t failure_delta = 0;

    uint64_t requested_hit_delta[3] = {};
    uint64_t requested_miss_delta[3] = {};
    uint64_t served_delta[3] = {};

    uint64_t acquire_sample_delta[3] = {};
    double acquire_latency_delta_us[3] = {};
    double acquire_average_latency_us[3] = {};
    bool acquire_latency_available[3] = {};

    double acquire_historical_average_latency_us[3] = {};
    bool acquire_historical_latency_available[3] = {};

    uint64_t migration_count_delta[5] = {};
    uint64_t migration_failure_delta[5] = {};
    uint64_t migration_bytes_delta[5] = {};
    double migration_latency_delta_us[5] = {};

    double migration_historical_average_latency_us[5] = {};
    bool migration_historical_latency_available[5] = {};

    double priority = 0.0;
};

// Shared deterministic ordering used by the planner and its tests.
inline std::vector<size_t> priority_order(
    const std::vector<ExpertDemand>& demand
) {
    std::vector<size_t> order(demand.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;

    std::sort(order.begin(), order.end(),
        [&demand](size_t a, size_t b) {
            const auto& lhs = demand[a];
            const auto& rhs = demand[b];

            if (lhs.priority != rhs.priority)
                return lhs.priority > rhs.priority;
            if (lhs.layer != rhs.layer)
                return lhs.layer < rhs.layer;
            return lhs.expert_id < rhs.expert_id;
        });

    return order;
}

struct ExpertCounterSnapshot {
    uint64_t access_attempts = 0;
    uint64_t access_failures = 0;
    uint64_t requested_hit[3] = {};
    uint64_t requested_miss[3] = {};
    uint64_t served_from[3] = {};
    uint64_t acquire_samples[3] = {};
    double acquire_latency_us[3] = {};
    double acquire_latency_us_total = 0.0;
    uint64_t migration_count[5] = {};
    uint64_t migration_failures[5] = {};
    uint64_t migration_bytes[5] = {};
    double migration_latency_us[5] = {};
};

struct PlacementRecommendation {
    int layer = -1;
    int expert_id = -1;

    Residency current = Residency::SSD;
    Residency target = Residency::SSD;

    double priority = 0.0;
    double estimated_benefit = 0.0;
    double estimated_cost_us = 0.0;

    size_t footprint = 0;

    bool cost_estimate_available = false;
    bool actionable = false;

    std::string reason;
};

struct OptimizerReport {
    uint64_t first_prompt_id = 0;
    uint64_t last_prompt_id = 0;
    uint64_t completed_prompts = 0;
    uint64_t dropped_events = 0;

    size_t experts_observed = 0;
    size_t recommendations = 0;

    std::vector<ExpertDemand> demand;
    std::vector<PlacementRecommendation> placements;

    std::string status;
};

struct OptimizerConfig {
    // Initial policy weights. These are configurable, not claimed as
    // empirically optimal; they must be calibrated through benchmarks.
    double activation_weight = 1.0;
    double activation_history_decay = 0.8;
    double routing_weight = 0.25;
    double miss_weight = 0.5;
    double failure_penalty = 1.0;

    // Hysteresis: require a meaningful priority difference before
    // recommending a change in residency.
    double minimum_priority = 0.0;
    double minimum_priority_gain = 0.10;

    // Relative activation thresholds, evaluated independently per layer.
    double hot_threshold = 0.75;
    double warm_threshold = 0.25;

    // Placement requires measured acquisition benefit and migration cost.
    bool require_measured_migration_cost = true;
    bool require_measured_acquire_benefit = true;
    double minimum_latency_benefit_us = 0.0;
    double maximum_migration_cost_ratio = 1.0;
};


// Deterministic virtual-capacity accounting for read-only placement planning.
// This helper never allocates memory or performs migrations.
struct PlacementCapacity {
    size_t ram_remaining;
    size_t vram_remaining;
    size_t physical_vram_remaining;
    bool cuda_available;

    bool can_reserve(Residency current, Residency target,
                     size_t footprint) const {
        if (footprint == 0) return false;

        // SSD -> VRAM stages through RAM. VRAM -> SSD directly releases
        // the device allocation because the immutable GGUF remains backing.
        const bool needs_ram =
            target == Residency::RAM ||
            (current == Residency::SSD && target == Residency::VRAM);

        if (needs_ram && ram_remaining < footprint) return false;

        if (target == Residency::VRAM) {
            if (!cuda_available ||
                vram_remaining < footprint ||
                physical_vram_remaining < footprint)
                return false;
        }

        return true;
    }

    bool reserve(Residency current, Residency target,
                 size_t footprint) {
        if (!can_reserve(current, target, footprint)) return false;

        if (target == Residency::RAM ||
            (current == Residency::SSD && target == Residency::VRAM))
            ram_remaining -= footprint;

        if (target == Residency::VRAM) {
            vram_remaining -= footprint;
            physical_vram_remaining -= footprint;
        }

        return true;
    }
};

class AdaptiveOptimizer {
public:
    explicit AdaptiveOptimizer(const OptimizerConfig& config = {})
        : config_(config) {}

    AdaptiveOptimizer(const AdaptiveOptimizer&) = delete;
    AdaptiveOptimizer& operator=(const AdaptiveOptimizer&) = delete;

    void prime(const ExpertStore& store, int n_layer, int n_expert) {
        previous_counters_.clear();
        activation_history_.clear();

        if (n_layer <= 0 || n_expert <= 0) return;

        for (int layer = 0; layer < n_layer; ++layer) {
            for (int expert = 0; expert < n_expert; ++expert) {
                const auto descriptor = store.descriptor(layer, expert);

                ExpertCounterSnapshot snapshot;
                snapshot.access_attempts = descriptor.access_attempts;
                snapshot.access_failures = descriptor.access_failures;

                for (int i = 0; i < 3; ++i) {
                    snapshot.requested_hit[i] = descriptor.requested_hit[i];
                    snapshot.requested_miss[i] = descriptor.requested_miss[i];
                    snapshot.served_from[i] = descriptor.served_from[i];
                    snapshot.acquire_samples[i] = descriptor.acquire_samples[i];
                    snapshot.acquire_latency_us[i] = descriptor.acquire_latency_us[i];
                }
                snapshot.acquire_latency_us_total =
                    descriptor.acquire_latency_us_total;

                for (int i = 0; i < 5; ++i) {
                    snapshot.migration_count[i] = descriptor.mig_n[i];
                    snapshot.migration_failures[i] = descriptor.mig_fail[i];
                    snapshot.migration_bytes[i] = descriptor.mig_bytes[i];
                    snapshot.migration_latency_us[i] = descriptor.mig_us[i];
                }

                previous_counters_[{layer, expert}] = snapshot;
            }
        }
    }

    OptimizerReport observe(
        const etf::TelemetrySnapshot& telemetry,
        const ExpertStore& store,
        int n_layer,
        int n_expert
    ) {
        OptimizerReport report;
        report.first_prompt_id = telemetry.first_prompt_id;
        report.last_prompt_id = telemetry.last_prompt_id;
        report.completed_prompts = telemetry.completed_prompts;
        report.dropped_events = telemetry.dropped_events;

        const std::string config_error = validate_config();
        if (!config_error.empty()) {
            report.status = "Invalid optimizer configuration: " + config_error;
            return report;
        }

        if (n_layer <= 0 || n_expert <= 0) {
            report.status = "Invalid model dimensions";
            return report;
        }

        using Key = std::pair<int, int>;

        std::map<Key, std::pair<uint64_t, double>> activations;

        for (const auto& event : telemetry.activations) {
            if (event.layer < 0 || event.layer >= n_layer ||
                event.expert_id < 0 || event.expert_id >= n_expert) {
                continue;
            }

            auto& entry = activations[{event.layer, event.expert_id}];
            ++entry.first;
            entry.second += event.routing_weight;
        }

        report.demand.reserve(
            static_cast<size_t>(n_layer) * static_cast<size_t>(n_expert)
        );

        for (int layer = 0; layer < n_layer; ++layer) {
            for (int expert = 0; expert < n_expert; ++expert) {
                const Key key{layer, expert};
                const auto descriptor = store.descriptor(layer, expert);

                ExpertCounterSnapshot current;
                current.access_attempts = descriptor.access_attempts;
                current.access_failures = descriptor.access_failures;

                for (int i = 0; i < 3; ++i) {
                    current.requested_hit[i] = descriptor.requested_hit[i];
                    current.requested_miss[i] = descriptor.requested_miss[i];
                    current.served_from[i] = descriptor.served_from[i];
                    current.acquire_samples[i] = descriptor.acquire_samples[i];
                    current.acquire_latency_us[i] = descriptor.acquire_latency_us[i];
                }
                current.acquire_latency_us_total =
                    descriptor.acquire_latency_us_total;

                for (int i = 0; i < 5; ++i) {
                    current.migration_count[i] = descriptor.mig_n[i];
                    current.migration_failures[i] = descriptor.mig_fail[i];
                    current.migration_bytes[i] = descriptor.mig_bytes[i];
                    current.migration_latency_us[i] = descriptor.mig_us[i];
                }

                const auto previous_it = previous_counters_.find(key);
                const ExpertCounterSnapshot* previous =
                    previous_it == previous_counters_.end()
                        ? nullptr
                        : &previous_it->second;

                ExpertDemand demand;
                demand.layer = layer;
                demand.expert_id = expert;
                demand.current_residency = descriptor.state;
                demand.footprint = descriptor.footprint;
                demand.pin_count = descriptor.pin_count;

                for (int path = 0; path < 5; ++path) {
                    if (descriptor.mig_n[path] > 0 &&
                        std::isfinite(descriptor.mig_us[path]) &&
                        descriptor.mig_us[path] >= 0.0) {
                        demand.migration_historical_average_latency_us[path] =
                            descriptor.mig_us[path] /
                            static_cast<double>(descriptor.mig_n[path]);
                        demand.migration_historical_latency_available[path] =
                            std::isfinite(
                                demand.migration_historical_average_latency_us[path]);
                    }
                }

                for (int tier = 0; tier < 3; ++tier) {
                    if (descriptor.acquire_samples[tier] > 0 &&
                        std::isfinite(descriptor.acquire_latency_us[tier]) &&
                        descriptor.acquire_latency_us[tier] >= 0.0) {
                        demand.acquire_historical_average_latency_us[tier] =
                            descriptor.acquire_latency_us[tier] /
                            static_cast<double>(descriptor.acquire_samples[tier]);
                        demand.acquire_historical_latency_available[tier] =
                            std::isfinite(
                                demand.acquire_historical_average_latency_us[tier]);
                    }
                }

                const auto activation_it = activations.find(key);
                if (activation_it != activations.end()) {
                    demand.activation_count = activation_it->second.first;
                    demand.routing_weight_sum = activation_it->second.second;
                }

                demand.access_delta = delta(
                    current.access_attempts,
                    previous ? previous->access_attempts : 0
                );
                demand.failure_delta = delta(
                    current.access_failures,
                    previous ? previous->access_failures : 0
                );

                for (int i = 0; i < 3; ++i) {
                    demand.requested_hit_delta[i] = delta(
                        current.requested_hit[i],
                        previous ? previous->requested_hit[i] : 0
                    );
                    demand.requested_miss_delta[i] = delta(
                        current.requested_miss[i],
                        previous ? previous->requested_miss[i] : 0
                    );
                    demand.served_delta[i] = delta(
                        current.served_from[i],
                        previous ? previous->served_from[i] : 0
                    );

                    demand.acquire_sample_delta[i] = delta(
                        current.acquire_samples[i],
                        previous ? previous->acquire_samples[i] : 0
                    );
                    demand.acquire_latency_delta_us[i] = delta_double(
                        current.acquire_latency_us[i],
                        previous ? previous->acquire_latency_us[i] : 0.0
                    );

                    if (demand.acquire_sample_delta[i] > 0 &&
                        std::isfinite(demand.acquire_latency_delta_us[i]) &&
                        demand.acquire_latency_delta_us[i] >= 0.0) {
                        demand.acquire_average_latency_us[i] =
                            demand.acquire_latency_delta_us[i] /
                            static_cast<double>(demand.acquire_sample_delta[i]);
                        demand.acquire_latency_available[i] =
                            std::isfinite(demand.acquire_average_latency_us[i]);
                    }
                }

                for (int i = 0; i < 5; ++i) {
                    demand.migration_count_delta[i] = delta(
                        current.migration_count[i],
                        previous ? previous->migration_count[i] : 0
                    );
                    demand.migration_failure_delta[i] = delta(
                        current.migration_failures[i],
                        previous ? previous->migration_failures[i] : 0
                    );
                    demand.migration_bytes_delta[i] = delta(
                        current.migration_bytes[i],
                        previous ? previous->migration_bytes[i] : 0
                    );
                    demand.migration_latency_delta_us[i] = delta_double(
                        current.migration_latency_us[i],
                        previous ? previous->migration_latency_us[i] : 0.0
                    );
                }

                previous_counters_[key] = current;
                report.demand.push_back(std::move(demand));
            }
        }

        // Normalize activation demand within each layer so layers with
        // different routing volumes remain comparable.
        std::map<int, uint64_t> max_activation_by_layer;
        for (const auto& demand : report.demand) {
            auto& maximum = max_activation_by_layer[demand.layer];
            maximum = std::max(maximum, demand.activation_count);
        }

        for (auto& demand : report.demand) {
            const uint64_t layer_max = max_activation_by_layer[demand.layer];

            const double activation_score = layer_max == 0
                ? 0.0
                : static_cast<double>(demand.activation_count) /
                  static_cast<double>(layer_max);

            // EMA history is updated once per completed ETF window.
            const auto history_key =
                std::make_pair(demand.layer, demand.expert_id);
            double& history = activation_history_[history_key];
            history = config_.activation_history_decay * history +
                (1.0 - config_.activation_history_decay) * activation_score;
            demand.historical_activation_score = history;

            if (layer_max != 0 &&
                activation_score >= config_.hot_threshold) {
                demand.classification = DemandClass::HOT;
            } else if (layer_max != 0 &&
                       activation_score >= config_.warm_threshold) {
                demand.classification = DemandClass::WARM;
            } else {
                demand.classification = DemandClass::COLD;
            }

            const double mean_routing_weight =
                demand.activation_count == 0
                    ? 0.0
                    : demand.routing_weight_sum /
                      static_cast<double>(demand.activation_count);

            uint64_t requested_hits = 0;
            uint64_t requested_misses = 0;
            for (int tier = 0; tier < 3; ++tier) {
                requested_hits += demand.requested_hit_delta[tier];
                requested_misses += demand.requested_miss_delta[tier];
            }

            const uint64_t requested_total = requested_hits + requested_misses;
            const double miss_rate = requested_total == 0
                ? 0.0
                : static_cast<double>(requested_misses) /
                  static_cast<double>(requested_total);

            const double failure_rate = demand.access_delta == 0
                ? 0.0
                : std::min(1.0,
                    static_cast<double>(demand.failure_delta) /
                    static_cast<double>(demand.access_delta));

            demand.priority =
                config_.activation_weight * activation_score +
                config_.routing_weight * mean_routing_weight +
                config_.miss_weight * miss_rate -
                config_.failure_penalty * failure_rate;

            if (!std::isfinite(demand.priority)) {
                demand.priority = 0.0;
            }
        }

        report.experts_observed = report.demand.size();

        build_placement_recommendations(report, store);

        report.status = telemetry.dropped_events == 0
            ? "Observation collected"
            : "Observation collected with dropped ETF events";

        return report;
    }

    double historical_activation(int layer, int expert_id) const {
        const auto it = activation_history_.find({layer, expert_id});
        return it == activation_history_.end() ? 0.0 : it->second;
    }

#ifdef EAHMMF_TESTING
    void build_placement_recommendations_for_testing(
        OptimizerReport& report,
        const ExpertStore& store
    ) const {
        build_placement_recommendations(report, store);
    }
#endif

private:
    static size_t available_capacity(size_t total, size_t used) {
        if (total == SIZE_MAX) return SIZE_MAX;
        return used >= total ? 0 : total - used;
    }

    void build_placement_recommendations(
        OptimizerReport& report,
        const ExpertStore& store
    ) const {
        report.placements.clear();
        report.recommendations = 0;

        const auto config = store.config();
        const bool cuda = store.has_cuda_device();
        const size_t ram_available =
            available_capacity(store.ram_total(), store.ram_used());
        const size_t vram_available =
            available_capacity(store.vram_total(), store.vram_used());

        const size_t physical_vram_free =
            cuda ? store.device_free_memory() : 0;
        const size_t physical_vram_available =
            physical_vram_free > config.vram_reserve
                ? physical_vram_free - config.vram_reserve
                : 0;

        const bool store_has_pins = store.stats().pinned_now != 0;

        // Virtual capacity: planning reserves space without allocating memory.
        PlacementCapacity capacity{
            ram_available, vram_available,
            physical_vram_available, cuda
        };

        // Preserve report.demand order; evaluate candidates by shared priority ordering.
        const std::vector<size_t> demand_order =
            priority_order(report.demand);

        for (const size_t demand_index : demand_order) {
            const auto& demand = report.demand[demand_index];
            PlacementRecommendation rec;
            rec.layer = demand.layer;
            rec.expert_id = demand.expert_id;
            rec.current = demand.current_residency;
            rec.target = demand.current_residency;
            rec.priority = demand.priority;
            rec.footprint = demand.footprint;

            if (demand.current_residency != Residency::SSD &&
                demand.current_residency != Residency::RAM &&
                demand.current_residency != Residency::VRAM) {
                continue;
            }

            switch (demand.classification) {
                case DemandClass::HOT: {
                    const bool vram_fits =
                        capacity.cuda_available &&
                        capacity.vram_remaining >= demand.footprint &&
                        capacity.physical_vram_remaining >= demand.footprint;
                    rec.target = vram_fits ? Residency::VRAM : Residency::RAM;
                    break;
                }
                case DemandClass::WARM:
                    rec.target = Residency::RAM;
                    break;
                case DemandClass::COLD:
                    rec.target = Residency::SSD;
                    break;
            }

            if (rec.target == rec.current) continue;

            auto reject = [&](const std::string& reason) {
                rec.reason = reason;
                rec.actionable = false;
                report.placements.push_back(rec);
            };

            if (demand.pin_count != 0) {
                reject("Expert is pinned");
                continue;
            }

            if (store_has_pins) {
                reject("Store has active pins; defer placement planning");
                continue;
            }

            if (demand.footprint == 0) {
                reject("Expert footprint is zero or unavailable");
                continue;
            }

            // Check all required capacity before any virtual reservation.
            if (!capacity.can_reserve(
                    rec.current, rec.target, demand.footprint)) {
                const bool needs_ram =
                    rec.target == Residency::RAM ||
                    (rec.current == Residency::SSD &&
                     rec.target == Residency::VRAM);

                if (needs_ram && capacity.ram_remaining < demand.footprint) {
                    reject("Insufficient RAM capacity");
                    continue;
                }
                if (rec.target == Residency::VRAM) {
                    if (!cuda) {
                        reject("A real CUDA device is required for VRAM placement");
                        continue;
                    }
                    if (capacity.vram_remaining < demand.footprint) {
                        reject("Insufficient logical VRAM capacity");
                        continue;
                    }
                    if (capacity.physical_vram_remaining < demand.footprint) {
                        reject("Insufficient physical free VRAM after reserve");
                        continue;
                    }
                }
                reject("Insufficient placement capacity");
                continue;
            }

            const int current_tier = static_cast<int>(rec.current);
            const int target_tier = static_cast<int>(rec.target);

            if (current_tier < 0 || current_tier >= 3 ||
                target_tier < 0 || target_tier >= 3) {
                reject("Unsupported residency transition");
                continue;
            }

            const bool current_latency_available =
                demand.acquire_historical_latency_available[current_tier];
            const bool target_latency_available =
                demand.acquire_historical_latency_available[target_tier];

            if (!current_latency_available || !target_latency_available) {
                if (config_.require_measured_acquire_benefit) {
                    reject("Historical acquisition latency missing for current or target tier");
                    continue;
                }
                reject("Unmeasured acquisition benefit is not estimated by this planner");
                continue;
            }

            const double benefit_per_acquire =
                demand.acquire_historical_average_latency_us[current_tier] -
                demand.acquire_historical_average_latency_us[target_tier];

            if (!std::isfinite(benefit_per_acquire) ||
                benefit_per_acquire <= config_.minimum_latency_benefit_us) {
                reject("Target tier has no sufficient measured latency benefit");
                continue;
            }

            const uint64_t observed_current_serves =
                demand.served_delta[current_tier];

            if (observed_current_serves == 0) {
                reject("No successful acquisitions observed from current tier in this window");
                continue;
            }

            rec.estimated_benefit =
                benefit_per_acquire *
                static_cast<double>(observed_current_serves);

            // Map a one-leg or two-leg transition to the measured path
            // counters. SSD -> VRAM is SSD -> RAM -> VRAM.
            int paths[2] = {-1, -1};
            int path_count = 0;

            if (rec.current == Residency::SSD &&
                rec.target == Residency::RAM) {
                paths[path_count++] = SSD_RAM;
            } else if (rec.current == Residency::RAM &&
                       rec.target == Residency::VRAM) {
                paths[path_count++] = RAM_VRAM;
            } else if (rec.current == Residency::VRAM &&
                       rec.target == Residency::RAM) {
                paths[path_count++] = VRAM_RAM;
            } else if (rec.current == Residency::RAM &&
                       rec.target == Residency::SSD) {
                paths[path_count++] = RAM_SSD;
            } else if (rec.current == Residency::VRAM &&
                       rec.target == Residency::SSD) {
                paths[path_count++] = VRAM_SSD;
            } else if (rec.current == Residency::SSD &&
                       rec.target == Residency::VRAM) {
                paths[path_count++] = SSD_RAM;
                paths[path_count++] = RAM_VRAM;
            } else {
                reject("Unsupported migration path");
                continue;
            }

            bool migration_cost_available = true;
            double migration_cost_us = 0.0;

            for (int i = 0; i < path_count; ++i) {
                const int path = paths[i];
                if (!demand.migration_historical_latency_available[path]) {
                    migration_cost_available = false;
                    break;
                }
                migration_cost_us +=
                    demand.migration_historical_average_latency_us[path];
            }

            if (!migration_cost_available ||
                !std::isfinite(migration_cost_us) ||
                migration_cost_us < 0.0) {
                if (config_.require_measured_migration_cost) {
                    reject("Historical migration cost missing for one or more path legs");
                    continue;
                }
                reject("Unmeasured migration cost is not estimated by this planner");
                continue;
            }

            rec.estimated_cost_us = migration_cost_us;
            rec.cost_estimate_available = true;

            if (demand.priority < config_.minimum_priority) {
                reject("Priority is below configured minimum");
                continue;
            }

            if (rec.estimated_benefit <= 0.0 ||
                migration_cost_us >
                    rec.estimated_benefit *
                    config_.maximum_migration_cost_ratio) {
                reject("Estimated migration cost exceeds benefit threshold");
                continue;
            }

            // Reserve only after every eligibility and measurement check passes.
            if (!capacity.reserve(
                    rec.current, rec.target, demand.footprint)) {
                reject("Capacity reservation failed");
                continue;
            }

            rec.actionable = true;
            rec.reason = "Measured benefit, migration cost, and capacity checks passed";
            report.placements.push_back(std::move(rec));
        }

        report.recommendations = static_cast<size_t>(
            std::count_if(
                report.placements.begin(), report.placements.end(),
                [](const PlacementRecommendation& rec) {
                    return rec.actionable;
                }));
    }

    std::string validate_config() const {
        const double weights[] = {
            config_.activation_weight,
            config_.routing_weight,
            config_.miss_weight,
            config_.failure_penalty
        };

        for (double weight : weights) {
            if (!std::isfinite(weight) || weight < 0.0)
                return "scoring weights must be finite and non-negative";
        }

        if (!std::isfinite(config_.activation_history_decay) ||
            config_.activation_history_decay < 0.0 ||
            config_.activation_history_decay >= 1.0)
            return "activation_history_decay must be finite and in [0, 1)";

        if (!std::isfinite(config_.minimum_priority))
            return "minimum_priority must be finite";

        if (!std::isfinite(config_.minimum_priority_gain) ||
            config_.minimum_priority_gain < 0.0)
            return "minimum_priority_gain must be finite and non-negative";

        if (!std::isfinite(config_.minimum_latency_benefit_us) ||
            config_.minimum_latency_benefit_us < 0.0)
            return "minimum_latency_benefit_us must be finite and non-negative";

        if (!std::isfinite(config_.maximum_migration_cost_ratio) ||
            config_.maximum_migration_cost_ratio < 0.0)
            return "maximum_migration_cost_ratio must be finite and non-negative";

        if (!std::isfinite(config_.hot_threshold) ||
            !std::isfinite(config_.warm_threshold) ||
            config_.hot_threshold < 0.0 || config_.hot_threshold > 1.0 ||
            config_.warm_threshold < 0.0 || config_.warm_threshold > 1.0)
            return "demand thresholds must be finite and within [0, 1]";

        if (config_.warm_threshold > config_.hot_threshold)
            return "warm_threshold must not exceed hot_threshold";

        return {};
    }

    static uint64_t delta(uint64_t current, uint64_t previous) {
        // A counter reset is treated as a fresh counter epoch.
        return current >= previous ? current - previous : current;
    }

    static double delta_double(double current, double previous) {
        return current >= previous ? current - previous : current;
    }

    OptimizerConfig config_;
    std::map<std::pair<int, int>, ExpertCounterSnapshot> previous_counters_;
    std::map<std::pair<int, int>, double> activation_history_;
};

} // namespace eahmmf
