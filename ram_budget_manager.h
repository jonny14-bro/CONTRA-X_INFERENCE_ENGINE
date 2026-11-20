#pragma once

#include "os_memory.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace eahmmf {
namespace osmem {

struct RamBudgetConfig {
    uint64_t hard_ceiling_bytes =
        std::numeric_limits<uint64_t>::max();

    uint64_t system_reserve_bytes =
        2ULL * 1024 * 1024 * 1024;

    uint64_t cgroup_reserve_bytes =
        512ULL * 1024 * 1024;

    // Capacity increases smaller than this are ignored.
    // Capacity reductions are applied immediately.
    uint64_t expansion_hysteresis_bytes =
        512ULL * 1024 * 1024;

    // Provisional PSI thresholds. PSI values are percentages.
    double elevated_pressure_percent = 2.0;
    double critical_pressure_percent = 5.0;

    double normal_headroom_factor = 1.0;
    double elevated_headroom_factor = 0.5;
    double critical_headroom_factor = 0.0;
    double unknown_headroom_factor = 0.5;
};

enum class RamPressureLevel {
    Unknown,
    Normal,
    Elevated,
    Critical
};

inline const char* ram_pressure_level_name(RamPressureLevel level) {
    switch (level) {
        case RamPressureLevel::Normal:   return "normal";
        case RamPressureLevel::Elevated: return "elevated";
        case RamPressureLevel::Critical: return "critical";
        default:                         return "unknown";
    }
}

struct RamBudgetDecision {
    bool valid = false;
    bool has_recommendation = false;
    bool changed = false;
    bool cgroup_limited = false;

    bool pressure_valid = false;
    RamPressureLevel pressure_level = RamPressureLevel::Unknown;
    double pressure_score_avg10 = 0.0;
    double pressure_headroom_factor = 0.5;

    uint64_t current_used_bytes = 0;
    uint64_t available_bytes = 0;
    uint64_t system_headroom_bytes = 0;
    uint64_t cgroup_headroom_bytes = 0;
    uint64_t pressure_adjusted_headroom_bytes = 0;
    uint64_t recommended_capacity_bytes = 0;
};

// Returns true when the configuration is safe to use in capacity math.
inline bool validate_ram_budget_config(
    const RamBudgetConfig& c, std::string* error = nullptr) {
    auto fail = [&](const char* m) {
        if (error) *error = m;
        return false;
    };
    const double thr[] = {c.elevated_pressure_percent,
                          c.critical_pressure_percent};
    for (double t : thr)
        if (!std::isfinite(t) || t < 0.0 || t > 100.0)
            return fail("PSI threshold must be finite and within [0,100]");
    if (c.critical_pressure_percent < c.elevated_pressure_percent)
        return fail("critical PSI threshold below elevated threshold");
    const double fac[] = {c.normal_headroom_factor,
                          c.elevated_headroom_factor,
                          c.critical_headroom_factor,
                          c.unknown_headroom_factor};
    for (double v : fac)
        if (!std::isfinite(v) || v < 0.0 || v > 1.0)
            return fail("headroom factor must be finite and within [0,1]");
    if (c.critical_headroom_factor > c.elevated_headroom_factor ||
        c.elevated_headroom_factor > c.normal_headroom_factor)
        return fail("headroom factors must not increase with pressure");
    return true;
}

class RamBudgetManager {
public:
    // An invalid configuration is rejected: the manager falls back to the
    // default configuration and reports the reason via config_error().
    explicit RamBudgetManager(
        const RamBudgetConfig& config = RamBudgetConfig()) {
        if (validate_ram_budget_config(config, &config_error_)) {
            config_ = config;
        } else {
            config_ = RamBudgetConfig();
            config_valid_ = false;
        }
    }

    bool config_valid() const { return config_valid_; }
    const std::string& config_error() const { return config_error_; }

    RamBudgetDecision observe(
        const SystemMemorySnapshot& system,
        const CgroupMemorySnapshot& cgroup,
        uint64_t current_used_bytes,
        const Pressure& pressure = Pressure{}) {

        RamBudgetDecision out;
        out.current_used_bytes = current_used_bytes;
        out.available_bytes = system.available_bytes;

        // Use the more conservative of PSI some/full avg10.
        // Invalid or non-finite PSI is explicitly classified as unknown.
        if (pressure.valid &&
            std::isfinite(pressure.some_avg10) &&
            std::isfinite(pressure.full_avg10) &&
            pressure.some_avg10 >= 0.0 &&
            pressure.full_avg10 >= 0.0) {
            out.pressure_valid = true;
            out.pressure_score_avg10 =
                std::max(pressure.some_avg10, pressure.full_avg10);

            if (out.pressure_score_avg10 >=
                config_.critical_pressure_percent) {
                out.pressure_level = RamPressureLevel::Critical;
                out.pressure_headroom_factor =
                    config_.critical_headroom_factor;
            } else if (out.pressure_score_avg10 >=
                       config_.elevated_pressure_percent) {
                out.pressure_level = RamPressureLevel::Elevated;
                out.pressure_headroom_factor =
                    config_.elevated_headroom_factor;
            } else {
                out.pressure_level = RamPressureLevel::Normal;
                out.pressure_headroom_factor =
                    config_.normal_headroom_factor;
            }
        } else {
            out.pressure_level = RamPressureLevel::Unknown;
            out.pressure_headroom_factor =
                config_.unknown_headroom_factor;
        }

        // Invalid system telemetry must never create a new recommendation.
        if (!system.meminfo_valid || system.total_bytes == 0)
            return retain_previous(out);

        out.valid = true;

        const uint64_t system_headroom =
            system.available_bytes > config_.system_reserve_bytes
                ? system.available_bytes - config_.system_reserve_bytes
                : 0;

        out.system_headroom_bytes = system_headroom;

        uint64_t effective_headroom = system_headroom;

        // Apply a finite cgroup memory.max constraint when available.
        if (cgroup.valid && cgroup.has_memory_max) {
            out.cgroup_limited = true;

            const uint64_t remaining =
                cgroup.memory_max_bytes > cgroup.current_bytes
                    ? cgroup.memory_max_bytes - cgroup.current_bytes
                    : 0;

            const uint64_t cgroup_headroom =
                remaining > config_.cgroup_reserve_bytes
                    ? remaining - config_.cgroup_reserve_bytes
                    : 0;

            out.cgroup_headroom_bytes = cgroup_headroom;

            effective_headroom = std::min(
                effective_headroom, cgroup_headroom);
        }

        // Pressure derates only additional headroom, never current usage.
        out.pressure_headroom_factor = std::clamp(
            out.pressure_headroom_factor, 0.0, 1.0);

        const long double adjusted =
            static_cast<long double>(effective_headroom) *
            static_cast<long double>(out.pressure_headroom_factor);

        out.pressure_adjusted_headroom_bytes =
            static_cast<uint64_t>(adjusted);

        uint64_t candidate = saturating_add(
            current_used_bytes,
            out.pressure_adjusted_headroom_bytes);

        candidate = std::min(
            candidate, config_.hard_ceiling_bytes);

        // First valid observation establishes the baseline.
        if (!has_previous_) {
            previous_capacity_ = candidate;
            has_previous_ = true;
            out.changed = true;
        } else if (candidate < previous_capacity_) {
            // Reduce immediately when available capacity falls.
            previous_capacity_ = candidate;
            out.changed = true;
        } else if (
            candidate - previous_capacity_ >=
            config_.expansion_hysteresis_bytes) {
            // Require a meaningful increase before expanding.
            previous_capacity_ = candidate;
            out.changed = true;
        }

        // A recommendation must never fall below RAM already occupied.
        // This can temporarily exceed hard_ceiling_bytes if the current
        // occupancy is already above that configured ceiling.
        if (previous_capacity_ < current_used_bytes) {
            previous_capacity_ = current_used_bytes;
            out.changed = true;
        }

        out.has_recommendation = true;
        out.recommended_capacity_bytes = previous_capacity_;
        return out;
    }

    bool has_recommendation() const {
        return has_previous_;
    }

    uint64_t last_recommendation_bytes() const {
        return has_previous_ ? previous_capacity_ : 0;
    }

private:
    static uint64_t saturating_add(uint64_t a, uint64_t b) {
        const uint64_t maximum =
            std::numeric_limits<uint64_t>::max();

        return b > maximum - a ? maximum : a + b;
    }

    RamBudgetDecision retain_previous(RamBudgetDecision out) const {
        if (has_previous_) {
            out.has_recommendation = true;
            out.recommended_capacity_bytes = previous_capacity_;
        }
        return out;
    }

    RamBudgetConfig config_;
    bool config_valid_ = true;
    std::string config_error_;
    uint64_t previous_capacity_ = 0;
    bool has_previous_ = false;
};

} // namespace osmem
} // namespace eahmmf
