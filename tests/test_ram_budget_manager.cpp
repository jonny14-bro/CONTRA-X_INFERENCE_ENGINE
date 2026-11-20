#include <limits>
#include "../ram_budget_manager.h"

#include <cstdio>
#include <cstdint>

using namespace eahmmf::osmem;

static int checks = 0;
static int failures = 0;

#define CHECK(expr, label) do { \
    ++checks; \
    if (!(expr)) { \
        ++failures; \
        std::fprintf(stderr, "FAIL: %s\n", label); \
    } else { \
        std::fprintf(stderr, "PASS: %s\n", label); \
    } \
} while (0)

static constexpr uint64_t GiB = 1024ULL * 1024 * 1024;
static constexpr uint64_t MiB = 1024ULL * 1024;

static SystemMemorySnapshot system_snapshot(uint64_t available) {
    SystemMemorySnapshot s;
    s.total_bytes = 24 * GiB;
    s.available_bytes = available;
    s.meminfo_valid = true;
    return s;
}

static Pressure normal_pressure() {
    Pressure p;
    p.valid = true;
    p.some_avg10 = 0.0;
    p.full_avg10 = 0.0;
    return p;
}

int main() {
    std::puts("[RAM budget manager tests]");

    RamBudgetConfig cfg;
    cfg.hard_ceiling_bytes = 12 * GiB;
    cfg.system_reserve_bytes = 2 * GiB;
    cfg.cgroup_reserve_bytes = 512 * MiB;
    cfg.expansion_hysteresis_bytes = 512 * MiB;

    const CgroupMemorySnapshot unlimited_cgroup{};

    // 1. System headroom calculation.
    {
        RamBudgetManager manager(cfg);

        auto result = manager.observe(
            system_snapshot(8 * GiB),
            unlimited_cgroup,
            1 * GiB, normal_pressure());

        CHECK(result.valid, "valid system snapshot accepted");
        CHECK(result.has_recommendation,
              "first observation establishes recommendation");
        CHECK(result.system_headroom_bytes == 6 * GiB,
              "system safety reserve deducted");
        CHECK(result.recommended_capacity_bytes == 7 * GiB,
              "current usage plus safe headroom calculated");
    }

    // 2. Finite cgroup limit constrains the recommendation.
    {
        RamBudgetManager manager(cfg);

        CgroupMemorySnapshot cgroup;
        cgroup.valid = true;
        cgroup.has_memory_max = true;
        cgroup.memory_max_bytes = 12 * GiB;
        cgroup.current_bytes = 8 * GiB;

        auto result = manager.observe(
            system_snapshot(8 * GiB), cgroup, 1 * GiB, normal_pressure());

        CHECK(result.valid, "finite cgroup snapshot accepted");
        CHECK(result.cgroup_limited,
              "finite cgroup limit detected");
        CHECK(result.cgroup_headroom_bytes ==
                  (uint64_t)(3 * GiB + 512 * MiB),
              "cgroup reserve deducted from remaining limit");
        CHECK(result.recommended_capacity_bytes ==
                  (uint64_t)(4 * GiB + 512 * MiB),
              "cgroup limit constrains total RAM capacity");
    }

    // 3. Hard ceiling is never exceeded.
    {
        RamBudgetConfig limited = cfg;
        limited.hard_ceiling_bytes = 4 * GiB;

        RamBudgetManager manager(limited);

        auto result = manager.observe(
            system_snapshot(16 * GiB),
            unlimited_cgroup,
            1 * GiB, normal_pressure());

        CHECK(result.recommended_capacity_bytes == 4 * GiB,
              "hard ceiling enforced");
    }

    // 4. Reductions are immediate; small increases are suppressed.
    {
        RamBudgetManager manager(cfg);

        auto initial = manager.observe(
            system_snapshot(8 * GiB),
            unlimited_cgroup,
            1 * GiB, normal_pressure());

        auto reduced = manager.observe(
            system_snapshot(4 * GiB),
            unlimited_cgroup,
            1 * GiB, normal_pressure());

        CHECK(initial.recommended_capacity_bytes == 7 * GiB,
              "initial capacity established");
        CHECK(reduced.changed,
              "capacity reduction applied immediately");
        CHECK(reduced.recommended_capacity_bytes == 3 * GiB,
              "lower capacity becomes recommendation");

        auto small_increase = manager.observe(
            system_snapshot(4 * GiB + 256 * MiB),
            unlimited_cgroup,
            1 * GiB, normal_pressure());

        CHECK(!small_increase.changed,
              "small increase suppressed by hysteresis");
        CHECK(small_increase.recommended_capacity_bytes == 3 * GiB,
              "small increase retains previous recommendation");

        auto large_increase = manager.observe(
            system_snapshot(5 * GiB),
            unlimited_cgroup,
            1 * GiB, normal_pressure());

        CHECK(large_increase.changed,
              "sufficient increase passes hysteresis");
        CHECK(large_increase.recommended_capacity_bytes == 4 * GiB,
              "expanded recommendation updated");
    }

    // 5. Invalid telemetry retains the previous recommendation.
    {
        RamBudgetManager manager(cfg);

        manager.observe(
            system_snapshot(8 * GiB),
            unlimited_cgroup,
            1 * GiB, normal_pressure());

        SystemMemorySnapshot invalid;
        auto result = manager.observe(
            invalid, unlimited_cgroup, 1 * GiB, normal_pressure());

        CHECK(!result.valid,
              "invalid system telemetry rejected");
        CHECK(result.has_recommendation,
              "previous recommendation remains available");
        CHECK(result.recommended_capacity_bytes == 7 * GiB,
              "invalid telemetry does not overwrite recommendation");
    }

    // 6. Exhausted cgroup headroom safely yields zero extra headroom.
    {
        RamBudgetManager manager(cfg);

        CgroupMemorySnapshot cgroup;
        cgroup.valid = true;
        cgroup.has_memory_max = true;
        cgroup.memory_max_bytes = 4 * GiB;
        cgroup.current_bytes = 4 * GiB;

        auto result = manager.observe(
            system_snapshot(8 * GiB), cgroup, 1 * GiB, normal_pressure());

        CHECK(result.cgroup_headroom_bytes == 0,
              "exhausted cgroup has zero additional headroom");
        CHECK(result.recommended_capacity_bytes == 1 * GiB,
              "recommendation does not add unavailable cgroup headroom");
    }

    // 7. Pressure-aware headroom derating.
    {
        RamBudgetConfig pressure_cfg = cfg;
        pressure_cfg.hard_ceiling_bytes =
            std::numeric_limits<uint64_t>::max();

        // Normal: retain all 6 GiB of additional headroom.
        {
            RamBudgetManager manager(pressure_cfg);
            Pressure p;
            p.valid = true;
            p.some_avg10 = 1.0;
            p.full_avg10 = 0.5;

            auto result = manager.observe(
                system_snapshot(8 * GiB),
                unlimited_cgroup, 1 * GiB, p);

            CHECK(result.pressure_level == RamPressureLevel::Normal,
                  "low PSI classified as normal");
            CHECK(result.pressure_adjusted_headroom_bytes == 6 * GiB,
                  "normal pressure retains full headroom");
            CHECK(result.recommended_capacity_bytes == 7 * GiB,
                  "normal pressure recommendation");
        }

        // Elevated: retain half of additional headroom.
        {
            RamBudgetManager manager(pressure_cfg);
            Pressure p;
            p.valid = true;
            p.some_avg10 = 3.0;
            p.full_avg10 = 1.0;

            auto result = manager.observe(
                system_snapshot(8 * GiB),
                unlimited_cgroup, 1 * GiB, p);

            CHECK(result.pressure_level == RamPressureLevel::Elevated,
                  "moderate PSI classified as elevated");
            CHECK(result.pressure_adjusted_headroom_bytes == 3 * GiB,
                  "elevated pressure halves additional headroom");
            CHECK(result.recommended_capacity_bytes == 4 * GiB,
                  "elevated pressure recommendation");
        }

        // Full PSI participates in conservative classification.
        {
            RamBudgetManager manager(pressure_cfg);
            Pressure p;
            p.valid = true;
            p.some_avg10 = 1.0;
            p.full_avg10 = 6.0;

            auto result = manager.observe(
                system_snapshot(8 * GiB),
                unlimited_cgroup, 1 * GiB, p);

            CHECK(result.pressure_level == RamPressureLevel::Critical,
                  "full PSI participates in classification");
            CHECK(result.pressure_adjusted_headroom_bytes == 0,
                  "critical pressure removes additional headroom");
            CHECK(result.recommended_capacity_bytes == 1 * GiB,
                  "critical pressure retains current RAM usage");
        }

        // Missing PSI is unknown, not normal.
        {
            RamBudgetManager manager(pressure_cfg);
            Pressure p;

            auto result = manager.observe(
                system_snapshot(8 * GiB),
                unlimited_cgroup, 1 * GiB, p);

            CHECK(!result.pressure_valid,
                  "missing PSI is explicitly invalid");
            CHECK(result.pressure_level == RamPressureLevel::Unknown,
                  "missing PSI classified as unknown");
            CHECK(result.pressure_adjusted_headroom_bytes == 3 * GiB,
                  "unknown PSI uses conservative half-headroom");
            CHECK(result.recommended_capacity_bytes == 4 * GiB,
                  "unknown PSI recommendation");
        }
    }

    // 8. Invalid PSI numbers must not be treated as normal.
    {
        RamBudgetManager manager(cfg);
        Pressure p;
        p.valid = true;
        p.some_avg10 = std::numeric_limits<double>::quiet_NaN();
        p.full_avg10 = 0.0;

        auto result = manager.observe(
            system_snapshot(8 * GiB),
            unlimited_cgroup, 1 * GiB, p);

        CHECK(!result.pressure_valid,
              "non-finite PSI rejected");
        CHECK(result.pressure_level == RamPressureLevel::Unknown,
              "non-finite PSI classified as unknown");
    }

    std::printf(
        "\nRAM budget manager tests: %d checks, %d failures\n",
        checks, failures);

    return failures ? 1 : 0;
}
