#include "../os_memory_monitor.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    using namespace eahmmf::osmem;

    auto before = capture_stage(nullptr, 0, 0, 0);
    const auto middle = capture_stage(nullptr, 0, 0, 0);
    const auto after = capture_stage(nullptr, 0, 0, 0);
    // Deterministic pressure-aware serializer fixture.
    before.ram_budget.valid = true;
    before.ram_budget.has_recommendation = true;
    before.ram_budget.pressure_valid = true;
    before.ram_budget.pressure_level = RamPressureLevel::Elevated;
    before.ram_budget.pressure_score_avg10 = 3.25;
    before.ram_budget.pressure_headroom_factor = 0.5;
    before.ram_budget.current_used_bytes = 1000000;
    before.ram_budget.available_bytes = 8000000000ULL;
    before.ram_budget.system_headroom_bytes = 6000000000ULL;
    before.ram_budget.cgroup_headroom_bytes = 0;
    before.ram_budget.pressure_adjusted_headroom_bytes = 3000000000ULL;
    before.ram_budget.recommended_capacity_bytes = 3001000000ULL;


    std::string error;

    const std::string path =
        "/tmp/eahmmf_os_memory_monitor_test.jsonl";

    std::remove(path.c_str());

    if (!append_prompt_record(
            path, 12345, true, "test-model.gguf",
            before, middle, after, &error)) {
        std::fprintf(stderr, "FAIL: append: %s\n", error.c_str());
        return 1;
    }

    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        std::fprintf(stderr, "FAIL: output not created\n");
        return 1;
    }

    char buffer[32768] = {};
    const size_t count = std::fread(
        buffer, 1, sizeof(buffer) - 1, f);
    std::fclose(f);

    if (count == 0) {
        std::fprintf(stderr, "FAIL: output is empty\n");
        return 1;
    }

    const std::string json(buffer, count);

    const bool fields_ok =
        json.find("\"schema_version\":1") != std::string::npos &&
        json.find("\"prompt_id\":12345") != std::string::npos &&
        json.find("\"S0\"") != std::string::npos &&
        json.find("\"S1\"") != std::string::npos &&
        json.find("\"S2\"") != std::string::npos &&
        json.find("\"model_mmap\"") != std::string::npos &&
        json.find("\"scan_performed\":false") != std::string::npos &&
        json.find("\"cgroup\"") != std::string::npos &&
        json.find("\"ram_budget\"") != std::string::npos &&
        json.find("\"recommended_capacity_bytes\"") != std::string::npos &&
        json.find("\"system_headroom_bytes\"") != std::string::npos &&
        json.find("\"pressure_valid\":true") != std::string::npos &&
        json.find("\"pressure_level\":\"elevated\"") != std::string::npos &&
        json.find("\"pressure_score_avg10\":3.250000") != std::string::npos &&
        json.find("\"pressure_headroom_factor\":0.500") != std::string::npos &&
        json.find("\"pressure_adjusted_headroom_bytes\":3000000000") != std::string::npos &&
        json.find("\"recommended_capacity_bytes\":3001000000") != std::string::npos;

    if (!fields_ok) {
        std::fprintf(stderr, "FAIL: required JSON fields missing\n");
        return 1;
    }

    std::puts("PASS: OS memory monitor record written");
    std::puts("PASS: S0/S1/S2 fields present");
    std::puts("PASS: process, cgroup and mmap fields present");
    std::puts("PASS: RAM budget recommendation fields present");
    std::puts("PASS: pressure classification and adjustment fields present");
    std::printf("Output: %s\n", path.c_str());

    return 0;
}
