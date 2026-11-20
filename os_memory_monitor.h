#pragma once

#include "os_memory.h"
#include "ram_budget_manager.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>

namespace eahmmf {
namespace osmem {

struct StageSnapshot {
    SystemMemorySnapshot system;
    ProcessMemorySnapshot process;
    CgroupMemorySnapshot cgroup;
    MmapResidencySnapshot model_mmap;

    uint64_t eahmmf_ram_bytes = 0;
    uint64_t eahmmf_vram_bytes = 0;
    RamBudgetDecision ram_budget;
    uint64_t monotonic_ns = 0;
    uint64_t capture_duration_ns = 0;
    uint64_t system_capture_ns = 0;
    uint64_t process_capture_ns = 0;
    uint64_t cgroup_capture_ns = 0;
};

inline StageSnapshot capture_stage(
    const void* model_mapping,
    size_t model_mapping_bytes,
    uint64_t eahmmf_ram_bytes,
    uint64_t eahmmf_vram_bytes,
    bool inspect_model_mapping = true
) {
    StageSnapshot out;
    const auto capture_start = std::chrono::steady_clock::now();
    out.monotonic_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            capture_start.time_since_epoch()).count());

    const pid_t pid = getpid();

    auto phase_start = std::chrono::steady_clock::now();
    out.system = read_system_memory();
    auto phase_end = std::chrono::steady_clock::now();
    out.system_capture_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            phase_end - phase_start).count());

    phase_start = std::chrono::steady_clock::now();
    out.process = read_process_memory(pid);
    phase_end = std::chrono::steady_clock::now();
    out.process_capture_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            phase_end - phase_start).count());

    phase_start = std::chrono::steady_clock::now();
    out.cgroup = read_cgroup_memory(pid);
    phase_end = std::chrono::steady_clock::now();
    out.cgroup_capture_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            phase_end - phase_start).count());

    if (inspect_model_mapping && model_mapping &&
        model_mapping_bytes != 0) {
        out.model_mmap =
            inspect_mmap_residency(
                model_mapping, model_mapping_bytes);
    }

    out.eahmmf_ram_bytes = eahmmf_ram_bytes;
    out.eahmmf_vram_bytes = eahmmf_vram_bytes;

    const auto capture_end = std::chrono::steady_clock::now();
    out.capture_duration_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            capture_end - capture_start).count());

    return out;
}

inline void json_string(FILE* f, const std::string& value) {
    std::fputc('"', f);

    for (unsigned char c : value) {
        switch (c) {
            case '"':  std::fputs("\\\"", f); break;
            case '\\': std::fputs("\\\\", f); break;
            case '\b': std::fputs("\\b", f); break;
            case '\f': std::fputs("\\f", f); break;
            case '\n': std::fputs("\\n", f); break;
            case '\r': std::fputs("\\r", f); break;
            case '\t': std::fputs("\\t", f); break;
            default:
                if (c < 0x20)
                    std::fprintf(f, "\\u%04x", c);
                else
                    std::fputc(c, f);
        }
    }

    std::fputc('"', f);
}

// JSON has no NaN/Infinity: emit null for non-finite values.
inline std::string json_num(double v, int precision) {
    if (!std::isfinite(v)) return "null";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
    return buf;
}

inline void write_pressure(FILE* f, const Pressure& p) {
    std::fprintf(f,
        "{\"valid\":%s,"
        "\"some_avg10\":%s,"
        "\"some_avg60\":%s,"
        "\"some_avg300\":%s,"
        "\"some_total_us\":%llu,"
        "\"full_avg10\":%s,"
        "\"full_avg60\":%s,"
        "\"full_avg300\":%s,"
        "\"full_total_us\":%llu}",
        p.valid ? "true" : "false",
        json_num(p.some_avg10,6).c_str(), json_num(p.some_avg60,6).c_str(),
        json_num(p.some_avg300,6).c_str(),
        (unsigned long long)p.some_total_us,
        json_num(p.full_avg10,6).c_str(), json_num(p.full_avg60,6).c_str(),
        json_num(p.full_avg300,6).c_str(),
        (unsigned long long)p.full_total_us);
}

inline void write_stage(FILE* f, const StageSnapshot& s) {
    std::fprintf(f,
        "{\"monotonic_ns\":%llu,\"capture_duration_ns\":%llu,"
        "\"system_capture_ns\":%llu,"
        "\"process_capture_ns\":%llu,"
        "\"cgroup_capture_ns\":%llu,",
        (unsigned long long)s.monotonic_ns,
        (unsigned long long)s.capture_duration_ns,
        (unsigned long long)s.system_capture_ns,
        (unsigned long long)s.process_capture_ns,
        (unsigned long long)s.cgroup_capture_ns);

    const auto& m = s.system;
    std::fprintf(f,
        "\"system\":{"
        "\"meminfo_valid\":%s,"
        "\"vmstat_valid\":%s,"
        "\"page_size\":%ld,"
        "\"total_bytes\":%llu,"
        "\"free_bytes\":%llu,"
        "\"available_bytes\":%llu,"
        "\"buffers_bytes\":%llu,"
        "\"cached_bytes\":%llu,"
        "\"swap_total_bytes\":%llu,"
        "\"swap_free_bytes\":%llu,"
        "\"page_faults\":%llu,"
        "\"major_faults\":%llu,"
        "\"swap_in_pages\":%llu,"
        "\"swap_out_pages\":%llu,"
        "\"page_scan_kswapd\":%llu,"
        "\"page_scan_direct\":%llu,"
        "\"memory_pressure\":",
        m.meminfo_valid ? "true" : "false",
        m.vmstat_valid ? "true" : "false",
        m.page_size,
        (unsigned long long)m.total_bytes,
        (unsigned long long)m.free_bytes,
        (unsigned long long)m.available_bytes,
        (unsigned long long)m.buffers_bytes,
        (unsigned long long)m.cached_bytes,
        (unsigned long long)m.swap_total_bytes,
        (unsigned long long)m.swap_free_bytes,
        (unsigned long long)m.page_faults,
        (unsigned long long)m.major_faults,
        (unsigned long long)m.swap_in_pages,
        (unsigned long long)m.swap_out_pages,
        (unsigned long long)m.page_scan_kswapd,
        (unsigned long long)m.page_scan_direct);

    write_pressure(f, m.memory_pressure);
    std::fputs(",\"io_pressure\":", f);
    write_pressure(f, m.io_pressure);
    std::fputs("},", f);

    const auto& p = s.process;
    std::fprintf(f,
        "\"process\":{"
        "\"valid\":%s,"
        "\"pid\":%d,"
        "\"name\":",
        p.valid ? "true" : "false", (int)p.pid);
    json_string(f, p.name);

    std::fprintf(f,
        ",\"rss_bytes\":%llu,"
        "\"anon_rss_bytes\":%llu,"
        "\"file_rss_bytes\":%llu,"
        "\"swap_bytes\":%llu,"
        "\"major_faults\":%llu,"
        "\"minor_faults\":%llu,"
        "\"threads\":%u},",
        (unsigned long long)p.rss_bytes,
        (unsigned long long)p.anon_rss_bytes,
        (unsigned long long)p.file_rss_bytes,
        (unsigned long long)p.swap_bytes,
        (unsigned long long)p.major_faults,
        (unsigned long long)p.minor_faults,
        p.threads);

    const auto& c = s.cgroup;
    std::fprintf(f,
        "\"cgroup\":{\"valid\":%s,\"path\":",
        c.valid ? "true" : "false");
    json_string(f, c.path);

    std::fprintf(f,
        ",\"current_bytes\":%llu,"
        "\"peak_bytes\":%llu,"
        "\"swap_current_bytes\":%llu,"
        "\"memory_min_bytes\":%llu,"
        "\"memory_low_bytes\":%llu,"
        "\"oom_events\":%llu,"
        "\"oom_kill_events\":%llu,"
        "\"high_events\":%llu,"
        "\"max_events\":%llu,"
        "\"memory_high_bytes\":",
        (unsigned long long)c.current_bytes,
        (unsigned long long)c.peak_bytes,
        (unsigned long long)c.swap_current_bytes,
        (unsigned long long)c.memory_min_bytes,
        (unsigned long long)c.memory_low_bytes,
        (unsigned long long)c.oom_events,
        (unsigned long long)c.oom_kill_events,
        (unsigned long long)c.high_events,
        (unsigned long long)c.max_events);

    if (c.has_memory_high)
        std::fprintf(f, "%llu",
            (unsigned long long)c.memory_high_bytes);
    else
        std::fputs("null", f);

    std::fputs(",\"memory_max_bytes\":", f);
    if (c.has_memory_max)
        std::fprintf(f, "%llu",
            (unsigned long long)c.memory_max_bytes);
    else
        std::fputs("null", f);

    std::fputs(",\"swap_max_bytes\":", f);
    if (c.has_swap_max)
        std::fprintf(f, "%llu",
            (unsigned long long)c.swap_max_bytes);
    else
        std::fputs("null", f);

    std::fputs("},", f);

    const auto& r = s.model_mmap;
    std::fprintf(f,
        "\"model_mmap\":{"
        "\"scan_performed\":%s,"
        "\"valid\":%s,"
        "\"mapped_length\":%zu,"
        "\"page_size\":%zu,"
        "\"pages_total\":%zu,"
        "\"pages_resident\":%zu,"
        "\"bytes_resident_estimate\":%zu,"
        "\"resident_fraction\":%s,"
        "\"allocation_duration_ns\":%llu,"
        "\"mincore_duration_ns\":%llu,"
        "\"count_duration_ns\":%llu,"
        "\"error_number\":%d},",
        r.scan_performed ? "true" : "false",
        r.valid ? "true" : "false",
        r.mapped_length,
        r.page_size,
        r.pages_total,
        r.pages_resident,
        r.bytes_resident_estimate,
        json_num(r.resident_fraction, 8).c_str(),
        (unsigned long long)r.allocation_duration_ns,
        (unsigned long long)r.mincore_duration_ns,
        (unsigned long long)r.count_duration_ns,
        r.error_number);

    std::fprintf(f,
        "\"eahmmf\":{"
        "\"ram_bytes\":%llu,"
        "\"vram_bytes\":%llu},"
        "\"ram_budget\":{"
        "\"valid\":%s,"
        "\"has_recommendation\":%s,"
        "\"changed\":%s,"
        "\"cgroup_limited\":%s,"
        "\"pressure_valid\":%s,"
        "\"pressure_level\":\"%s\","
        "\"pressure_score_avg10\":%s,"
        "\"pressure_headroom_factor\":%s,"
        "\"current_used_bytes\":%llu,"
        "\"available_bytes\":%llu,"
        "\"system_headroom_bytes\":%llu,"
        "\"cgroup_headroom_bytes\":%llu,"
        "\"pressure_adjusted_headroom_bytes\":%llu,"
        "\"recommended_capacity_bytes\":%llu}}",
        (unsigned long long)s.eahmmf_ram_bytes,
        (unsigned long long)s.eahmmf_vram_bytes,
        s.ram_budget.valid ? "true" : "false",
        s.ram_budget.has_recommendation ? "true" : "false",
        s.ram_budget.changed ? "true" : "false",
        s.ram_budget.cgroup_limited ? "true" : "false",
        s.ram_budget.pressure_valid ? "true" : "false",
        ram_pressure_level_name(s.ram_budget.pressure_level),
        json_num(s.ram_budget.pressure_score_avg10, 6).c_str(),
        json_num(s.ram_budget.pressure_headroom_factor, 3).c_str(),
        (unsigned long long)s.ram_budget.current_used_bytes,
        (unsigned long long)s.ram_budget.available_bytes,
        (unsigned long long)s.ram_budget.system_headroom_bytes,
        (unsigned long long)s.ram_budget.cgroup_headroom_bytes,
        (unsigned long long)s.ram_budget.pressure_adjusted_headroom_bytes,
        (unsigned long long)s.ram_budget.recommended_capacity_bytes);
}

inline bool append_prompt_record(
    const std::string& output_path,
    uint64_t prompt_id,
    bool completed,
    const std::string& model_path,
    const StageSnapshot& s0,
    const StageSnapshot& s1,
    const StageSnapshot& s2,
    std::string* error = nullptr
) {
    FILE* f = nullptr;

    try {
        const std::filesystem::path output(output_path);

        if (!output.parent_path().empty())
            std::filesystem::create_directories(
                output.parent_path());

        f = std::fopen(output_path.c_str(), "a");

        if (!f) {
            if (error) *error = "fopen failed";
            return false;
        }

        std::fprintf(f,
            "{\"schema_version\":1,"
            "\"prompt_id\":%llu,"
            "\"completed\":%s,"
            "\"pid\":%d,"
            "\"model_path\":",
            (unsigned long long)prompt_id,
            completed ? "true" : "false",
            (int)getpid());

        json_string(f, model_path);

        std::fputs(",\"S0\":", f);
        write_stage(f, s0);

        std::fputs(",\"S1\":", f);
        write_stage(f, s1);

        std::fputs(",\"S2\":", f);
        write_stage(f, s2);

        std::fputs("}\n", f);

        const bool flushed = std::fflush(f) == 0;
        const bool closed = std::fclose(f) == 0;
        f = nullptr;

        if (!flushed || !closed) {
            if (error) *error = "flush or close failed";
            return false;
        }

        return true;
    } catch (const std::exception& ex) {
        if (f) std::fclose(f);
        if (error) *error = ex.what();
        return false;
    } catch (...) {
        if (f) std::fclose(f);
        if (error) *error = "unknown exception";
        return false;
    }
}

} // namespace osmem
} // namespace eahmmf
