#pragma once

// Linux OS memory observation and bounded page-locking primitives.
// C++17. Header-only. Does not own or modify model mappings.

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

namespace eahmmf {
namespace osmem {

struct Pressure {
    double some_avg10 = 0.0;
    double some_avg60 = 0.0;
    double some_avg300 = 0.0;
    uint64_t some_total_us = 0;

    double full_avg10 = 0.0;
    double full_avg60 = 0.0;
    double full_avg300 = 0.0;
    uint64_t full_total_us = 0;

    bool valid = false;
};

struct SystemMemorySnapshot {
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;
    uint64_t available_bytes = 0;
    uint64_t buffers_bytes = 0;
    uint64_t cached_bytes = 0;
    uint64_t swap_total_bytes = 0;
    uint64_t swap_free_bytes = 0;

    uint64_t page_faults = 0;
    uint64_t major_faults = 0;
    uint64_t swap_in_pages = 0;
    uint64_t swap_out_pages = 0;
    uint64_t page_scan_kswapd = 0;
    uint64_t page_scan_direct = 0;

    Pressure memory_pressure;
    Pressure io_pressure;

    long page_size = 0;
    bool meminfo_valid = false;
    bool vmstat_valid = false;
};

struct ProcessMemorySnapshot {
    pid_t pid = -1;
    uint64_t rss_bytes = 0;
    uint64_t anon_rss_bytes = 0;
    uint64_t file_rss_bytes = 0;
    uint64_t swap_bytes = 0;
    uint64_t major_faults = 0;
    uint64_t minor_faults = 0;
    uint32_t threads = 0;
    std::string name;
    bool valid = false;
};

struct CgroupMemorySnapshot {
    std::string path;

    uint64_t current_bytes = 0;
    uint64_t peak_bytes = 0;
    uint64_t swap_current_bytes = 0;
    uint64_t memory_min_bytes = 0;
    uint64_t memory_low_bytes = 0;
    uint64_t memory_high_bytes = 0;
    uint64_t memory_max_bytes = 0;
    uint64_t swap_max_bytes = 0;

    uint64_t oom_events = 0;
    uint64_t oom_kill_events = 0;
    uint64_t high_events = 0;
    uint64_t max_events = 0;

    bool has_memory_high = false;
    bool has_memory_max = false;
    bool has_swap_max = false;
    bool valid = false;
};

struct MmapResidencySnapshot {
    uintptr_t aligned_start = 0;
    size_t mapped_length = 0;
    size_t page_size = 0;
    size_t pages_total = 0;
    size_t pages_resident = 0;
    size_t bytes_resident_estimate = 0;
    uint64_t allocation_duration_ns = 0;
    uint64_t mincore_duration_ns = 0;
    uint64_t count_duration_ns = 0;
    double resident_fraction = 0.0;
    int error_number = 0;
    bool scan_performed = false;
    bool valid = false;
};

struct LockResult {
    bool ok = false;
    bool budget_rejected = false;
    bool overlap_rejected = false;
    int error_number = 0;
    size_t locked_bytes = 0;
};

struct AdmissionResult {
    bool allowed = false;
    bool system_headroom_ok = false;
    bool cgroup_headroom_ok = true;
    bool crosses_memory_high = false;
    uint64_t available_bytes = 0;
    uint64_t cgroup_remaining_bytes = 0;
    std::string reason;
};

inline bool read_text_file(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

inline bool read_u64_file(const std::string& path, uint64_t& value) {
    std::ifstream f(path);
    if (!f) return false;

    std::string s;
    if (!(f >> s) || s == "max") return false;

    try {
        size_t used = 0;
        unsigned long long v = std::stoull(s, &used);
        if (used != s.size()) return false;
        value = static_cast<uint64_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool read_meminfo_value(
    const std::map<std::string, uint64_t>& values,
    const char* key,
    uint64_t& bytes) {

    auto it = values.find(key);
    if (it == values.end()) return false;

    // Linux /proc/meminfo values are reported in KiB.
    if (it->second > UINT64_MAX / 1024) return false;
    bytes = it->second * 1024;
    return true;
}

inline bool parse_finite_nonneg_double(const char* s, double& out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || errno == ERANGE) return false;
    if (!std::isfinite(v) || v < 0.0) return false;
    out = v;
    return true;
}

inline bool parse_u64_strict(const char* s, uint64_t& out) {
    if (!s || !*s || *s == '-' || *s == '+') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE) return false;
    out = static_cast<uint64_t>(v);
    return true;
}

// Any malformed or non-finite field invalidates the whole result; partially
// parsed PSI is never reported as valid.
inline Pressure read_pressure_file(const std::string& path) {
    Pressure p;
    std::ifstream f(path);
    if (!f) return p;

    bool got_some = false, got_full = false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string kind;
        ss >> kind;
        if (kind != "some" && kind != "full") return Pressure{};

        double a10 = 0.0, a60 = 0.0, a300 = 0.0;
        uint64_t total = 0;
        bool h10 = false, h60 = false, h300 = false, htotal = false;

        std::string field;
        while (ss >> field) {
            if (field.rfind("avg10=", 0) == 0) {
                if (!parse_finite_nonneg_double(field.c_str() + 6, a10))
                    return Pressure{};
                h10 = true;
            } else if (field.rfind("avg60=", 0) == 0) {
                if (!parse_finite_nonneg_double(field.c_str() + 6, a60))
                    return Pressure{};
                h60 = true;
            } else if (field.rfind("avg300=", 0) == 0) {
                if (!parse_finite_nonneg_double(field.c_str() + 7, a300))
                    return Pressure{};
                h300 = true;
            } else if (field.rfind("total=", 0) == 0) {
                if (!parse_u64_strict(field.c_str() + 6, total))
                    return Pressure{};
                htotal = true;
            }
        }

        if (!(h10 && h60 && h300 && htotal)) return Pressure{};
        if (a10 > 100.0 || a60 > 100.0 || a300 > 100.0) return Pressure{};

        if (kind == "some") {
            p.some_avg10 = a10; p.some_avg60 = a60; p.some_avg300 = a300;
            p.some_total_us = total; got_some = true;
        } else {
            p.full_avg10 = a10; p.full_avg60 = a60; p.full_avg300 = a300;
            p.full_total_us = total; got_full = true;
        }
    }

    p.valid = got_some && got_full;
    if (!p.valid) return Pressure{};
    return p;
}

inline SystemMemorySnapshot read_system_memory() {
    SystemMemorySnapshot s;
    s.page_size = sysconf(_SC_PAGESIZE);

    std::ifstream mem("/proc/meminfo");
    if (mem) {
        std::map<std::string, uint64_t> values;
        std::string key;
        uint64_t value = 0;
        std::string unit;

        while (mem >> key >> value) {
            mem >> unit;
            if (!key.empty() && key.back() == ':')
                key.pop_back();
            values[key] = value;
        }

        s.meminfo_valid =
            read_meminfo_value(values, "MemTotal", s.total_bytes) &&
            read_meminfo_value(values, "MemAvailable", s.available_bytes);

        read_meminfo_value(values, "MemFree", s.free_bytes);
        read_meminfo_value(values, "Buffers", s.buffers_bytes);
        read_meminfo_value(values, "Cached", s.cached_bytes);
        read_meminfo_value(values, "SwapTotal", s.swap_total_bytes);
        read_meminfo_value(values, "SwapFree", s.swap_free_bytes);
    }

    std::ifstream vm("/proc/vmstat");
    if (vm) {
        std::map<std::string, uint64_t> values;
        std::string key;
        uint64_t value = 0;

        while (vm >> key >> value)
            values[key] = value;

        auto get = [&](const char* k, uint64_t& out) {
            auto it = values.find(k);
            if (it != values.end()) out = it->second;
        };

        get("pgfault", s.page_faults);
        get("pgmajfault", s.major_faults);
        get("pswpin", s.swap_in_pages);
        get("pswpout", s.swap_out_pages);
        get("pgscan_kswapd", s.page_scan_kswapd);
        get("pgscan_direct", s.page_scan_direct);
        s.vmstat_valid = true;
    }

    s.memory_pressure = read_pressure_file("/proc/pressure/memory");
    s.io_pressure = read_pressure_file("/proc/pressure/io");
    return s;
}

inline ProcessMemorySnapshot read_process_memory(pid_t pid) {
    ProcessMemorySnapshot p;
    p.pid = pid;

    std::ifstream f("/proc/" + std::to_string(pid) + "/status");
    if (!f) return p;

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;

        uint64_t value = 0;
        std::string unit;

        if (key == "Name:") {
            std::getline(ss, p.name);
            if (!p.name.empty() && p.name[0] == ' ')
                p.name.erase(0, 1);
        } else if (key == "VmRSS:" && ss >> value >> unit) {
            p.rss_bytes = value * 1024;
        } else if (key == "RssAnon:" && ss >> value >> unit) {
            p.anon_rss_bytes = value * 1024;
        } else if (key == "RssFile:" && ss >> value >> unit) {
            p.file_rss_bytes = value * 1024;
        } else if (key == "VmSwap:" && ss >> value >> unit) {
            p.swap_bytes = value * 1024;
        } else if (key == "Threads:" && ss >> value) {
            p.threads = static_cast<uint32_t>(value);
        }
    }

    // /proc/PID/stat: fields 10 and 12 are minor and major faults.
    // comm is parenthesized and may contain spaces, so parse after its final ')'.
    std::string stat_text;
    if (read_text_file("/proc/" + std::to_string(pid) + "/stat", stat_text)) {
        const size_t close = stat_text.rfind(')');
        if (close != std::string::npos && close + 2 < stat_text.size()) {
            std::istringstream ss(stat_text.substr(close + 2));
            std::vector<std::string> fields;
            std::string field;
            while (ss >> field) fields.push_back(field);

            // After comm: state is field 3; minflt is field 10; majflt is field 12.
            if (fields.size() > 10)
                p.minor_faults = std::strtoull(fields[7].c_str(), nullptr, 10);
            if (fields.size() > 10)
                p.major_faults = std::strtoull(fields[9].c_str(), nullptr, 10);
        }
    }

    p.valid = true;
    return p;
}

inline CgroupMemorySnapshot read_cgroup_memory(pid_t pid) {
    CgroupMemorySnapshot c;
    std::string contents;

    if (!read_text_file(
            "/proc/" + std::to_string(pid) + "/cgroup", contents))
        return c;

    std::istringstream lines(contents);
    std::string line;
    std::string relative;

    while (std::getline(lines, line)) {
        if (line.rfind("0::", 0) == 0) {
            relative = line.substr(3);
            break;
        }
    }

    if (relative.empty() || relative[0] != '/')
        return c;

    c.path = "/sys/fs/cgroup" + relative;

    const std::string& base = c.path;
    if (!read_u64_file(base + "/memory.current", c.current_bytes))
        return c;

    read_u64_file(base + "/memory.peak", c.peak_bytes);
    read_u64_file(base + "/memory.swap.current", c.swap_current_bytes);
    read_u64_file(base + "/memory.min", c.memory_min_bytes);
    read_u64_file(base + "/memory.low", c.memory_low_bytes);

    c.has_memory_high =
        read_u64_file(base + "/memory.high", c.memory_high_bytes);
    c.has_memory_max =
        read_u64_file(base + "/memory.max", c.memory_max_bytes);
    c.has_swap_max =
        read_u64_file(base + "/memory.swap.max", c.swap_max_bytes);

    std::string events;
    if (read_text_file(base + "/memory.events", events)) {
        std::istringstream ss(events);
        std::string key;
        uint64_t value = 0;
        while (ss >> key >> value) {
            if (key == "high") c.high_events = value;
            else if (key == "max") c.max_events = value;
            else if (key == "oom") c.oom_events = value;
            else if (key == "oom_kill") c.oom_kill_events = value;
        }
    }

    c.valid = true;
    return c;
}

// mincore() observes current page residency. It does not fault pages in
// intentionally and does not guarantee that resident pages remain resident.
inline MmapResidencySnapshot inspect_mmap_residency(
    const void* address, size_t length) {

    MmapResidencySnapshot out;
    out.scan_performed = true;
    const long ps = sysconf(_SC_PAGESIZE);
    if (!address || length == 0 || ps <= 0) {
        out.error_number = EINVAL;
        return out;
    }

    const uintptr_t page = static_cast<uintptr_t>(ps);
    const uintptr_t raw = reinterpret_cast<uintptr_t>(address);
    const uintptr_t start = raw - (raw % page);

    if (length > UINTPTR_MAX - raw) {
        out.error_number = EOVERFLOW;
        return out;
    }

    const uintptr_t raw_end = raw + length;
    if (raw_end > UINTPTR_MAX - (page - 1)) {
        out.error_number = EOVERFLOW;
        return out;
    }

    const uintptr_t end = ((raw_end + page - 1) / page) * page;
    const size_t span = static_cast<size_t>(end - start);
    const size_t count = span / static_cast<size_t>(ps);

    auto phase_start = std::chrono::steady_clock::now();
    std::vector<unsigned char> vec(count, 0);
    auto phase_end = std::chrono::steady_clock::now();
    out.allocation_duration_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            phase_end - phase_start).count());

    phase_start = std::chrono::steady_clock::now();
    const int mincore_result = mincore(
        reinterpret_cast<void*>(start), span, vec.data());
    const int mincore_error = errno;
    phase_end = std::chrono::steady_clock::now();
    out.mincore_duration_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            phase_end - phase_start).count());

    if (mincore_result != 0) {
        out.error_number = mincore_error;
        return out;
    }

    phase_start = std::chrono::steady_clock::now();
    for (unsigned char state : vec)
        if (state & 1) ++out.pages_resident;
    phase_end = std::chrono::steady_clock::now();
    out.count_duration_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            phase_end - phase_start).count());

    out.aligned_start = start;
    out.mapped_length = span;
    out.page_size = static_cast<size_t>(ps);
    out.pages_total = count;
    out.bytes_resident_estimate =
        out.pages_resident * static_cast<size_t>(ps);
    out.resident_fraction =
        count ? static_cast<double>(out.pages_resident) / count : 0.0;
    out.valid = true;
    return out;
}

// Bounded lock manager for non-overlapping ranges.
// Callers must keep allocations alive until unlock() or manager destruction.
// Overlapping ranges are rejected to prevent incorrect munlock accounting.
class LockedMemoryBudget {
public:
    explicit LockedMemoryBudget(size_t limit_bytes)
        : limit_bytes_(limit_bytes) {}

    LockedMemoryBudget(const LockedMemoryBudget&) = delete;
    LockedMemoryBudget& operator=(const LockedMemoryBudget&) = delete;

    ~LockedMemoryBudget() {
        std::lock_guard<std::mutex> guard(mu_);
        for (const auto& item : ranges_)
            munlock(reinterpret_cast<void*>(item.first), item.second);
        ranges_.clear();
        used_bytes_ = 0;
    }

    LockResult lock(void* address, size_t length) {
        LockResult result;

        const long ps = sysconf(_SC_PAGESIZE);
        if (!address || length == 0 || ps <= 0) {
            result.error_number = EINVAL;
            return result;
        }

        const uintptr_t page = static_cast<uintptr_t>(ps);
        const uintptr_t raw = reinterpret_cast<uintptr_t>(address);
        const uintptr_t start = raw - (raw % page);

        if (length > UINTPTR_MAX - raw) {
            result.error_number = EOVERFLOW;
            return result;
        }

        const uintptr_t raw_end = raw + length;
        if (raw_end > UINTPTR_MAX - (page - 1)) {
            result.error_number = EOVERFLOW;
            return result;
        }

        const uintptr_t end = ((raw_end + page - 1) / page) * page;
        const size_t bytes = static_cast<size_t>(end - start);

        std::lock_guard<std::mutex> guard(mu_);

        if (bytes > limit_bytes_ || used_bytes_ > limit_bytes_ - bytes) {
            result.budget_rejected = true;
            return result;
        }

        auto next = ranges_.lower_bound(start);

        if (next != ranges_.end() && next->first < end) {
            result.overlap_rejected = true;
            return result;
        }

        if (next != ranges_.begin()) {
            auto prev = std::prev(next);
            const uintptr_t prev_end = prev->first + prev->second;
            if (prev_end > start) {
                result.overlap_rejected = true;
                return result;
            }
        }

        // Reserve the bookkeeping node BEFORE locking so a failed allocation
        // can never leave a successfully mlock()ed range untracked.
        std::map<uintptr_t, size_t>::iterator slot;
        try {
            slot = ranges_.emplace_hint(next, start, bytes);
        } catch (...) {
            result.error_number = ENOMEM;
            return result;
        }

        if (mlock(reinterpret_cast<void*>(start), bytes) != 0) {
            const int saved = errno;
            ranges_.erase(slot);
            result.error_number = saved;
            return result;
        }

        used_bytes_ += bytes;

        result.ok = true;
        result.locked_bytes = bytes;
        return result;
    }

    bool unlock(void* address) {
        if (!address) return false;

        const long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0) return false;

        const uintptr_t page = static_cast<uintptr_t>(ps);
        const uintptr_t raw = reinterpret_cast<uintptr_t>(address);
        const uintptr_t start = raw - (raw % page);

        std::lock_guard<std::mutex> guard(mu_);
        auto it = ranges_.find(start);
        if (it == ranges_.end()) return false;

        if (munlock(reinterpret_cast<void*>(it->first), it->second) != 0)
            return false;

        used_bytes_ -= it->second;
        ranges_.erase(it);
        return true;
    }

    size_t used_bytes() const {
        std::lock_guard<std::mutex> guard(mu_);
        return used_bytes_;
    }

    size_t limit_bytes() const {
        return limit_bytes_;
    }

    bool set_limit(size_t new_limit) {
        std::lock_guard<std::mutex> guard(mu_);
        if (new_limit < used_bytes_) return false;
        limit_bytes_ = new_limit;
        return true;
    }

private:
    mutable std::mutex mu_;
    size_t limit_bytes_ = 0;
    size_t used_bytes_ = 0;
    std::map<uintptr_t, size_t> ranges_;
};

inline uint64_t memlock_soft_limit_bytes() {
    struct rlimit lim {};
    if (getrlimit(RLIMIT_MEMLOCK, &lim) != 0)
        return 0;
    if (lim.rlim_cur == RLIM_INFINITY)
        return UINT64_MAX;
    return static_cast<uint64_t>(lim.rlim_cur);
}

inline AdmissionResult check_admission(
    const SystemMemorySnapshot& system,
    const CgroupMemorySnapshot& cgroup,
    uint64_t requested_bytes,
    uint64_t system_reserve_bytes,
    uint64_t cgroup_reserve_bytes = 0) {

    AdmissionResult out;
    out.available_bytes = system.available_bytes;

    if (!system.meminfo_valid) {
        out.reason = "system memory snapshot unavailable";
        return out;
    }

    if (requested_bytes > UINT64_MAX - system_reserve_bytes ||
        system.available_bytes < requested_bytes + system_reserve_bytes) {
        out.reason = "insufficient system MemAvailable headroom";
        return out;
    }

    out.system_headroom_ok = true;

    if (cgroup.valid && cgroup.has_memory_max) {
        const uint64_t max_bytes = cgroup.memory_max_bytes;

        if (cgroup.current_bytes >= max_bytes ||
            requested_bytes > max_bytes - cgroup.current_bytes ||
            cgroup_reserve_bytes >
                max_bytes - cgroup.current_bytes - requested_bytes) {
            out.cgroup_headroom_ok = false;
            out.reason = "insufficient cgroup memory.max headroom";
            return out;
        }

        out.cgroup_remaining_bytes = max_bytes - cgroup.current_bytes;

        if (cgroup.has_memory_high &&
            cgroup.current_bytes < cgroup.memory_high_bytes &&
            requested_bytes >
                cgroup.memory_high_bytes - cgroup.current_bytes) {
            out.crosses_memory_high = true;
        }
    }

    out.allowed = true;
    out.reason = out.crosses_memory_high
        ? "allowed, but projected usage crosses memory.high"
        : "admitted";
    return out;
}

} // namespace osmem
} // namespace eahmmf
