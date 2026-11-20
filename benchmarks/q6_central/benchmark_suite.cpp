#include "benchmark_runner.hpp"
#include "build_metadata.hpp"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>
#include <cstdio>
#include <sys/utsname.h>

namespace fs = std::filesystem;

struct TestSpec {
    std::string name;
    std::string executable;
};

static std::string status_name(TestStatus status) {
    switch (status) {
        case TestStatus::PASS: return "PASS";
        case TestStatus::FAIL: return "FAIL";
        case TestStatus::SKIP: return "SKIP";
    }
    return "UNKNOWN";
}

static std::string timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);
    return buffer;
}

static std::string json_escape(const std::string& value) {
    std::ostringstream out;

    for (unsigned char c : value) {
        switch (c) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    out << "\\u00"
                        << hex[(c >> 4) & 0xf]
                        << hex[c & 0xf];
                } else {
                    out << static_cast<char>(c);
                }
        }
    }

    return out.str();
}

static std::string architecture() {
#if defined(__x86_64__)
    return "x86_64";
#elif defined(__aarch64__)
    return "aarch64";
#elif defined(__i386__)
    return "i386";
#else
    return "unknown";
#endif
}

static std::string cpu_model() {
    std::ifstream input("/proc/cpuinfo");
    std::string line;

    while (std::getline(input, line)) {
        const std::string key = "model name";

        if (line.compare(0, key.size(), key) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos)
                return line.substr(colon + 2);
        }
    }

    return "unavailable";
}

static void write_json_string(
    std::ofstream& out,
    const std::string& value
) {
    out << '"' << json_escape(value) << '"';
}

static void write_json_report(
    const fs::path& path,
    const std::string& mode,
    const std::string& run_timestamp,
    const std::vector<BenchmarkResult>& results,
    size_t passed,
    size_t failed,
    size_t skipped
) {
    struct utsname system_info{};
    const bool uname_ok = uname(&system_info) == 0;

    double total_duration_ms = 0.0;

    for (const auto& result : results)
        total_duration_ms += result.duration_ms;

    std::ofstream out(path);

    if (!out)
        throw std::runtime_error("Could not create JSON report: " + path.string());

    out << "{\n";
    out << "  \"schema_version\": \"1.0\",\n";

    out << "  \"run\": {\n";
    out << "    \"mode\": ";
    write_json_string(out, mode);
    out << ",\n";

    out << "    \"timestamp\": ";
    write_json_string(out, run_timestamp);
    out << ",\n";

    out << "    \"os\": ";
    write_json_string(out, uname_ok ? system_info.sysname : "unknown");
    out << ",\n";

    out << "    \"kernel\": ";
    write_json_string(out, uname_ok ? system_info.release : "unknown");
    out << ",\n";

    out << "    \"machine\": ";
    write_json_string(out, uname_ok ? system_info.machine : "unknown");
    out << ",\n";

    out << "    \"architecture\": ";
    write_json_string(out, architecture());
    out << ",\n";

    out << "    \"cpu_model\": ";
    write_json_string(out, cpu_model());
    out << ",\n";

    out << "    \"compiler\": ";
    write_json_string(out, __VERSION__);
    out << ",\n";

    out << "    \"cpp_standard\": \"C++17\",\n";
    out << "    \"build_flags\": ";
    write_json_string(out, Q6_BUILD_FLAGS);
    out << ",\n";

    out << "    \"git_revision\": ";
    write_json_string(out, Q6_GIT_REVISION);
    out << ",\n";

    out << "    \"git_worktree\": ";
    write_json_string(out, Q6_GIT_STATE);
    out << "\n";
    out << "  },\n";

    out << "  \"summary\": {\n";
    out << "    \"total_tests\": " << results.size() << ",\n";
    out << "    \"passed\": " << passed << ",\n";
    out << "    \"failed\": " << failed << ",\n";
    out << "    \"skipped\": " << skipped << ",\n";
    out << "    \"total_duration_ms\": "
        << std::fixed << std::setprecision(3)
        << total_duration_ms << "\n";
    out << "  },\n";

    out << "  \"tests\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];

        out << "    {\n";
        out << "      \"name\": ";
        write_json_string(out, result.name);
        out << ",\n";

        out << "      \"status\": ";
        write_json_string(out, status_name(result.status));
        out << ",\n";

        out << "      \"exit_code\": " << result.exit_code << ",\n";
        out << "      \"duration_ms\": "
            << std::fixed << std::setprecision(3)
            << result.duration_ms << ",\n";

        out << "      \"stdout_log\": ";
        write_json_string(out, result.stdout_log);
        out << ",\n";

        out << "      \"stderr_log\": ";
        write_json_string(out, result.stderr_log);
        out << ",\n";

        out << "      \"metrics\": [\n";

        for (size_t j = 0; j < result.metrics.size(); ++j) {
            const auto& metric = result.metrics[j];

            out << "        {\n";
            out << "          \"name\": ";
            write_json_string(out, metric.name);
            out << ",\n";

            out << "          \"value\": ";

            if (metric.numeric)
                out << metric.value;
            else
                write_json_string(out, metric.value);

            out << ",\n";
            out << "          \"numeric\": "
                << (metric.numeric ? "true" : "false") << ",\n";

            out << "          \"unit\": ";
            write_json_string(out, metric.unit);
            out << ",\n";

            out << "          \"source_line\": ";
            write_json_string(out, metric.source_line);
            out << "\n";

            out << "        }";
            if (j + 1 < result.metrics.size()) out << ",";
            out << "\n";
        }

        out << "      ]\n";
        out << "    }";

        if (i + 1 < results.size()) out << ",";
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";

    if (!out)
        throw std::runtime_error("Failed while writing JSON report");
}


static std::string json_string_field(
    const std::string& json,
    const std::string& field
) {
    const std::regex pattern(
        "\"" + field + "\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\""
    );
    std::smatch match;
    if (std::regex_search(json, match, pattern))
        return match[1].str();
    return "unavailable";
}

static std::string json_number_field(
    const std::string& json,
    const std::string& field
) {
    const std::regex pattern(
        "\"" + field + "\"\\s*:\\s*([-+]?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
    );
    std::smatch match;
    if (std::regex_search(json, match, pattern))
        return match[1].str();
    return "unavailable";
}

static int count_json_status(
    const std::string& json,
    const std::string& status
) {
    const std::regex pattern(
        "\"status\"\\s*:\\s*\"" + status + "\""
    );

    return static_cast<int>(std::distance(
        std::sregex_iterator(json.begin(), json.end(), pattern),
        std::sregex_iterator()
    ));
}

static int count_json_metrics(const std::string& json) {
    const std::regex pattern("\"source_line\"\\s*:");

    return static_cast<int>(std::distance(
        std::sregex_iterator(json.begin(), json.end(), pattern),
        std::sregex_iterator()
    ));
}

static int inspect_report(const std::string& requested_run) {
    const fs::path reports_dir = "benchmarks/q6_central/reports";
    fs::path selected_report;

    if (!requested_run.empty()) {
        if (requested_run.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            ) != std::string::npos) {
            std::cerr << "Invalid run identifier.\n";
            return 2;
        }

        selected_report = reports_dir / requested_run / "report.json";

        if (!fs::exists(selected_report)) {
            std::cerr << "Report not found: "
                      << selected_report << '\n';
            return 2;
        }
    } else {
        bool found = false;
        fs::file_time_type newest_time{};

        if (fs::exists(reports_dir)) {
            for (const auto& entry : fs::directory_iterator(reports_dir)) {
                if (!entry.is_directory())
                    continue;

                const fs::path candidate =
                    entry.path() / "report.json";

                if (!fs::exists(candidate))
                    continue;

                const auto modified =
                    fs::last_write_time(candidate);

                if (!found || modified > newest_time) {
                    newest_time = modified;
                    selected_report = candidate;
                    found = true;
                }
            }
        }

        if (!found) {
            std::cerr << "No historical JSON reports found.\n";
            return 2;
        }
    }

    std::ifstream input(selected_report);

    if (!input) {
        std::cerr << "Could not open report.\n";
        return 2;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string json = buffer.str();

    if (json.empty() || json.front() != '{' ||
        json.find("\"schema_version\"") == std::string::npos) {
        std::cerr << "Invalid or unsupported JSON report.\n";
        return 2;
    }

    std::cout << "\n========== HISTORICAL BENCHMARK REPORT ==========\n";
    std::cout << "File: " << selected_report << '\n';

    std::cout << "\nRUN METADATA\n";

    for (const std::string field : {
        "mode", "timestamp", "os", "kernel", "machine",
        "architecture", "cpu_model", "compiler",
        "cpp_standard", "build_flags", "git_revision",
        "git_worktree"
    }) {
        std::cout << "  " << field << ": "
                  << json_string_field(json, field) << '\n';
    }

    std::cout << "\nSUMMARY\n";

    for (const std::string field : {
        "total_tests", "passed", "failed",
        "skipped", "total_duration_ms"
    }) {
        std::cout << "  " << field << ": "
                  << json_number_field(json, field) << '\n';
    }

    std::cout << "\nSTATUS COUNTS\n";
    std::cout << "  PASS: " << count_json_status(json, "PASS") << '\n';
    std::cout << "  FAIL: " << count_json_status(json, "FAIL") << '\n';
    std::cout << "  SKIP: " << count_json_status(json, "SKIP") << '\n';
    std::cout << "  Extracted metrics: "
              << count_json_metrics(json) << '\n';

    std::cout << "\nREPORT INSPECTION COMPLETE\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--help") {
        std::cout
            << "Q6 Central Benchmark Suite\n"
            << "Usage:\n"
            << "  --quick                    Core correctness and edge tests\n"
            << "  --stress                   Randomized stress tests\n"
            << "  --full                     All currently self-contained tests\n"
            << "  --report-only              Inspect latest historical report\n"
            << "  --report-only <run_id>     Inspect a specific historical report\n";
        return 0;
    }

    if (argc >= 2 && std::string(argv[1]) == "--report-only") {
        if (argc > 3) {
            std::cerr << "Usage: --report-only [run_id]\n";
            return 2;
        }

        return inspect_report(argc == 3 ? argv[2] : "");
    }

    if (argc != 2) {
        std::cerr << "Expected one benchmark mode. Use --help.\n";
        return 2;
    }

    const std::string mode = argv[1];

    const std::vector<TestSpec> quick_tests = {
        {"integrated_q6_kernel", "test_integrated_q6_kernel"},
        {"q8g16_checked", "test_q8g16_checked"},
        {"q8g16_finite_edges", "test_q8g16_finite_edges"},
        {"q8g16_nonfinite_edges", "test_q8g16_nonfinite_edges"},
        {"q8g16_subnormal_edges", "test_q8g16_subnormal_edges"},
        {"q8g16_rounding_modes", "test_q8g16_rounding_modes"},
        {"q8g16_ftz_daz", "test_q8g16_ftz_daz"}
    };

    const std::vector<TestSpec> stress_tests = {
        {"q8g16_v2_stress", "test_q8g16_v2_stress"}
    };

    const std::vector<TestSpec> diagnostic_tests = {
        {"diagnose_q8g16_rounding", "diagnose_q8g16_rounding"}
    };

    std::vector<TestSpec> selected;

    if (mode == "--quick") {
        selected = quick_tests;
    } else if (mode == "--stress") {
        selected = stress_tests;
    } else if (mode == "--full") {
        selected = quick_tests;
        selected.insert(selected.end(), stress_tests.begin(),
                        stress_tests.end());
        selected.insert(selected.end(), diagnostic_tests.begin(),
                        diagnostic_tests.end());
    } else {
        std::cerr << "Unknown mode: " << mode << '\n';
        return 2;
    }

    const fs::path base = "benchmarks/q6_integration_20261118";
    const std::string run_timestamp = timestamp();

    const fs::path report_dir =
        fs::path("benchmarks/q6_central/reports") /
        ("run_" + run_timestamp);

    const fs::path log_dir = report_dir / "raw_logs";
    fs::create_directories(log_dir);

    std::vector<BenchmarkResult> results;

    std::cout << "\nQ6 CENTRAL BENCHMARK SUITE\n";
    std::cout << "Mode: " << mode << '\n';
    std::cout << "Tests: " << selected.size() << '\n';
    std::cout << "Report: " << report_dir << "\n\n";

    for (const auto& test : selected) {
        const fs::path executable = base / test.executable;

        if (!fs::exists(executable)) {
            BenchmarkResult missing;
            missing.name = test.name;
            missing.status = TestStatus::FAIL;
            missing.exit_code = 127;
            results.push_back(missing);

            std::cout << "[FAIL] " << test.name
                      << " - executable missing\n";
            continue;
        }

        auto result = run_benchmark(
            test.name,
            executable.string(),
            {},
            log_dir.string()
        );

        std::cout << '[' << status_name(result.status) << "] "
                  << test.name
                  << " | exit=" << result.exit_code
                  << " | " << std::fixed << std::setprecision(2)
                  << result.duration_ms << " ms"
                  << " | metrics=" << result.metrics.size() << '\n';

        results.push_back(result);
    }

    size_t passed = 0, failed = 0, skipped = 0;

    std::ofstream report(report_dir / "report.txt");
    report << "Q6 CENTRAL BENCHMARK REPORT\n";
    report << "Mode: " << mode << '\n';
    report << "Timestamp: " << run_timestamp << "\n\n";

    report << std::left << std::setw(32) << "Benchmark"
           << std::setw(12) << "Status"
           << std::setw(10) << "Exit code"
           << "Duration (ms)\n";

    for (const auto& result : results) {
        report << std::left << std::setw(32) << result.name
               << std::setw(12) << status_name(result.status)
               << std::setw(10) << result.exit_code
               << std::fixed << std::setprecision(3)
               << result.duration_ms << '\n';

        if (result.status == TestStatus::PASS) ++passed;
        else if (result.status == TestStatus::SKIP) ++skipped;
        else ++failed;
    }

    report << "\nSUMMARY\n";
    report << "PASS: " << passed << '\n';
    report << "FAIL: " << failed << '\n';
    report << "SKIP: " << skipped << '\n';
    report.close();

    write_json_report(
        report_dir / "report.json",
        mode,
        run_timestamp,
        results,
        passed,
        failed,
        skipped
    );

    std::cout << "\n========== SUMMARY ==========\n";
    std::cout << "PASS: " << passed << '\n';
    std::cout << "FAIL: " << failed << '\n';
    std::cout << "SKIP: " << skipped << '\n';
    std::cout << "TXT:  " << report_dir / "report.txt" << '\n';
    std::cout << "JSON: " << report_dir / "report.json" << '\n';

    return failed ? 1 : 0;
}
