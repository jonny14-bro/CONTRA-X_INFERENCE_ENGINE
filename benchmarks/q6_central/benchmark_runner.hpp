#pragma once

#include <string>
#include <vector>

enum class TestStatus {
    PASS,
    FAIL,
    SKIP
};

struct Metric {
    std::string name;
    std::string value;
    std::string unit;
    std::string source_line;
    bool numeric = false;
};

struct BenchmarkResult {
    std::string name;
    TestStatus status = TestStatus::FAIL;
    int exit_code = -1;
    double duration_ms = 0.0;
    std::string stdout_log;
    std::string stderr_log;
    std::vector<Metric> metrics;
};

BenchmarkResult run_benchmark(
    const std::string& name,
    const std::string& executable,
    const std::vector<std::string>& arguments,
    const std::string& log_directory
);

std::vector<Metric> extract_metrics(const std::string& log_path);
