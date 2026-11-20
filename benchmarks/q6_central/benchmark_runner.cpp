#include "benchmark_runner.hpp"

#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <regex>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static std::string safe_name(const std::string& name) {
    std::string result;

    for (unsigned char c : name) {
        result += (std::isalnum(c) || c == '-' || c == '_')
            ? static_cast<char>(c) : '_';
    }

    return result.empty() ? "benchmark" : result;
}

static std::string normalize_metric_name(const std::string& input) {
    std::string result;
    bool previous_separator = false;

    for (unsigned char c : input) {
        if (std::isalnum(c)) {
            result += static_cast<char>(std::tolower(c));
            previous_separator = false;
        } else if (!previous_separator && !result.empty()) {
            result += '_';
            previous_separator = true;
        }
    }

    while (!result.empty() && result.back() == '_')
        result.pop_back();

    return result;
}

static bool is_numeric_value(const std::string& value) {
    if (value.empty()) return false;

    return std::isdigit(static_cast<unsigned char>(value[0])) ||
           value[0] == '-' || value[0] == '+' || value[0] == '.';
}

std::vector<Metric> extract_metrics(const std::string& log_path) {
    std::vector<Metric> metrics;
    std::ifstream input(log_path);

    if (!input) return metrics;

    const std::regex key_value(
        R"(([A-Za-z][A-Za-z0-9_.-]*)=([-+]?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?|[-+]?\.[0-9]+([eE][-+]?[0-9]+)?|[A-Za-z_][A-Za-z0-9_-]*))"
    );

    const std::regex label_value(
        R"(^\s*([A-Za-z][A-Za-z0-9 _().<>/-]*):\s*([-+]?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?|[-+]?\.[0-9]+([eE][-+]?[0-9]+)?))"
    );

    const std::regex denominator_pattern(
        R"(^\s*/\s*([0-9]+))"
    );

    const std::regex categorical_label(
        R"(^\s*([A-Za-z][A-Za-z0-9 _().<>/-]*):\s*([A-Za-z_][A-Za-z0-9_-]*)\s*$)"
    );

    std::string line;

    while (std::getline(input, line)) {
        std::smatch match;

        // Extract label-based metrics, e.g. "Iterations: 100000".
        if (std::regex_search(line, match, label_value)) {
            Metric metric;
            metric.name = normalize_metric_name(match[1].str());
            metric.value = match[2].str();
            metric.source_line = line;
            metric.numeric = true;

            metrics.push_back(metric);

            // Preserve denominators, e.g. "0 / 100000".
            std::string remainder = line.substr(
                static_cast<size_t>(match.length())
            );

            std::smatch denominator_match;

            if (std::regex_search(
                    remainder, denominator_match,
                    denominator_pattern)) {
                Metric denominator;
                denominator.name = metric.name + "_denominator";
                denominator.value = denominator_match[1].str();
                denominator.source_line = line;
                denominator.numeric = true;
                metrics.push_back(denominator);
            }
        }

        // Extract categorical label values, e.g. AVX-VNNI available: yes.
        if (!std::regex_search(line, match, label_value) &&
            std::regex_match(line, match, categorical_label)) {
            Metric metric;
            metric.name = normalize_metric_name(match[1].str());
            metric.value = match[2].str();
            metric.source_line = line;
            metric.numeric = false;
            metrics.push_back(metric);
        }

        // Extract key=value metrics. Prefix each metric with its
        // comparison/test context to prevent duplicate names.
        auto begin = std::sregex_iterator(
            line.begin(), line.end(), key_value
        );
        auto end = std::sregex_iterator();

        // Extract the context only once, before the first key=value pair.
        std::string prefix;

        if (begin != end) {
            prefix = normalize_metric_name(
                line.substr(0, static_cast<size_t>(begin->position()))
            );
        }

        for (auto it = begin; it != end; ++it) {
            Metric metric;

            metric.name = normalize_metric_name((*it)[1].str());

            if (!prefix.empty())
                metric.name = prefix + "_" + metric.name;

            metric.value = (*it)[2].str();
            metric.source_line = line;
            metric.numeric = is_numeric_value(metric.value);

            metrics.push_back(metric);
        }
    }

    return metrics;
}

BenchmarkResult run_benchmark(
    const std::string& name,
    const std::string& executable,
    const std::vector<std::string>& arguments,
    const std::string& log_directory
) {
    BenchmarkResult result;
    result.name = name;

    std::filesystem::create_directories(log_directory);

    const std::string prefix =
        log_directory + "/" + safe_name(name);

    result.stdout_log = prefix + ".stdout.log";
    result.stderr_log = prefix + ".stderr.log";

    const int out_fd = open(
        result.stdout_log.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC, 0644
    );

    const int err_fd = open(
        result.stderr_log.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC, 0644
    );

    if (out_fd < 0 || err_fd < 0) {
        if (out_fd >= 0) close(out_fd);
        if (err_fd >= 0) close(err_fd);
        return result;
    }

    std::vector<std::string> command{executable};
    command.insert(command.end(), arguments.begin(), arguments.end());

    std::vector<char*> argv;

    for (auto& item : command)
        argv.push_back(item.data());

    argv.push_back(nullptr);

    const auto start = std::chrono::steady_clock::now();
    const pid_t pid = fork();

    if (pid == 0) {
        dup2(out_fd, STDOUT_FILENO);
        dup2(err_fd, STDERR_FILENO);

        close(out_fd);
        close(err_fd);

        execv(executable.c_str(), argv.data());

        dprintf(
            STDERR_FILENO,
            "execv failed: %s\n",
            strerror(errno)
        );

        _exit(127);
    }

    close(out_fd);
    close(err_fd);

    if (pid < 0) return result;

    int status = 0;
    pid_t waited;

    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);

    const auto finish = std::chrono::steady_clock::now();

    result.duration_ms =
        std::chrono::duration<double, std::milli>(
            finish - start
        ).count();

    if (waited < 0) return result;

    if (WIFEXITED(status))
        result.exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status))
        result.exit_code = 128 + WTERMSIG(status);

    if (result.exit_code == 0)
        result.status = TestStatus::PASS;
    else if (result.exit_code == 77)
        result.status = TestStatus::SKIP;
    else
        result.status = TestStatus::FAIL;

    result.metrics = extract_metrics(result.stdout_log);

    return result;
}
