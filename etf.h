#pragma once

#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace etf {

struct ActivationEvent {
    uint64_t prompt_id = 0;
    int layer = -1;
    int token_position = -1;
    int expert_id = -1;
    float routing_weight = 0.0f;
};

// Explicit workload context supplied by the caller. ETF never infers
// these labels from prompt text.
struct WorkloadMetadata {
    uint32_t schema_version = 1;
    std::string workload_id = "unknown";
    std::string language = "unknown";
    std::string framework = "unknown";
    std::string task_category = "unknown";
    std::string workload_phase = "unknown";
};

// Metadata is associated with a prompt, not with individual token events.
// This preserves workload identity when a snapshot contains multiple prompts.
struct PromptWorkloadMetadata {
    uint64_t prompt_id = 0;
    WorkloadMetadata workload;
};

struct TelemetrySnapshot {
    uint64_t first_prompt_id = 0;
    uint64_t last_prompt_id = 0;
    uint64_t completed_prompts = 0;
    uint64_t dropped_events = 0;
    std::vector<ActivationEvent> activations;
    std::vector<PromptWorkloadMetadata> workloads;
};

// Append-only JSONL persistence for completed ETF snapshots.
// One record is emitted per completed prompt; activation events are joined
// by prompt_id so mixed-workload windows cannot cross-contaminate labels.
class JsonlExporter {
public:
    bool append(const std::string& path,
                const TelemetrySnapshot& snapshot,
                std::string* error = nullptr) const {
        auto fail = [&](const std::string& message) {
            if (error) *error = message;
            return false;
        };

        if (path.empty()) return fail("empty ETF output path");
        if (snapshot.completed_prompts != snapshot.workloads.size())
            return fail("completed prompt count does not match workload records");

        for (const auto& event : snapshot.activations) {
            const auto it = std::find_if(
                snapshot.workloads.begin(), snapshot.workloads.end(),
                [&](const PromptWorkloadMetadata& item) {
                    return item.prompt_id == event.prompt_id;
                });
            if (it == snapshot.workloads.end())
                return fail("activation references an unknown prompt ID");
        }

        std::ostringstream buffer;
        buffer << std::setprecision(9);

        for (size_t i = 0; i < snapshot.workloads.size(); ++i) {
            const auto& item = snapshot.workloads[i];
            const auto& w = item.workload;

            buffer << "{\"record_type\":\"etf_prompt\""
                   << ",\"schema_version\":" << w.schema_version
                   << ",\"prompt_id\":" << item.prompt_id
                   << ",\"workload\":{"
                   << "\"workload_id\":";
            write_string(buffer, w.workload_id);
            buffer << ",\"language\":";
            write_string(buffer, w.language);
            buffer << ",\"framework\":";
            write_string(buffer, w.framework);
            buffer << ",\"task_category\":";
            write_string(buffer, w.task_category);
            buffer << ",\"workload_phase\":";
            write_string(buffer, w.workload_phase);
            buffer << "},\"window\":{"
                   << "\"first_prompt_id\":" << snapshot.first_prompt_id
                   << ",\"last_prompt_id\":" << snapshot.last_prompt_id
                   << ",\"completed_prompts\":" << snapshot.completed_prompts
                   << ",\"dropped_events\":" << snapshot.dropped_events
                   << "},\"activations\":[";

            bool first = true;
            for (const auto& event : snapshot.activations) {
                if (event.prompt_id != item.prompt_id) continue;
                if (!first) buffer << ',';
                first = false;

                buffer << "{\"layer\":" << event.layer
                       << ",\"token_position\":" << event.token_position
                       << ",\"expert_id\":" << event.expert_id
                       << ",\"routing_weight\":";
                if (std::isfinite(event.routing_weight))
                    buffer << event.routing_weight;
                else
                    buffer << "null";
                buffer << '}';
            }

            buffer << "]}\n";
        }

        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) return fail("cannot open ETF output file: " + path);
        out << buffer.str();
        out.flush();
        if (!out) return fail("failed writing ETF output file: " + path);
        return true;
    }

private:
    static void write_string(std::ostream& out, const std::string& value) {
        out << '"';
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
                        static const char hex[] = "0123456789abcdef";
                        out << "\\u00" << hex[c >> 4] << hex[c & 0x0f];
                    } else {
                        out << static_cast<char>(c);
                    }
            }
        }
        out << '"';
    }
};

class TelemetryCollector {
public:
    explicit TelemetryCollector(
        uint64_t optimization_interval = 2,
        size_t max_events_per_window = 1'000'000)
        : interval_(optimization_interval ? optimization_interval : 2),
          max_events_(max_events_per_window) {
        events_.reserve(max_events_ < 65536 ? max_events_ : 65536);
    }

    TelemetryCollector(const TelemetryCollector&) = delete;
    TelemetryCollector& operator=(const TelemetryCollector&) = delete;

    // Backward-compatible entry point: workload labels remain explicitly unknown.
    uint64_t begin_prompt() {
        return begin_prompt(WorkloadMetadata{});
    }

    uint64_t begin_prompt(const WorkloadMetadata& workload) {
        std::lock_guard<std::mutex> lock(mu_);
        if (active_prompt_id_ != 0) return 0;

        active_prompt_id_ = ++next_prompt_id_;
        active_workload_ = workload;
        if (first_prompt_id_ == 0) first_prompt_id_ = active_prompt_id_;
        return active_prompt_id_;
    }

    void record_activation(uint64_t prompt_id,
                           int layer,
                           int token_position,
                           int expert_id,
                           float routing_weight) {
        if (prompt_id == 0 || layer < 0 || expert_id < 0) return;
        if (token_position < 0 || !std::isfinite(routing_weight) ||
            routing_weight < 0.0f) return;

        std::lock_guard<std::mutex> lock(mu_);
        if (prompt_id != active_prompt_id_) return;

        if (events_.size() >= max_events_) {
            // Tracked per active prompt; folded into the window total only
            // when the prompt completes, so cancelled prompts cannot
            // contaminate completed telemetry windows.
            ++active_dropped_events_;
            return;
        }

        events_.push_back({
            prompt_id, layer, token_position, expert_id, routing_weight
        });
    }

    std::optional<TelemetrySnapshot> complete_prompt(uint64_t prompt_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (prompt_id == 0 || prompt_id != active_prompt_id_) {
            return std::nullopt;
        }

        const uint64_t completed_id = active_prompt_id_;
        active_prompt_id_ = 0;
        ++completed_in_window_;
        dropped_events_ += active_dropped_events_;
        active_dropped_events_ = 0;
        workloads_.push_back({completed_id, active_workload_});
        active_workload_ = WorkloadMetadata{};

        if (completed_in_window_ < interval_) return std::nullopt;

        TelemetrySnapshot snapshot;
        snapshot.first_prompt_id = first_prompt_id_;
        snapshot.last_prompt_id = prompt_id;
        snapshot.completed_prompts = completed_in_window_;
        snapshot.dropped_events = dropped_events_;
        snapshot.activations = std::move(events_);
        snapshot.workloads = std::move(workloads_);

        events_.clear();
        workloads_.clear();
        events_.reserve(max_events_ < 65536 ? max_events_ : 65536);

        completed_in_window_ = 0;
        first_prompt_id_ = 0;
        dropped_events_ = 0;

        return snapshot;
    }

    // Emit a partial window at a clean shutdown boundary.
    // An active prompt is never included in a completed snapshot.
    std::optional<TelemetrySnapshot> flush() {
        std::lock_guard<std::mutex> lock(mu_);

        if (active_prompt_id_ != 0 || completed_in_window_ == 0) {
            return std::nullopt;
        }

        TelemetrySnapshot snapshot;
        snapshot.first_prompt_id = first_prompt_id_;
        snapshot.last_prompt_id = workloads_.back().prompt_id;
        snapshot.completed_prompts = completed_in_window_;
        snapshot.dropped_events = dropped_events_;
        snapshot.activations = std::move(events_);
        snapshot.workloads = std::move(workloads_);

        events_.clear();
        workloads_.clear();
        events_.reserve(max_events_ < 65536 ? max_events_ : 65536);

        completed_in_window_ = 0;
        first_prompt_id_ = 0;
        dropped_events_ = 0;

        return snapshot;
    }

    void cancel_prompt(uint64_t prompt_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (prompt_id != active_prompt_id_) return;

        active_prompt_id_ = 0;
        active_dropped_events_ = 0;
        active_workload_ = WorkloadMetadata{};
        events_.erase(
            std::remove_if(events_.begin(), events_.end(),
                [prompt_id](const ActivationEvent& event) {
                    return event.prompt_id == prompt_id;
                }),
            events_.end());

        if (completed_in_window_ == 0) first_prompt_id_ = 0;
    }

    size_t pending_events() const {
        std::lock_guard<std::mutex> lock(mu_);
        return events_.size();
    }

private:
    const uint64_t interval_;
    const size_t max_events_;

    mutable std::mutex mu_;
    uint64_t next_prompt_id_ = 0;
    uint64_t active_prompt_id_ = 0;
    uint64_t first_prompt_id_ = 0;
    uint64_t completed_in_window_ = 0;
    uint64_t dropped_events_ = 0;         // completed prompts in window
    uint64_t active_dropped_events_ = 0;  // active prompt only

    WorkloadMetadata active_workload_;
    std::vector<ActivationEvent> events_;
    std::vector<PromptWorkloadMetadata> workloads_;
};

} // namespace etf
