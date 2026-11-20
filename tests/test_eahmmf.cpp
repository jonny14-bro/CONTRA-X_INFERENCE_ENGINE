// test_eahmmf.cpp - validation suite for the expert residency layer (requirements section 4.13).
// Needs tiny.gguf (python3 test_tiny.py) and benchmarks/baseline/logits.txt (captured from the pre-EAHMMF binary).
#include <type_traits>

#define EAHMMF_TESTING
#define QWEN3MOE_NO_MAIN
#include "../qwen3moe.cpp"

using namespace eahmmf;

static int g_fail = 0, g_checks = 0;
#define CHECK(c, ...)                                                      \
    do {                                                                   \
        g_checks++;                                                        \
        if (!(c)) {                                                        \
            g_fail++;                                                      \
            fprintf(stderr, "  FAIL %s:%d: %s  ", __FILE__, __LINE__, #c); \
            fprintf(stderr, " " __VA_ARGS__);                              \
            fprintf(stderr, "\n");                                         \
        }                                                                  \
    } while (0)
#define SECTION(name) fprintf(stderr, "[%s]\n", name)

static const std::vector<int> PROMPT = {5, 17, 200, 33, 99, 3, 150, 256, 42, 77, 8, 9, 10, 11, 12};

struct Result {
    std::vector<float> logits;  // logits after the prompt
    std::vector<int> gen;       // greedy continuation
    std::vector<int> rid;       // routing trace
    std::vector<float> rw;
};
static bool same(const Result& a, const Result& b) {
    return a.logits.size() == b.logits.size() && !memcmp(a.logits.data(), b.logits.data(), a.logits.size() * 4) && a.gen == b.gen && a.rid == b.rid &&
           a.rw.size() == b.rw.size() && !memcmp(a.rw.data(), b.rw.data(), a.rw.size() * 4);
}
// Prompt (batched prefill, batch 4) + 12 greedy tokens, with routing trace. Store must already be configured.
static Result infer(Engine& E) {
    E.trace_routing = true;
    E.trace_ids.clear();
    E.trace_w.clear();
    Result r;
    r.logits.resize(E.n_vocab);
    E.eval(PROMPT, 0, 4, r.logits.data());
    int pos = (int)PROMPT.size();
    std::vector<float> lg = r.logits;
    for (int i = 0; i < 12; i++) {
        int t = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
        r.gen.push_back(t);
        E.forward(&t, 1, pos++, lg.data());
    }
    r.rid = E.trace_ids;
    r.rw = E.trace_w;
    E.trace_routing = false;
    return r;
}
static Result run(Engine& E, const Config& cfg, std::unique_ptr<DeviceBackend> dev = nullptr) {
    if (!E.init_experts(cfg, std::move(dev))) {
        CHECK(false, "test setup rejected expert-store reinitialization");
        return {};
    }
    return infer(E);
}
static bool bytes_match_source(Engine& E, int l, int x, Residency* served = nullptr, bool* copied = nullptr) {
    ExpertLease lease = E.store->acquire_lease(l, x);
    if (!lease) return false;
    const ExpertViews& v = lease.views();
    const ExpertDescriptor& d = E.store->descriptor(l, x);
    const WeightView* w[3] = {&v.gate, &v.up, &v.down};
    bool ok = true, cp = true;
    for (int c = 0; c < 3; c++) {
        const uint8_t* src = E.g.map + d.slice[c].file_offset;
        ok &= !memcmp(w[c]->data, src, d.bytes[c]);
        cp &= w[c]->data != src;
    }
    if (served) *served = v.served_from;
    if (copied) *copied = cp;
    return ok;
}

// Device whose memory holds an XOR-scrambled image: reading a device pointer from the CPU yields garbage,
// so correct logits prove the store bounced the data through d2h instead.
struct OpaqueDevice : SimDevice {
    const char* name() const override { return "opaque"; }
    bool h2d(void* d, const void* s, size_t n) override {
        if (!SimDevice::h2d(d, s, n)) return false;
        for (size_t i = 0; i < n; i++) ((uint8_t*)d)[i] ^= 0xFF;
        return true;
    }
    bool d2h(void* d, const void* s, size_t n) override {
        if (!SimDevice::d2h(d, s, n)) return false;
        for (size_t i = 0; i < n; i++) ((uint8_t*)d)[i] ^= 0xFF;
        return true;
    }
    bool cpu_readable() const override { return false; }
};

// Simulated device with externally controllable physical free memory.
struct PhysicalPressureSimDevice : SimDevice {
    size_t free_bytes = 0;
    uint64_t alloc_attempts = 0;

    const char* name() const override { return "physical-pressure-sim"; }

    void* alloc(size_t n) override {
        alloc_attempts++;
        if (n > free_bytes) return nullptr;
        void* p = SimDevice::alloc(n);
        if (p) free_bytes -= n;
        return p;
    }

    void release(void* p, size_t n) override {
        if (!p) return;
        SimDevice::release(p, n);
        free_bytes += n;
    }

    // This becomes a DeviceBackend override when the interface is extended.
    size_t free_memory() const override { return free_bytes; }
};

int main(int argc, char** argv) {
    init_types();

    SECTION("0. ETF workload metadata: prompt association and cancellation");
    {
        etf::TelemetryCollector collector(2);

        etf::WorkloadMetadata python_workload;
        python_workload.workload_id = "python-fastapi-api";
        python_workload.language = "python";
        python_workload.framework = "fastapi";
        python_workload.task_category = "coding";
        python_workload.workload_phase = "inference";

        etf::WorkloadMetadata cpp_workload;
        cpp_workload.workload_id = "cpp-cmake-build";
        cpp_workload.language = "cpp";
        cpp_workload.framework = "cmake";
        cpp_workload.task_category = "build";
        cpp_workload.workload_phase = "compilation";

        const uint64_t p1 = collector.begin_prompt(python_workload);
        CHECK(p1 != 0, "metadata prompt begins");
        collector.record_activation(p1, 0, 0, 1, 0.75f);
        auto first = collector.complete_prompt(p1);
        CHECK(!first.has_value(),
              "first prompt remains inside the two-prompt window");

        const uint64_t p2 = collector.begin_prompt(cpp_workload);
        CHECK(p2 != 0 && p2 != p1, "second metadata prompt begins");
        collector.record_activation(p2, 0, 1, 2, 0.50f);
        auto window = collector.complete_prompt(p2);

        CHECK(window.has_value(), "second prompt completes telemetry window");
        if (window) {
            CHECK(window->workloads.size() == 2,
                  "snapshot preserves metadata for both prompts");
            if (window->workloads.size() == 2) {
                CHECK(window->workloads[0].prompt_id == p1 &&
                      window->workloads[0].workload.language == "python" &&
                      window->workloads[0].workload.framework == "fastapi",
                      "first prompt retains its Python/FastAPI identity");
                CHECK(window->workloads[1].prompt_id == p2 &&
                      window->workloads[1].workload.language == "cpp" &&
                      window->workloads[1].workload.framework == "cmake",
                      "second prompt retains its C++/CMake identity");
            }
            CHECK(window->activations.size() == 2,
                  "existing activation events remain intact");
        }

        etf::WorkloadMetadata cancelled_workload;
        cancelled_workload.language = "java";
        const uint64_t cancelled = collector.begin_prompt(cancelled_workload);
        CHECK(cancelled != 0, "cancellable prompt begins");
        collector.record_activation(cancelled, 0, 2, 3, 0.25f);
        collector.cancel_prompt(cancelled);

        const uint64_t after_cancel = collector.begin_prompt();
        CHECK(after_cancel != 0, "prompt after cancellation begins");
        auto no_window = collector.complete_prompt(after_cancel);
        CHECK(!no_window.has_value(),
              "cancelled prompt does not count toward completion interval");
    }


    SECTION("0A. ETF partial-window flush: shutdown and active-prompt safety");
    {
        etf::TelemetryCollector collector(2);

        etf::WorkloadMetadata workload;
        workload.workload_id = "single-prompt-shutdown";
        workload.language = "python";

        const uint64_t p1 = collector.begin_prompt(workload);
        CHECK(p1 != 0, "partial-window prompt begins");
        collector.record_activation(p1, 0, 3, 4, 0.75f);

        auto normal = collector.complete_prompt(p1);
        CHECK(!normal.has_value(),
              "normal interval does not emit a partial window");

        auto flushed = collector.flush();
        CHECK(flushed.has_value(),
              "flush emits one completed prompt below interval");

        if (flushed) {
            CHECK(flushed->completed_prompts == 1 &&
                  flushed->first_prompt_id == p1 &&
                  flushed->last_prompt_id == p1,
                  "partial snapshot preserves prompt boundaries");
            CHECK(flushed->workloads.size() == 1 &&
                  flushed->workloads[0].prompt_id == p1 &&
                  flushed->workloads[0].workload.workload_id ==
                      "single-prompt-shutdown",
                  "partial snapshot preserves workload metadata");
            CHECK(flushed->activations.size() == 1 &&
                  flushed->activations[0].expert_id == 4,
                  "partial snapshot preserves activation events");
        }

        CHECK(!collector.flush().has_value(),
              "second flush does not duplicate an emitted window");

        const uint64_t p2 = collector.begin_prompt();
        CHECK(p2 != 0, "collector accepts prompt after flush");
        auto active_flush = collector.flush();
        CHECK(!active_flush.has_value(),
              "flush refuses to emit while a prompt is active");

        collector.cancel_prompt(p2);
        CHECK(!collector.flush().has_value(),
              "cancelled prompt does not create a partial snapshot");
    }

    SECTION("0B. ETF JSONL exporter: schema, escaping, isolation and append");
    {
        const std::string path =
            "/tmp/etf_jsonl_export_test_" + std::to_string(getpid()) + ".jsonl";
        std::remove(path.c_str());

        etf::TelemetrySnapshot snapshot;
        snapshot.first_prompt_id = 101;
        snapshot.last_prompt_id = 102;
        snapshot.completed_prompts = 2;
        snapshot.dropped_events = 3;

        etf::WorkloadMetadata first_workload;
        first_workload.workload_id = "python\"api";
        first_workload.language = "python\n3";
        first_workload.framework = "fastapi";
        first_workload.task_category = "coding";
        first_workload.workload_phase = "inference";

        etf::WorkloadMetadata second_workload;
        second_workload.workload_id = "cpp-build";
        second_workload.language = "cpp";
        second_workload.framework = "cmake";
        second_workload.task_category = "build";
        second_workload.workload_phase = "compilation";

        snapshot.workloads.push_back({101, first_workload});
        snapshot.workloads.push_back({102, second_workload});
        snapshot.activations.push_back({101, 0, 4, 2, 0.75f});
        snapshot.activations.push_back({102, 1, 7, 5, 0.25f});

        etf::JsonlExporter exporter;
        std::string error;
        CHECK(exporter.append(path, snapshot, &error),
              "valid mixed-workload snapshot exports: %s", error.c_str());

        std::ifstream input(path);
        std::string line1, line2, extra;
        CHECK(static_cast<bool>(std::getline(input, line1)),
              "first prompt JSONL record exists");
        CHECK(static_cast<bool>(std::getline(input, line2)),
              "second prompt JSONL record exists");
        CHECK(!std::getline(input, extra),
              "one record emitted per completed prompt");

        CHECK(line1.find("\"prompt_id\":101") != std::string::npos &&
              line1.find("python\\\"api") != std::string::npos &&
              line1.find("python\\n3") != std::string::npos,
              "first record preserves prompt identity and escaped metadata");

        CHECK(line2.find("\"prompt_id\":102") != std::string::npos &&
              line2.find("cpp-build") != std::string::npos,
              "second record preserves its own workload identity");

        CHECK(line1.find("\"expert_id\":2") != std::string::npos &&
              line1.find("\"expert_id\":5") == std::string::npos &&
              line2.find("\"expert_id\":5") != std::string::npos &&
              line2.find("\"expert_id\":2") == std::string::npos,
              "activations are associated only with their own prompt");

        CHECK(line1.find("\"dropped_events\":3") != std::string::npos,
              "window dropped-event count is persisted");

        etf::TelemetrySnapshot empty_activation;
        empty_activation.first_prompt_id = 103;
        empty_activation.last_prompt_id = 103;
        empty_activation.completed_prompts = 1;
        empty_activation.workloads.push_back(
            {103, etf::WorkloadMetadata{}});
        CHECK(exporter.append(path, empty_activation, &error),
              "completed prompt with no activations exports");

        std::ifstream appended(path);
        int lines = 0;
        std::string row;
        std::string last_row;
        while (std::getline(appended, row)) {
            ++lines;
            last_row = row;
        }
        CHECK(lines == 3, "append mode preserves earlier records");
        CHECK(last_row.find("\"prompt_id\":103") != std::string::npos &&
              last_row.find("\"activations\":[]") != std::string::npos,
              "empty activation prompt is represented explicitly");

        etf::TelemetrySnapshot invalid = snapshot;
        invalid.activations.push_back({999, 0, 0, 1, 0.5f});
        CHECK(!exporter.append(path, invalid, &error) &&
              error.find("unknown prompt ID") != std::string::npos,
              "unknown activation prompt ID is rejected");

        invalid = snapshot;
        invalid.completed_prompts = 99;
        CHECK(!exporter.append(path, invalid, &error),
              "inconsistent completed prompt count is rejected");

        std::remove(path.c_str());
    }

    const char* model = argc > 1 ? argv[1] : "tiny.gguf";
    Engine E;
    E.load(model, 128, false);
    const size_t FP = E.store->descriptor(0, 0).footprint;

    SECTION("1. baseline: default (SSD tier) logits match the pre-EAHMMF binary");
    Result base;
    {
        std::vector<float> lg(E.n_vocab);
        E.eval(PROMPT, 0, 32, lg.data());
        FILE* f = fopen("benchmarks/baseline/logits.txt", "r");
        char buf[1024] = {0};
        CHECK(f && fgets(buf, sizeof buf, f), "benchmarks/baseline/logits.txt missing");
        if (f) fclose(f);
        // Baseline numbers were printed with %.6f by the pre-EAHMMF binary; allow 2e-6 so that other
        // compiler flags (FMA / vectorisation order) do not cause false alarms. Bitwise equality is
        // enforced between tiers *within* one binary below.
        const char* p = buf;
        int n = 0;
        double maxd = 0;
        for (int i = 0; i < 16; i++) {
            char* end;
            double want = strtod(p, &end);
            if (end == p) break;
            p = end;
            maxd = std::max(maxd, std::fabs(want - (double)lg[i]));
            n++;
        }
        CHECK(n == 16 && maxd < 2e-6, "baseline logits differ by %.2e (n=%d)", maxd, n);
        base = run(E, Config());
        CHECK(!base.rid.empty() && base.rid.size() == base.rw.size());
        CHECK(base.rid.size() == (size_t)E.n_layer * (PROMPT.size() + 12) * E.n_used, "trace has %zu entries", base.rid.size());
    }

    SECTION("2. descriptors: stable identity, slices match GGUF, sizes");
    {
        E.init_experts(Config());
        bool all = true;
        for (int l = 0; l < E.n_layer; l++)
            for (int x = 0; x < E.n_expert; x++) {
                const ExpertDescriptor& d = E.store->descriptor(l, x);
                const Tensor* ts[3] = {E.layers[l].gate, E.layers[l].up, E.layers[l].down};
                all &= d.layer == l && d.expert_id == x && d.state == Residency::SSD && d.pin_count == 0;
                size_t sum = 0;
                for (int c = 0; c < 3; c++) {
                    size_t rows = (size_t)ts[c]->ne[1];
                    all &= d.slice[c].file_offset == ts[c]->file_off + (uint64_t)x * rows * ts[c]->row_bytes;
                    all &= d.bytes[c] == rows * ts[c]->row_bytes && d.slice[c].type == ts[c]->type;
                    all &= E.g.map + d.slice[c].file_offset == ts[c]->data + (size_t)x * rows * ts[c]->row_bytes;
                    all &= d.res_off[c] % 64 == 0;
                    sum += d.bytes[c];
                }
                all &= sum == d.total_bytes && d.footprint >= d.total_bytes && d.footprint % 4096 == 0;
            }
        CHECK(all, "descriptor metadata mismatch");
        CHECK(E.store->count_in(Residency::SSD) == (size_t)E.n_layer * E.n_expert);
        ExpertViews v = E.store->acquire(1, 5);  // SSD-served view is zero-copy into the mmap
        CHECK(v.ok && v.served_from == Residency::SSD && v.gate.data == E.g.map + E.store->descriptor(1, 5).slice[0].file_offset);
        E.store->release(1, 5);
    }

    SECTION("3. tier matrix: logits, generation and routing identical (bitwise) to baseline");
    {
        struct Case { const char* name; Config cfg; };
        std::vector<Case> cases;
        auto mk = [&](Residency t, size_t ram, size_t vram, bool verify) { Config c; c.exec_tier = t; c.ram_budget = ram; c.vram_budget = vram; c.verify = verify; c.record_events = true; return c; };
        cases.push_back({"ram unlimited", mk(Residency::RAM, SIZE_MAX, SIZE_MAX, true)});
        cases.push_back({"vram unlimited", mk(Residency::VRAM, SIZE_MAX, SIZE_MAX, true)});
        cases.push_back({"ram, 3-expert budget", mk(Residency::RAM, 3 * FP, SIZE_MAX, true)});
        cases.push_back({"vram 2-expert, ram unlimited", mk(Residency::VRAM, SIZE_MAX, 2 * FP, true)});
        cases.push_back({"vram 2-expert, ram 1-expert", mk(Residency::VRAM, 1 * FP, 2 * FP, true)});
        cases.push_back({"vram 1-expert, ram 1-expert", mk(Residency::VRAM, 1 * FP, 1 * FP, true)});
        cases.push_back({"vram budget < 1 expert (all fallback)", mk(Residency::VRAM, SIZE_MAX, FP / 2, false)});
        for (auto& c : cases) {
            Result r = run(E, c.cfg);
            CHECK(same(r, base), "case '%s' differs from baseline", c.name);
            std::string inv = E.store->check_invariants();
            CHECK(inv.empty(), "%s: %s", c.name, inv.c_str());
            const Stats& st = E.store->stats();
            CHECK(st.acquires == st.releases && st.pinned_now == 0, "pin leak in '%s'", c.name);
            fprintf(stderr, "    %-40s ssd->ram=%llu ram->vram=%llu vram->ram=%llu ram->ssd=%llu fallbacks=%llu\n", c.name, (unsigned long long)st.mig_n[0],
                    (unsigned long long)st.mig_n[1], (unsigned long long)st.mig_n[2], (unsigned long long)st.mig_n[3], (unsigned long long)st.fallbacks);
        }
        {   // RAM can hold the incoming expert plus one evicted victim: every pager path is exercised
            Result r = run(E, mk(Residency::VRAM, 2 * FP, 2 * FP, true));
            const Stats& st = E.store->stats();
            CHECK(same(r, base));
            CHECK(st.mig_n[SSD_RAM] > 0 && st.mig_n[RAM_VRAM] > 0 && st.mig_n[VRAM_RAM] > 0 && st.mig_n[RAM_SSD] > 0, "constrained run should exercise every path");
            CHECK(E.store->ram_used() <= 2 * FP && E.store->vram_used() <= 2 * FP);
        }
        {   // RAM holds only the expert being promoted: VRAM victims cannot be written back and are dropped to SSD
            Result r = run(E, mk(Residency::VRAM, 1 * FP, 2 * FP, true));
            const Stats& st = E.store->stats();
            CHECK(same(r, base));
            CHECK(st.mig_n[VRAM_SSD] > 0 && st.mig_n[VRAM_RAM] == 0, "drop path used when RAM cannot take the copy");
        }
    }

    SECTION("4. residency walk SSD->RAM->VRAM->RAM->SSD with byte-exact data at each tier");
    {
        Config c;
        c.verify = true;
        E.init_experts(c);
        Residency sv;
        bool cp;
        CHECK(E.store->residency(0, 7) == Residency::SSD && bytes_match_source(E, 0, 7, &sv, &cp) && sv == Residency::SSD && !cp);
        CHECK(E.store->promote(0, 7, Residency::RAM) && E.store->residency(0, 7) == Residency::RAM);
        CHECK(bytes_match_source(E, 0, 7, &sv, &cp) && sv == Residency::RAM && cp);
        CHECK(E.store->promote(0, 7, Residency::VRAM) && E.store->residency(0, 7) == Residency::VRAM);
        CHECK(E.store->descriptor(0, 7).ram_ptr == nullptr && E.store->ram_used() == 0 && E.store->vram_used() == FP, "exclusive residency / accounting");
        CHECK(bytes_match_source(E, 0, 7, &sv, &cp) && sv == Residency::VRAM && cp);
        CHECK(E.store->demote(0, 7, Residency::RAM) && E.store->residency(0, 7) == Residency::RAM && E.store->vram_used() == 0);
        CHECK(bytes_match_source(E, 0, 7, &sv) && sv == Residency::RAM);
        CHECK(E.store->demote(0, 7, Residency::SSD) && E.store->residency(0, 7) == Residency::SSD && E.store->ram_used() == 0);
        CHECK(bytes_match_source(E, 0, 7, &sv, &cp) && sv == Residency::SSD && !cp);
        CHECK(E.store->promote(0, 7, Residency::VRAM), "direct SSD->VRAM goes through RAM");
        CHECK(E.store->check_invariants().empty());
    }

    SECTION("5. repeated acquire of a resident expert: hits, no migration, nested pins");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        E.init_experts(c);
        CHECK(E.store->acquire(1, 2).ok);
        E.store->release(1, 2);
        const Stats before = E.store->stats();
        ExpertViews a = E.store->acquire(1, 2), b = E.store->acquire(1, 2);
        CHECK(a.ok && b.ok && a.gate.data == b.gate.data && E.store->descriptor(1, 2).pin_count == 2);
        CHECK(!E.store->demote(1, 2, Residency::RAM), "pinned expert must not move");
        CHECK(!E.store->promote(1, 2, Residency::RAM) &&
              E.store->residency(1, 2) == Residency::VRAM &&
              E.store->descriptor(1, 2).pin_count == 2,
              "pinned expert must not be promoted");
        E.store->release(1, 2);
        CHECK(E.store->descriptor(1, 2).pin_count == 1 && !E.store->demote(1, 2, Residency::SSD));
        E.store->release(1, 2);
        const Stats& st = E.store->stats();
        CHECK(st.hit[2] == before.hit[2] + 2 && st.miss[2] == before.miss[2]);
        uint64_t mig = 0, mig0 = 0;
        for (int p = 0; p < 5; p++) { mig += st.mig_n[p]; mig0 += before.mig_n[p]; }
        CHECK(mig == mig0, "no migration on a resident hit");
        CHECK(st.pinned_now == 0 && !E.store->release(1, 2), "release of unpinned expert is rejected");
    }

    SECTION("6. two layers, different experts, different tiers");
    {
        E.init_experts(Config());
        CHECK(E.store->promote(0, 3, Residency::VRAM) && E.store->promote(1, 5, Residency::RAM) && E.store->promote(1, 3, Residency::RAM));
        CHECK(E.store->residency(0, 3) == Residency::VRAM && E.store->residency(1, 5) == Residency::RAM && E.store->residency(1, 3) == Residency::RAM);
        CHECK(E.store->residency(0, 5) == Residency::SSD && E.store->residency(1, 4) == Residency::SSD);
        CHECK(E.store->count_in(Residency::VRAM) == 1 && E.store->count_in(Residency::RAM) == 2);
        CHECK(bytes_match_source(E, 0, 3) && bytes_match_source(E, 1, 5) && bytes_match_source(E, 1, 3) && bytes_match_source(E, 0, 5));
        CHECK(E.store->descriptor(0, 3).slice[0].file_offset != E.store->descriptor(1, 3).slice[0].file_offset, "same expert id in different layers differs");
        CHECK(E.store->check_invariants().empty());
    }

    SECTION("7. VRAM capacity forces LRU eviction (VRAM->RAM->SSD)");
    {
        Config c;
        c.vram_budget = 2 * FP;
        c.ram_budget = 2 * FP;  // incoming expert's staging copy + one evicted victim
        c.verify = true;
        E.init_experts(c);
        CHECK(E.store->promote(0, 0, Residency::VRAM) && E.store->promote(0, 1, Residency::VRAM));
        CHECK(E.store->vram_used() == 2 * FP);
        CHECK(E.store->promote(0, 2, Residency::VRAM));  // evicts LRU = (0,0) to RAM
        CHECK(E.store->residency(0, 0) == Residency::RAM && E.store->residency(0, 1) == Residency::VRAM && E.store->residency(0, 2) == Residency::VRAM);
        CHECK(E.store->promote(0, 3, Residency::VRAM));  // evicts (0,1) to RAM, which pushes (0,0) RAM->SSD
        CHECK(E.store->residency(0, 0) == Residency::SSD && E.store->residency(0, 1) == Residency::RAM);
        CHECK(E.store->vram_used() <= 2 * FP && E.store->ram_used() <= 2 * FP && E.store->check_invariants().empty());
        for (int x = 0; x < 4; x++) CHECK(bytes_match_source(E, 0, x), "expert %d corrupt after eviction chain", x);
    }

    SECTION("8. in-use expert cannot be evicted; pool exhausted by pinned experts");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 2 * FP;
        E.init_experts(c);
        ExpertViews a = E.store->acquire(0, 0), b = E.store->acquire(0, 1);
        CHECK(a.ok && b.ok && a.served_from == Residency::VRAM && b.served_from == Residency::VRAM);
        ExpertViews cc = E.store->acquire(0, 2);  // non-strict: cannot evict pinned -> degraded service, still valid
        CHECK(cc.ok && cc.served_from == Residency::RAM, "falls back to its valid RAM copy");
        CHECK(E.store->residency(0, 0) == Residency::VRAM && E.store->residency(0, 1) == Residency::VRAM, "pinned experts stay put");
        CHECK(E.store->stats().refused_evictions > 0 && E.store->stats().fallbacks > 0);
        CHECK(!memcmp(cc.gate.data, E.g.map + E.store->descriptor(0, 2).slice[0].file_offset, E.store->descriptor(0, 2).bytes[0]));
        E.store->release(0, 2);
        E.store->set_strict(true);
        ExpertViews d = E.store->acquire(0, 3);
        CHECK(!d.ok, "strict mode refuses instead of degrading");
        CHECK(E.store->descriptor(0, 3).pin_count == 0, "failed acquire must not leave a pin");
        E.store->set_strict(false);
        E.store->release(0, 0);
        ExpertViews e = E.store->acquire(0, 2);  // (0,0) unpinned now -> LRU victim
        CHECK(e.ok && e.served_from == Residency::VRAM && E.store->residency(0, 0) == Residency::RAM && E.store->residency(0, 1) == Residency::VRAM);
        E.store->release(0, 2);
        E.store->release(0, 1);
        CHECK(E.store->check_invariants().empty() && E.store->stats().pinned_now == 0);
    }

    SECTION("9. failed staging never exposes invalid memory (fault injection)");
    {
        Config c;
        c.verify = true;
        E.init_experts(c);
        auto* sim = static_cast<SimDevice*>(E.dev.get());
        E.store->set_fault_injection(0, 1, 0);  // SSD->RAM faults
        CHECK(!E.store->promote(0, 0, Residency::RAM) && E.store->residency(0, 0) == Residency::SSD && E.store->ram_used() == 0);
        CHECK(E.store->stats().alloc_fail[SSD_RAM] == 1, "SSD->RAM allocation failure is classified");
        E.store->set_fault_injection(1, 0, 0);
        CHECK(!E.store->promote(0, 0, Residency::RAM) && E.store->residency(0, 0) == Residency::SSD && E.store->ram_used() == 0);
        CHECK(E.store->stats().copy_fail[SSD_RAM] == 1, "SSD->RAM backing-copy failure is classified");
        E.store->set_fault_injection(0, 0, 1);
        CHECK(!E.store->promote(0, 0, Residency::RAM), "checksum must catch a corrupt staged copy");
        CHECK(E.store->residency(0, 0) == Residency::SSD && E.store->ram_used() == 0 && bytes_match_source(E, 0, 0));
        CHECK(E.store->stats().copy_fail[SSD_RAM] == 2, "SSD->RAM validation failure is classified");
        CHECK(E.store->promote(0, 0, Residency::RAM));
        sim->fail_alloc = 1;  // RAM->VRAM faults: the RAM copy must stay valid
        CHECK(!E.store->promote(0, 0, Residency::VRAM) && E.store->residency(0, 0) == Residency::RAM && E.store->descriptor(0, 0).vram_ptr == nullptr);
        CHECK(E.store->stats().alloc_fail[RAM_VRAM] == 1, "RAM->VRAM final allocation failure is classified");
        sim->fail_h2d = 1;
        CHECK(!E.store->promote(0, 0, Residency::VRAM) && E.store->residency(0, 0) == Residency::RAM);
        CHECK(E.store->stats().copy_fail[RAM_VRAM] == 1, "RAM->VRAM H2D failure is classified");
        sim->partial_h2d = 1;
        CHECK(!E.store->promote(0, 0, Residency::VRAM) && E.store->residency(0, 0) == Residency::RAM && E.store->vram_used() == 0);
        CHECK(E.store->stats().copy_fail[RAM_VRAM] == 2, "RAM->VRAM partial-copy failure is classified");
        sim->corrupt_h2d = 1;
        CHECK(!E.store->promote(0, 0, Residency::VRAM) && E.store->residency(0, 0) == Residency::RAM && E.store->vram_used() == 0);
        CHECK(E.store->stats().copy_fail[RAM_VRAM] == 3, "RAM->VRAM validation failure is classified");
        CHECK(sim->live_bytes == 0, "failed device copies must free their allocation");
        CHECK(bytes_match_source(E, 0, 0) && E.store->check_invariants().empty());
        CHECK(E.store->promote(0, 0, Residency::VRAM) && sim->pending == 0, "sync point: no un-synchronised copies after publish");
        sim->fail_d2h = 1;  // VRAM->RAM fault: the VRAM copy must stay valid
        CHECK(!E.store->demote(0, 0, Residency::RAM) && E.store->residency(0, 0) == Residency::VRAM && bytes_match_source(E, 0, 0));
        CHECK(E.store->stats().copy_fail[VRAM_RAM] == 1, "VRAM->RAM D2H failure is classified");
        CHECK(E.store->check_invariants().empty());

        Config c2;  // d2h failure during capacity-forced eviction: victim dropped to SSD, newcomer still lands in VRAM
        c2.vram_budget = 1 * FP;
        c2.verify = true;
        E.init_experts(c2);
        sim = static_cast<SimDevice*>(E.dev.get());
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        sim->fail_d2h = 1;
        CHECK(E.store->promote(0, 1, Residency::VRAM) && E.store->residency(0, 0) == Residency::SSD && E.store->residency(0, 1) == Residency::VRAM);
        CHECK(bytes_match_source(E, 0, 0) && bytes_match_source(E, 0, 1) && E.store->check_invariants().empty());

        Config c3;  // intermittent faults throughout a full inference: result must still equal the baseline
        c3.exec_tier = Residency::VRAM;
        c3.vram_budget = 3 * FP;
        c3.ram_budget = 2 * FP;
        c3.verify = true;
        E.init_experts(c3);
        sim = static_cast<SimDevice*>(E.dev.get());
        sim->fail_alloc = 3; sim->fail_h2d = 4; sim->partial_h2d = 3; sim->corrupt_h2d = 2; sim->fail_d2h = 4;
        E.store->set_fault_injection(3, 3, 3);
        Result r = infer(E);
        CHECK(same(r, base), "inference with injected faults must match baseline");
        CHECK(E.store->stats().fallbacks > 0 && E.store->stats().mig_fail[0] + E.store->stats().mig_fail[1] > 0, "faults were actually hit");
        CHECK(E.store->check_invariants().empty() && sim->live_bytes == E.store->vram_used());
    }

    SECTION("10. physical VRAM pressure evicts before allocation");
    {
        Config c;
        c.vram_budget = 2 * FP;
        c.ram_budget = 2 * FP;
        c.verify = true;

        auto dev = std::unique_ptr<PhysicalPressureSimDevice>(new PhysicalPressureSimDevice());
        PhysicalPressureSimDevice* sim = dev.get();
        sim->free_bytes = 2 * FP;
        CHECK(E.init_experts(c, std::move(dev)));

        CHECK(E.store->promote(0, 0, Residency::VRAM));
        CHECK(sim->free_bytes == FP && sim->alloc_attempts == 1);

        // Simulate another CUDA allocation consuming all remaining free memory.
        sim->free_bytes = 0;

        CHECK(E.store->promote(0, 1, Residency::VRAM),
              "pager should evict a VRAM resident before allocating under physical pressure");
        CHECK(sim->alloc_attempts == 2, "only the initial and successful incoming allocations attempted");
        CHECK(E.store->residency(0, 0) == Residency::RAM &&
              E.store->residency(0, 1) == Residency::VRAM);
        CHECK(sim->free_bytes == 0 && sim->live_bytes == FP);
        CHECK(bytes_match_source(E, 0, 0) && bytes_match_source(E, 0, 1));
        CHECK(E.store->check_invariants().empty());

        // A pinned resident cannot be evicted to satisfy physical pressure.
        auto dev2 = std::unique_ptr<PhysicalPressureSimDevice>(new PhysicalPressureSimDevice());
        PhysicalPressureSimDevice* sim2 = dev2.get();
        sim2->free_bytes = 2 * FP;
        CHECK(E.init_experts(c, std::move(dev2)));
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        ExpertViews pinned = E.store->acquire(0, 0);
        CHECK(pinned.ok && pinned.served_from == Residency::VRAM);

        sim2->free_bytes = 0;
        CHECK(!E.store->promote(0, 1, Residency::VRAM),
              "must refuse when physical memory is low and every VRAM resident is pinned");
        CHECK(sim2->alloc_attempts == 1, "must refuse before another device allocation");
        CHECK(E.store->residency(0, 0) == Residency::VRAM &&
              E.store->residency(0, 1) == Residency::RAM);
        CHECK(E.store->descriptor(0, 0).pin_count == 1);
        CHECK(E.store->release(0, 0));
        CHECK(E.store->check_invariants().empty());

        // A configured physical reserve triggers eviction before allocation.
        Config reserve_cfg = c;
        reserve_cfg.vram_budget = 2 * FP;
        reserve_cfg.vram_reserve = FP / 2;
        auto reserve_dev = std::unique_ptr<PhysicalPressureSimDevice>(new PhysicalPressureSimDevice());
        PhysicalPressureSimDevice* reserve_sim = reserve_dev.get();
        reserve_sim->free_bytes = 2 * FP;
        CHECK(E.init_experts(reserve_cfg, std::move(reserve_dev)));
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        CHECK(reserve_sim->free_bytes == FP);
        reserve_sim->free_bytes = FP + FP / 2 - 1;
        CHECK(E.store->promote(0, 1, Residency::VRAM),
              "reserve pressure should evict an unpinned resident before allocation");
        CHECK(reserve_sim->alloc_attempts == 2);
        CHECK(E.store->residency(0, 0) == Residency::RAM &&
              E.store->residency(0, 1) == Residency::VRAM);
        CHECK(reserve_sim->free_bytes >= reserve_cfg.vram_reserve);
        CHECK(E.store->check_invariants().empty());

        // The reserve must not be consumed by evicting a pinned resident.
        auto reserve_dev2 = std::unique_ptr<PhysicalPressureSimDevice>(new PhysicalPressureSimDevice());
        PhysicalPressureSimDevice* reserve_sim2 = reserve_dev2.get();
        reserve_sim2->free_bytes = 2 * FP;
        CHECK(E.init_experts(reserve_cfg, std::move(reserve_dev2)));
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        ExpertViews reserve_pinned = E.store->acquire(0, 0);
        CHECK(reserve_pinned.ok && reserve_pinned.served_from == Residency::VRAM);
        reserve_sim2->free_bytes = FP + FP / 2 - 1;
        CHECK(!E.store->promote(0, 1, Residency::VRAM),
              "reserve must be preserved when the only VRAM resident is pinned");
        CHECK(reserve_sim2->alloc_attempts == 1);
        CHECK(E.store->descriptor(0, 0).pin_count == 1);
        CHECK(E.store->release(0, 0));
        CHECK(E.store->check_invariants().empty());

        // Allocation fails once; reclaiming an LRU resident makes the retry succeed.
        CHECK(E.init_experts(c));
        auto* recovery_sim = static_cast<SimDevice*>(E.dev.get());
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        recovery_sim->fail_alloc = 1;
        CHECK(E.store->promote(0, 1, Residency::VRAM),
              "allocation should recover after evicting an unpinned VRAM resident");

        Stats recovered = E.store->stats();
        CHECK(recovered.vram_alloc_failures == 1);
        CHECK(recovered.vram_alloc_retries == 1);
        CHECK(recovered.vram_recovery_evictions == 1);
        CHECK(recovered.vram_recovered_allocations == 1);
        CHECK(recovered.vram_unrecovered_allocations == 0);
        CHECK(E.store->residency(0, 0) == Residency::RAM &&
              E.store->residency(0, 1) == Residency::VRAM);
        CHECK(bytes_match_source(E, 0, 0) && bytes_match_source(E, 0, 1));
        CHECK(E.store->check_invariants().empty());

        // Frequency recovery must protect a hot but older VRAM resident.
        {
            Config frequency_cfg = c;
            frequency_cfg.vram_budget = 3 * FP;
            frequency_cfg.ram_budget = 2 * FP;
            frequency_cfg.expert_policy = ExpertPolicy::FREQUENCY;
            frequency_cfg.expert_aging_interval = 0;

            CHECK(E.init_experts(frequency_cfg));
            auto* frequency_sim = static_cast<SimDevice*>(E.dev.get());
            CHECK(E.store->promote(0, 0, Residency::VRAM));
            CHECK(E.store->promote(0, 1, Residency::VRAM));

            auto request = [&](int expert, int count) {
                bool all_ok = true;
                for (int i = 0; i < count; ++i) {
                    auto v = E.store->acquire(0, expert);
                    all_ok &= v.ok;
                    if (v.ok) E.store->release(0, expert);
                }
                return all_ok;
            };

            CHECK(request(0, 10), "older hot VRAM expert receives ten requests");
            CHECK(request(1, 1), "newer cold VRAM expert receives one request");
            CHECK(request(2, 5), "incoming expert receives five requests");
            CHECK(E.store->promote(0, 2, Residency::RAM),
                  "incoming expert is staged in RAM before allocation recovery");

            CHECK(E.store->descriptor(0, 0).last_use <
                  E.store->descriptor(0, 1).last_use,
                  "hot expert is older than the cold VRAM candidate");
            CHECK(E.store->descriptor(0, 0).policy_frequency == 10 &&
                  E.store->descriptor(0, 1).policy_frequency == 1 &&
                  E.store->descriptor(0, 2).policy_frequency == 5,
                  "frequency priorities are established before recovery");

            frequency_sim->fail_alloc = 1;
            CHECK(E.store->promote(0, 2, Residency::VRAM),
                  "frequency recovery retries allocation after evicting the cold resident");

            CHECK(E.store->residency(0, 0) == Residency::VRAM,
                  "hot older expert remains in VRAM");
            CHECK(E.store->residency(0, 1) == Residency::RAM,
                  "newer cold expert is selected for recovery eviction");
            CHECK(E.store->residency(0, 2) == Residency::VRAM,
                  "incoming expert reaches VRAM after retry");

            Stats frequency_recovered = E.store->stats();
            CHECK(frequency_recovered.vram_alloc_failures == 1 &&
                  frequency_recovered.vram_alloc_retries == 1 &&
                  frequency_recovered.vram_recovery_evictions == 1 &&
                  frequency_recovered.vram_recovered_allocations == 1,
                  "frequency recovery counters record one successful retry");

            CHECK(bytes_match_source(E, 0, 0) &&
                  bytes_match_source(E, 0, 1) &&
                  bytes_match_source(E, 0, 2),
                  "all experts retain byte-exact backing data after recovery");
            CHECK(E.store->check_invariants().empty(),
                  "frequency physical recovery preserves invariants");
        }

        // Both allocation attempts fail; the incoming expert must remain in RAM.
        CHECK(E.init_experts(c));
        auto* failed_sim = static_cast<SimDevice*>(E.dev.get());
        CHECK(E.store->promote(0, 0, Residency::VRAM));
        failed_sim->fail_alloc = 2;
        CHECK(!E.store->promote(0, 1, Residency::VRAM),
              "promotion must fail when the allocation retry also fails");

        Stats unrecovered = E.store->stats();
        CHECK(unrecovered.vram_alloc_failures == 2);
        CHECK(unrecovered.vram_alloc_retries == 1);
        CHECK(unrecovered.vram_recovery_evictions == 1);
        CHECK(unrecovered.vram_recovered_allocations == 0);
        CHECK(unrecovered.vram_unrecovered_allocations == 1);
        CHECK(E.store->residency(0, 0) == Residency::RAM &&
              E.store->residency(0, 1) == Residency::RAM);
        CHECK(bytes_match_source(E, 0, 0) && bytes_match_source(E, 0, 1));
        CHECK(failed_sim->live_bytes == 0);
        CHECK(E.store->check_invariants().empty());
    }

    SECTION("11. unavailable backing range -> ERROR state, never served");
    {
        std::vector<std::array<TensorRef, 3>> refs(E.n_layer);
        for (int l = 0; l < E.n_layer; l++) {
            const Tensor* ts[3] = {E.layers[l].gate, E.layers[l].up, E.layers[l].down};
            for (int c = 0; c < 3; c++) refs[l][c] = {ts[c]->file_off, ts[c]->type, ts[c]->ne[0], ts[c]->ne[1], ts[c]->ne[2], ts[c]->row_bytes};
        }
        refs[1][0].file_offset = E.g.size - 100;  // runs off the end of the file
        SimDevice sd;
        ExpertStore s(E.g.map, E.g.size, E.n_layer, E.n_expert, refs, Config(), &sd);
        CHECK(s.residency(1, 0) == Residency::ERROR && s.residency(0, 0) == Residency::SSD);
        CHECK(!s.acquire(1, 0).ok && !s.promote(1, 0, Residency::RAM) && s.check_invariants().empty());
        CHECK(s.acquire(0, 0).ok);
        s.release(0, 0);
    }

    SECTION("12. device memory the CPU cannot read (bounce path)");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 3 * FP;
        c.verify = true;
        Result r = run(E, c, std::unique_ptr<DeviceBackend>(new OpaqueDevice()));
        CHECK(same(r, base), "bounce-through-d2h result differs");
        CHECK(E.store->stats().mig_n[RAM_VRAM] > 0 && E.store->check_invariants().empty());
        bool none = true;
        for (int l = 0; l < E.n_layer; l++) for (int x = 0; x < E.n_expert; x++) none &= E.store->descriptor(l, x).bounce_ptr == nullptr;
        CHECK(none, "bounce buffers released on unpin");
    }

    SECTION("13. instrumentation: auditable events and counters");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 2 * FP;
        c.ram_budget = 2 * FP;
        c.record_events = true;
        c.log_path = "/tmp/eahmmf_test.log";
        Result r = run(E, c);
        CHECK(same(r, base));
        size_t acq = 0, rel = 0, mig = 0;
        uint64_t mbytes = 0;
        bool sane = true;
        for (auto& ev : E.store->events()) {
            if (ev.kind == Event::ACQUIRE) acq++;
            if (ev.kind == Event::RELEASE) rel++;
            if (ev.kind == Event::MIGRATE) {
                mig++;
                if (ev.ok) mbytes += ev.bytes;
                sane &= ev.from != ev.to && ev.layer >= 0 && ev.expert >= 0 && ev.ram_used <= 2 * FP && ev.vram_used <= 2 * FP;
            }
        }
        const Stats& st = E.store->stats();
        uint64_t sb = 0, sn = 0;
        for (int p = 0; p < 5; p++) { sb += st.mig_bytes[p]; sn += st.mig_n[p] + st.mig_fail[p]; }
        CHECK(acq == st.acquires && rel == st.releases && st.pins == st.unpins);
        CHECK(mig == sn && mbytes == sb && sane, "event stream agrees with counters");
        CHECK(st.hit[2] + st.miss[2] == st.acquires, "every acquire is a VRAM hit or miss");

       uint64_t expert_samples = 0;
       double expert_latency = 0.0;
       bool latency_sane = true;
       for (int layer = 0; layer < E.n_layer; ++layer) {
           for (int expert = 0; expert < E.n_expert; ++expert) {
               const auto d = E.store->descriptor(layer, expert);
               for (int tier = 0; tier < 3; ++tier) {
                   expert_samples += d.acquire_samples[tier];
                   expert_latency += d.acquire_latency_us[tier];
                   latency_sane &= std::isfinite(d.acquire_latency_us[tier]) &&
                                   d.acquire_latency_us[tier] >= 0.0;
               }
               latency_sane &= std::isfinite(d.acquire_latency_us_total) &&
                               d.acquire_latency_us_total >= 0.0;
           }
       }
       CHECK(expert_samples == st.successful_acquires,
             "per-expert successful samples match aggregate successful acquires");

       const uint64_t serving_total =
           st.served[0] + st.served[1] + st.served[2];

       CHECK(serving_total == st.successful_acquires,
             "aggregate serving tiers equal successful acquisitions");

       CHECK(st.successful_acquires <= st.acquires,
             "successful acquisitions never exceed total attempts");
       CHECK(latency_sane,
             "per-expert acquisition latency values are finite and non-negative");
       CHECK(std::abs(expert_latency -
                      [&]() {
                          double total = 0.0;
                          for (int layer = 0; layer < E.n_layer; ++layer)
                              for (int expert = 0; expert < E.n_expert; ++expert)
                                  total += E.store->descriptor(layer, expert)
                                               .acquire_latency_us_total;
                          return total;
                      }()) < 1e-6,
             "per-tier latency sums match per-expert total latency");
        E.init_experts(Config());  // closes the log
        FILE* f = fopen("/tmp/eahmmf_test.log", "r");
        int lines = 0, macq = 0, mmig = 0;
        char buf[512];
        while (f && fgets(buf, sizeof buf, f)) { lines++; macq += strstr(buf, " ACQUIRE ") != nullptr; mmig += strstr(buf, " MIGRATE ") != nullptr; }
        if (f) fclose(f);
        CHECK(macq == (int)acq && mmig == (int)mig && lines > 0, "log file: %d acquire / %d migrate lines", macq, mmig);
    }


    SECTION("14. move-only ExpertLease: automatic pin ownership and release");
    {
        static_assert(!std::is_copy_constructible<ExpertLease>::value,
                      "ExpertLease must not be copy constructible");
        static_assert(!std::is_copy_assignable<ExpertLease>::value,
                      "ExpertLease must not be copy assignable");
        static_assert(std::is_move_constructible<ExpertLease>::value,
                      "ExpertLease must be move constructible");
        static_assert(std::is_move_assignable<ExpertLease>::value,
                      "ExpertLease must be move assignable");

        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 4 * FP;
        E.init_experts(c);

        // Scope exit must release exactly one pin.
        {
            ExpertLease lease = E.store->acquire_lease(0, 0);
            CHECK(static_cast<bool>(lease), "lease acquisition succeeds");
            CHECK(lease.views().ok, "lease exposes valid views");
            CHECK(E.store->descriptor(0, 0).pin_count == 1,
                  "lease owns one pin");
        }
        CHECK(E.store->descriptor(0, 0).pin_count == 0,
              "scope exit releases the pin");

        // Move construction transfers ownership; self-move is harmless.
        {
            ExpertLease source = E.store->acquire_lease(0, 1);
            CHECK(static_cast<bool>(source), "source lease acquired");

            ExpertLease destination(std::move(source));
            CHECK(!static_cast<bool>(source), "moved-from lease is empty");
            CHECK(static_cast<bool>(destination), "destination owns lease");
            CHECK(E.store->descriptor(0, 1).pin_count == 1,
                  "move construction preserves exactly one pin");

            destination = std::move(destination);
            CHECK(static_cast<bool>(destination),
                  "self-move assignment preserves ownership");
            CHECK(E.store->descriptor(0, 1).pin_count == 1,
                  "self-move does not alter pin count");

            destination.reset();
            destination.reset();
            CHECK(!static_cast<bool>(destination), "reset empties lease");
            CHECK(E.store->descriptor(0, 1).pin_count == 0,
                  "repeated reset releases only once");
        }

        // Move assignment releases the old destination pin before transfer.
        {
            ExpertLease destination = E.store->acquire_lease(0, 2);
            ExpertLease source = E.store->acquire_lease(0, 3);
            CHECK(destination && source, "both move-assignment leases acquired");

            destination = std::move(source);

            CHECK(!source, "move-assignment source becomes empty");
            CHECK(destination, "destination receives source ownership");
            CHECK(E.store->descriptor(0, 2).pin_count == 0,
                  "move assignment releases previous destination pin");
            CHECK(E.store->descriptor(0, 3).pin_count == 1,
                  "move assignment retains source pin exactly once");
        }
        CHECK(E.store->descriptor(0, 2).pin_count == 0 &&
              E.store->descriptor(0, 3).pin_count == 0,
              "scope exit releases moved destination pin");
        CHECK(E.store->stats().pinned_now == 0,
              "all lease pins released");

        // Failed strict acquisition must return an empty lease without a pin.
        Config strict_cfg;
        strict_cfg.exec_tier = Residency::VRAM;
        strict_cfg.vram_budget = 2 * FP;
        E.init_experts(strict_cfg);

        {
            ExpertLease first = E.store->acquire_lease(0, 0);
            ExpertLease second = E.store->acquire_lease(0, 1);
            CHECK(first && second, "two VRAM leases acquired");

            E.store->set_strict(true);
            ExpertLease refused = E.store->acquire_lease(0, 2);
            CHECK(!refused, "strict mode refuses unavailable lease");
            CHECK(E.store->descriptor(0, 2).pin_count == 0,
                  "failed lease acquisition leaves no pin");
        }

        CHECK(E.store->stats().pinned_now == 0,
              "strict-mode leases release cleanly");
        CHECK(E.store->check_invariants().empty(),
              "lease lifecycle preserves store invariants");
    }

    SECTION("15. store reinitialization is rejected while a lease is active");
    {
        Config c;
        c.exec_tier = Residency::VRAM;
        c.vram_budget = 4 * FP;
        CHECK(E.init_experts(c), "initial store configuration succeeds");

        ExpertStore* const old_store = E.store.get();
        DeviceBackend* const old_dev = E.dev.get();

        {
            ExpertLease active = E.store->acquire_lease(0, 0);
            CHECK(static_cast<bool>(active), "active lease acquired");
            CHECK(E.store->stats().pinned_now == 1,
                  "store reports one pinned expert");

            CHECK(!E.init_experts(Config()),
                  "reinitialization is rejected while pinned");
            CHECK(E.store.get() == old_store,
                  "rejected reinitialization preserves original store");
            CHECK(E.dev.get() == old_dev,
                  "rejected reinitialization preserves original device");
            CHECK(E.store->descriptor(0, 0).pin_count == 1,
                  "rejected reinitialization preserves active pin");
        }

        CHECK(E.store->stats().pinned_now == 0,
              "lease release restores idle store");

        CHECK(E.init_experts(Config()),
              "reinitialization succeeds after lease release");
        CHECK(E.store != nullptr,
              "store remains available after successful reinitialization");
        CHECK(E.store->stats().pinned_now == 0,
              "replacement store starts without pins");
        CHECK(E.store->check_invariants().empty(),
              "replacement store invariants are valid");
    }

    SECTION("16. placement capacity: deterministic virtual reservations");
    {
        const size_t FP = 100;

        // SSD -> RAM consumes RAM.
        PlacementCapacity ram_ok{FP, 0, 0, false};
        CHECK(ram_ok.can_reserve(Residency::SSD, Residency::RAM, FP),
              "SSD to RAM fits when RAM is sufficient");
        CHECK(ram_ok.reserve(Residency::SSD, Residency::RAM, FP),
              "SSD to RAM reservation succeeds");
        CHECK(ram_ok.ram_remaining == 0,
              "SSD to RAM consumes exactly one footprint");

        PlacementCapacity ram_short{FP - 1, 0, 0, false};
        CHECK(!ram_short.reserve(Residency::SSD, Residency::RAM, FP),
              "SSD to RAM rejects insufficient capacity");
        CHECK(ram_short.ram_remaining == FP - 1,
              "failed RAM reservation consumes no capacity");

        // SSD -> VRAM requires RAM staging plus both VRAM capacities.
        PlacementCapacity ssd_vram{FP, FP, FP, true};
        CHECK(ssd_vram.reserve(Residency::SSD, Residency::VRAM, FP),
              "SSD to VRAM reserves staging RAM and both VRAM capacities");
        CHECK(ssd_vram.ram_remaining == 0 &&
              ssd_vram.vram_remaining == 0 &&
              ssd_vram.physical_vram_remaining == 0,
              "SSD to VRAM consumes all three required capacities");

        PlacementCapacity no_staging{FP - 1, FP, FP, true};
        CHECK(!no_staging.reserve(Residency::SSD, Residency::VRAM, FP),
              "SSD to VRAM rejects insufficient staging RAM");
        CHECK(no_staging.vram_remaining == FP &&
              no_staging.physical_vram_remaining == FP,
              "failed staging check leaves VRAM untouched");

        // RAM -> VRAM does not need additional RAM staging.
        PlacementCapacity ram_vram{0, FP, FP, true};
        CHECK(ram_vram.reserve(Residency::RAM, Residency::VRAM, FP),
              "RAM to VRAM succeeds without additional RAM staging");
        CHECK(ram_vram.ram_remaining == 0 &&
              ram_vram.vram_remaining == 0 &&
              ram_vram.physical_vram_remaining == 0,
              "RAM to VRAM consumes only logical and physical VRAM");

        PlacementCapacity no_cuda{FP, FP, FP, false};
        CHECK(!no_cuda.reserve(Residency::RAM, Residency::VRAM, FP),
              "VRAM placement rejects a non-CUDA device");
        CHECK(no_cuda.ram_remaining == FP &&
              no_cuda.vram_remaining == FP,
              "non-CUDA rejection consumes no capacity");

        PlacementCapacity zero_footprint{FP, FP, FP, true};
        CHECK(!zero_footprint.reserve(
                  Residency::SSD, Residency::RAM, 0),
              "zero footprint is rejected");
        CHECK(zero_footprint.ram_remaining == FP,
              "zero footprint rejection leaves capacity unchanged");

        // A failed reservation must not partially consume any resource.
        PlacementCapacity atomic_failure{FP, FP - 1, FP, true};
        CHECK(!atomic_failure.reserve(
                  Residency::SSD, Residency::VRAM, FP),
              "logical VRAM shortage rejects the whole reservation");
        CHECK(atomic_failure.ram_remaining == FP &&
              atomic_failure.vram_remaining == FP - 1 &&
              atomic_failure.physical_vram_remaining == FP,
              "failed multi-resource reservation is atomic");

        // VRAM -> SSD directly releases VRAM; immutable GGUF remains backing.
        PlacementCapacity vram_ssd_no_ram{0, 0, 0, false};
        CHECK(vram_ssd_no_ram.can_reserve(
                  Residency::VRAM, Residency::SSD, FP),
              "VRAM to SSD does not require temporary RAM");
        CHECK(vram_ssd_no_ram.reserve(
                  Residency::VRAM, Residency::SSD, FP),
              "VRAM to SSD direct-release reservation succeeds");
        CHECK(vram_ssd_no_ram.ram_remaining == 0,
              "VRAM to SSD direct release consumes no RAM");

        // Repeated reservations cannot underflow capacity.
        PlacementCapacity repeated{FP, 0, 0, false};
        CHECK(repeated.reserve(Residency::SSD, Residency::RAM, FP),
              "first reservation succeeds");
        CHECK(!repeated.reserve(Residency::SSD, Residency::RAM, FP),
              "second reservation is rejected when capacity is exhausted");
        CHECK(repeated.ram_remaining == 0,
              "repeated reservations never underflow RAM");
    }

    SECTION("16. adaptive optimizer: normalized demand priority scoring");
    {
        CHECK(E.init_experts(Config()),
              "optimizer test store initializes");

        if (E.store) {
            AdaptiveOptimizer optimizer;
            optimizer.prime(*E.store, E.n_layer, E.n_expert);

            etf::TelemetrySnapshot snapshot;
            snapshot.first_prompt_id = 1;
            snapshot.last_prompt_id = 2;
            snapshot.completed_prompts = 2;

            const auto timing_acquire = E.store->acquire(0, 0);
            CHECK(timing_acquire.ok, "timing test acquire succeeds");
            if (timing_acquire.ok) {
                CHECK(E.store->release(0, 0),
                      "timing test acquire is released");
            }

            // Expert 0: highest activation frequency.
            for (int i = 0; i < 4; ++i)
                snapshot.activations.push_back({1, 0, i, 0, 0.5});

            // Expert 1: fewer activations but higher routing weight.
            for (int i = 0; i < 2; ++i)
                snapshot.activations.push_back({2, 0, i, 1, 0.9});

            const auto report = optimizer.observe(
                snapshot, *E.store, E.n_layer, E.n_expert);

            auto priority_of = [&](int layer, int expert) {
                for (const auto& demand : report.demand) {
                    if (demand.layer == layer && demand.expert_id == expert)
                        return demand.priority;
                }
                return -std::numeric_limits<double>::infinity();
            };

            CHECK(report.completed_prompts == 2,
                  "optimizer preserves completed prompt count");
            CHECK(report.experts_observed ==
                      static_cast<size_t>(E.n_layer) * E.n_expert,
                  "optimizer observes every model expert");
            CHECK(report.demand.size() == report.experts_observed,
                  "demand entries cover the complete expert set");

            const ExpertDemand* timing_demand = nullptr;
            for (const auto& demand : report.demand) {
                if (demand.layer == 0 && demand.expert_id == 0) {
                    timing_demand = &demand;
                    break;
                }
            }

            const int serving_tier = static_cast<int>(timing_acquire.served_from);
            CHECK(timing_demand != nullptr,
                  "optimizer exposes timing demand for acquired expert");
            if (timing_demand && timing_acquire.ok &&
                serving_tier >= 0 && serving_tier < 3) {
                CHECK(timing_demand->acquire_sample_delta[serving_tier] == 1,
                      "optimizer captures the new acquire sample in its serving tier");
                CHECK(timing_demand->acquire_latency_available[serving_tier],
                      "optimizer marks measured acquire latency available");
                CHECK(std::isfinite(
                          timing_demand->acquire_average_latency_us[serving_tier]) &&
                      timing_demand->acquire_average_latency_us[serving_tier] >= 0.0,
                      "optimizer reports a finite non-negative average acquire latency");
            }
            CHECK(priority_of(0, 0) > priority_of(0, 1),
                  "higher normalized activation demand wins");
            CHECK(priority_of(0, 1) > priority_of(0, 2),
                  "routing weight contributes to priority");
            CHECK(priority_of(0, 2) == 0.0,
                  "expert without demand receives zero priority");

            auto classification_of = [&](int layer, int expert) {
                for (const auto& demand : report.demand) {
                    if (demand.layer == layer && demand.expert_id == expert)
                        return demand.classification;
                }
                return DemandClass::COLD;
            };

            CHECK(classification_of(0, 0) == DemandClass::HOT,
                  "highest-demand expert is classified HOT");
            CHECK(classification_of(0, 1) == DemandClass::WARM,
                  "medium-demand expert is classified WARM");
            CHECK(classification_of(0, 2) == DemandClass::COLD,
                  "zero-demand expert is classified COLD");

           CHECK(report.placements.size() == 2,
                 "HOT and WARM SSD experts produce placement candidates");
           CHECK(report.recommendations == 0,
                 "missing target-tier measurements produce no actionable recommendations");
           CHECK(std::all_of(
                     report.placements.begin(), report.placements.end(),
                     [](const PlacementRecommendation& rec) {
                         return !rec.actionable &&
                                rec.target == Residency::RAM &&
                                rec.reason.find("Historical acquisition latency missing") !=
                                    std::string::npos;
                     }),
                 "unmeasured RAM placements are rejected with an explicit reason");

           const Stats planner_stats = E.store->stats();
           uint64_t planner_migrations = 0;
           for (int path = 0; path < 5; ++path)
               planner_migrations += planner_stats.mig_n[path] +
                                     planner_stats.mig_fail[path];
           CHECK(planner_migrations == 0,
                 "read-only placement planning performs no migrations");
           CHECK(E.store->count_in(Residency::SSD) ==
                     static_cast<size_t>(E.n_layer) * E.n_expert,
                 "read-only placement planning preserves all expert residencies");

// Persistent EMA activation history across ETF windows.
           CHECK(std::abs(optimizer.historical_activation(0, 0) - 0.20) < 1e-12,
                 "first observation initializes expert 0 EMA history");
           CHECK(std::abs(optimizer.historical_activation(0, 1) - 0.10) < 1e-12,
                 "first observation initializes expert 1 EMA history");

           etf::TelemetrySnapshot history_snapshot;
           history_snapshot.first_prompt_id = 3;
           history_snapshot.last_prompt_id = 4;
           history_snapshot.completed_prompts = 2;
           history_snapshot.activations.push_back({3, 0, 0, 1, 0.9});

           const auto history_report = optimizer.observe(
               history_snapshot, *E.store, E.n_layer, E.n_expert);

           CHECK(std::abs(optimizer.historical_activation(0, 0) - 0.16) < 1e-12,
                 "inactive expert history decays across observation windows");
           CHECK(std::abs(optimizer.historical_activation(0, 1) - 0.28) < 1e-12,
                 "active expert history incorporates the new observation");

           const ExpertDemand* history_demand = nullptr;
           for (const auto& demand : history_report.demand) {
               if (demand.layer == 0 && demand.expert_id == 0) {
                   history_demand = &demand;
                   break;
               }
           }

           CHECK(history_demand != nullptr,
                 "demand report exposes historical activation");
           if (history_demand) {
               CHECK(std::abs(history_demand->historical_activation_score - 0.16) < 1e-12,
                     "demand historical score matches optimizer EMA state");
           }

           optimizer.prime(*E.store, E.n_layer, E.n_expert);
           CHECK(optimizer.historical_activation(0, 0) == 0.0 &&
                 optimizer.historical_activation(0, 1) == 0.0,
                 "prime resets activation history");

           OptimizerConfig invalid_decay;
           invalid_decay.activation_history_decay = 1.0;
           AdaptiveOptimizer bad_decay_optimizer(invalid_decay);
           const auto bad_decay_report = bad_decay_optimizer.observe(
               snapshot, *E.store, E.n_layer, E.n_expert);

           CHECK(bad_decay_report.status.find(
                     "Invalid optimizer configuration") == 0,
                 "EMA decay of 1.0 is rejected");

           OptimizerConfig negative_decay;
           negative_decay.activation_history_decay = -0.1;
           AdaptiveOptimizer bad_negative_decay_optimizer(negative_decay);
           const auto bad_negative_decay_report =
               bad_negative_decay_optimizer.observe(
                   snapshot, *E.store, E.n_layer, E.n_expert);

           CHECK(bad_negative_decay_report.status.find(
                     "Invalid optimizer configuration") == 0,
                 "negative EMA decay is rejected");

           OptimizerConfig invalid_thresholds;
           invalid_thresholds.warm_threshold = 0.80;
           invalid_thresholds.hot_threshold = 0.70;
           AdaptiveOptimizer bad_threshold_optimizer(invalid_thresholds);
           const auto bad_threshold_report = bad_threshold_optimizer.observe(
               snapshot, *E.store, E.n_layer, E.n_expert);

           CHECK(bad_threshold_report.status.find(
                     "Invalid optimizer configuration") == 0,
                 "reversed demand thresholds are rejected");
           CHECK(bad_threshold_report.demand.empty(),
                 "invalid threshold configuration produces no demand");

           OptimizerConfig invalid_weight;
           invalid_weight.activation_weight = -1.0;
           AdaptiveOptimizer bad_weight_optimizer(invalid_weight);
           const auto bad_weight_report = bad_weight_optimizer.observe(
               snapshot, *E.store, E.n_layer, E.n_expert);

           CHECK(bad_weight_report.status.find(
                     "Invalid optimizer configuration") == 0,
                 "negative scoring weight is rejected");

           OptimizerConfig invalid_nan;
           invalid_nan.hot_threshold =
               std::numeric_limits<double>::quiet_NaN();
           AdaptiveOptimizer bad_nan_optimizer(invalid_nan);
           const auto bad_nan_report = bad_nan_optimizer.observe(
               snapshot, *E.store, E.n_layer, E.n_expert);

           CHECK(bad_nan_report.status.find(
                     "Invalid optimizer configuration") == 0,
                 "NaN demand threshold is rejected");
        }
    }



    SECTION("18. multi-expert capacity competition: priority reservations");
    {
        const size_t FP = 100;

        PlacementCapacity capacity{FP, 0, 0, false};

        std::vector<ExpertDemand> demands(2);
        demands[0].layer = 0;
        demands[0].expert_id = 0;
        demands[0].priority = 0.90;

        demands[1].layer = 0;
        demands[1].expert_id = 1;
        demands[1].priority = 0.60;

        const auto order = priority_order(demands);

        CHECK(order.size() == 2,
              "priority helper returns every candidate");
        CHECK(order[0] == 0,
              "higher-priority expert is ordered first");
        CHECK(order[1] == 1,
              "lower-priority expert is ordered second");

        std::vector<bool> accepted(demands.size(), false);

        for (const size_t index : order) {
            accepted[index] = capacity.reserve(
                Residency::SSD, Residency::RAM, FP);
        }

        CHECK(accepted[0],
              "higher-priority expert receives available RAM");
        CHECK(!accepted[1],
              "lower-priority expert is rejected after capacity exhaustion");
        CHECK(capacity.ram_remaining == 0,
              "one accepted expert consumes the entire RAM budget");

        CHECK(!capacity.reserve(
                  Residency::SSD, Residency::RAM, FP),
              "additional competing reservation is rejected");
        CHECK(capacity.ram_remaining == 0,
              "rejected reservation leaves capacity unchanged");

        // Equal priorities must have deterministic layer/expert tie-breaking.
        demands[0].priority = demands[1].priority = 0.75;
        demands[0].layer = 1;
        demands[1].layer = 0;

        const auto tie_order = priority_order(demands);
        CHECK(tie_order[0] == 1,
              "equal priorities prefer the lower layer");

        demands[0].layer = demands[1].layer = 0;
        demands[0].expert_id = 3;
        demands[1].expert_id = 2;

        const auto expert_tie_order = priority_order(demands);
        CHECK(expert_tie_order[0] == 1,
              "equal priority and layer prefer the lower expert ID");
    }


    SECTION("19. deterministic planner capacity competition");
    {
        CHECK(E.store != nullptr,
              "deterministic competition test has an initialized store");

        if (E.store) {
            const size_t fp0 = E.store->descriptor(0, 0).footprint;
            const size_t fp1 = E.store->descriptor(0, 1).footprint;
            const size_t one_expert_capacity = std::max(fp0, fp1);

            Config competition_store_config;
            competition_store_config.ram_budget = one_expert_capacity;
            competition_store_config.vram_budget = 0;

            CHECK(E.init_experts(competition_store_config),
                  "one-expert RAM capacity store initializes");

            if (E.store) {
                const auto descriptor0 = E.store->descriptor(0, 0);
                const auto descriptor1 = E.store->descriptor(0, 1);

                CHECK(descriptor0.state == Residency::SSD &&
                      descriptor1.state == Residency::SSD,
                      "both competing experts start in SSD");

                CHECK(E.store->ram_total() == one_expert_capacity,
                      "RAM budget is exactly enough for the larger expert");

                AdaptiveOptimizer competition_optimizer;
                OptimizerReport competition_report;

                auto configure_demand = [](
                    ExpertDemand& demand,
                    int expert_id,
                    DemandClass classification,
                    double priority,
                    size_t footprint
                ) {
                    demand.layer = 0;
                    demand.expert_id = expert_id;
                    demand.classification = classification;
                    demand.current_residency = Residency::SSD;
                    demand.footprint = footprint;
                    demand.priority = priority;

                    const int ssd_tier =
                        static_cast<int>(Residency::SSD);
                    const int ram_tier =
                        static_cast<int>(Residency::RAM);

                    // Deterministic acquisition measurements:
                    // SSD = 100 us, RAM = 10 us, benefit = 90 us.
                    demand.acquire_historical_latency_available[ssd_tier] = true;
                    demand.acquire_historical_latency_available[ram_tier] = true;
                    demand.acquire_historical_average_latency_us[ssd_tier] = 100.0;
                    demand.acquire_historical_average_latency_us[ram_tier] = 10.0;

                    demand.served_delta[ssd_tier] = 1;

                    // Deterministic SSD -> RAM migration cost.
                    demand.migration_historical_latency_available[SSD_RAM] = true;
                    demand.migration_historical_average_latency_us[SSD_RAM] = 5.0;
                };

                ExpertDemand high_priority;
                configure_demand(
                    high_priority, 0, DemandClass::HOT, 2.0,
                    descriptor0.footprint);

                ExpertDemand low_priority;
                configure_demand(
                    low_priority, 1, DemandClass::WARM, 1.0,
                    descriptor1.footprint);

                competition_report.demand.push_back(high_priority);
                competition_report.demand.push_back(low_priority);

                const Stats before_planning = E.store->stats();

                competition_optimizer.build_placement_recommendations_for_testing(
                    competition_report, *E.store);

                const PlacementRecommendation* placement0 = nullptr;
                const PlacementRecommendation* placement1 = nullptr;

                for (const auto& placement : competition_report.placements) {
                    if (placement.layer != 0) continue;

                    if (placement.expert_id == 0)
                        placement0 = &placement;

                    if (placement.expert_id == 1)
                        placement1 = &placement;
                }

                CHECK(placement0 != nullptr && placement1 != nullptr,
                      "planner produces decisions for both competing experts");

                CHECK(competition_report.placements.size() == 2,
                      "both placement decisions are retained");

                if (placement0 && placement1) {
                    CHECK(placement0->priority > placement1->priority,
                          "higher-priority expert is evaluated first");

                    CHECK(placement0->target == Residency::RAM &&
                          placement1->target == Residency::RAM,
                          "both candidates request RAM placement");

                    CHECK(placement0->actionable,
                          "higher-priority expert receives the RAM reservation");

                    CHECK(!placement1->actionable,
                          "lower-priority expert is rejected after reservation");

                    CHECK(placement1->reason == "Insufficient RAM capacity",
                          "lower-priority rejection identifies RAM exhaustion");

                    CHECK(placement0->estimated_benefit == 90.0,
                          "higher-priority benefit uses deterministic measurements");

                    CHECK(placement0->estimated_cost_us == 5.0,
                          "higher-priority migration cost uses deterministic measurements");

                    CHECK(placement0->cost_estimate_available,
                          "higher-priority migration cost is marked available");
                }

                CHECK(competition_report.recommendations == 1,
                      "planner produces exactly one actionable recommendation");

                const Stats after_planning = E.store->stats();

                uint64_t migrations_before = 0;
                uint64_t migrations_after = 0;

                for (int path = 0; path < 5; ++path) {
                    migrations_before += before_planning.mig_n[path];
                    migrations_after += after_planning.mig_n[path];
                }

                CHECK(migrations_after == migrations_before,
                      "planning performs no real migrations");

                CHECK(E.store->residency(0, 0) == Residency::SSD &&
                      E.store->residency(0, 1) == Residency::SSD,
                      "both experts remain in SSD after read-only planning");

                CHECK(E.store->ram_used() == 0,
                      "virtual reservations do not allocate real RAM");

                CHECK(E.store->stats().pinned_now == 0,
                      "planning leaves no active pins");
            }
        }
    }

    SECTION("17. planner: real acquisition and migration measurements");
    {
        CHECK(E.init_experts(Config()),
              "measured planner test store initializes");

        if (E.store) {
            AdaptiveOptimizer measured_optimizer;
            measured_optimizer.prime(*E.store, E.n_layer, E.n_expert);

            const auto ssd_acquire = E.store->acquire(0, 0);
            CHECK(ssd_acquire.ok && ssd_acquire.served_from == Residency::SSD,
                  "expert is acquired from SSD before promotion");
            if (ssd_acquire.ok)
                CHECK(E.store->release(0, 0), "SSD acquisition is released");

            CHECK(E.store->promote(0, 0, Residency::RAM),
                  "expert is promoted through the real pager");

            const auto ram_acquire = E.store->acquire(0, 0);
            CHECK(ram_acquire.ok && ram_acquire.served_from == Residency::RAM,
                  "expert is acquired from RAM after promotion");
            if (ram_acquire.ok)
                CHECK(E.store->release(0, 0), "RAM acquisition is released");

            CHECK(E.store->demote(0, 0, Residency::SSD),
                  "expert is returned to SSD before planning");

            const ExpertDescriptor measured =
                E.store->descriptor(0, 0);

            CHECK(measured.acquire_samples[static_cast<int>(Residency::SSD)] > 0,
                  "real SSD acquisition sample is recorded");
            CHECK(measured.acquire_samples[static_cast<int>(Residency::RAM)] > 0,
                  "real RAM acquisition sample is recorded");
            CHECK(measured.mig_n[SSD_RAM] > 0 &&
                  measured.mig_n[RAM_SSD] > 0,
                  "real migration samples are recorded for both directions");

            etf::TelemetrySnapshot measured_snapshot;
            measured_snapshot.first_prompt_id = 101;
            measured_snapshot.last_prompt_id = 102;
            measured_snapshot.completed_prompts = 2;
            for (int i = 0; i < 4; ++i)
                measured_snapshot.activations.push_back(
                    {101, 0, i, 0, 0.5});

            const auto measured_report = measured_optimizer.observe(
                measured_snapshot, *E.store, E.n_layer, E.n_expert);

            const ExpertDemand* measured_demand = nullptr;
            for (const auto& demand : measured_report.demand) {
                if (demand.layer == 0 && demand.expert_id == 0) {
                    measured_demand = &demand;
                    break;
                }
            }

            CHECK(measured_demand != nullptr,
                  "planner reports the measured expert");
            if (measured_demand) {
                CHECK(measured_demand->classification == DemandClass::HOT,
                      "measured expert is classified HOT");
                CHECK(measured_demand->acquire_historical_latency_available[
                          static_cast<int>(Residency::SSD)] &&
                      measured_demand->acquire_historical_latency_available[
                          static_cast<int>(Residency::RAM)],
                      "planner receives both real historical acquisition latencies");
                CHECK(measured_demand->migration_historical_latency_available[SSD_RAM],
                      "planner receives real SSD-to-RAM migration latency");
                CHECK(measured_demand->served_delta[
                          static_cast<int>(Residency::SSD)] > 0,
                      "planner observes SSD service in the current window");
            }

            const PlacementRecommendation* measured_placement = nullptr;
            for (const auto& placement : measured_report.placements) {
                if (placement.layer == 0 && placement.expert_id == 0) {
                    measured_placement = &placement;
                    break;
                }
            }

            CHECK(measured_placement != nullptr,
                  "HOT SSD expert produces a RAM placement candidate");

            if (measured_demand && measured_placement) {
                const double benefit =
                    measured_demand->acquire_historical_average_latency_us[
                        static_cast<int>(Residency::SSD)] -
                    measured_demand->acquire_historical_average_latency_us[
                        static_cast<int>(Residency::RAM)];
                const double cost =
                    measured_demand->migration_historical_average_latency_us[SSD_RAM];

                const bool expected_actionable =
                    std::isfinite(benefit) && std::isfinite(cost) &&
                    benefit > 0.0 && cost <= benefit;

                CHECK(measured_placement->cost_estimate_available == (benefit > 0.0),
                      "migration cost is evaluated only after positive measured benefit");
                CHECK(measured_placement->actionable == expected_actionable,
                      "planner decision follows measured benefit and migration cost");
                CHECK(measured_report.recommendations ==
                          (expected_actionable ? 1u : 0u),
                      "actionable recommendation count matches measured decision");
            }

            uint64_t migrations_after_planning = 0;
            const Stats after_planning = E.store->stats();
            for (int path = 0; path < 5; ++path)
                migrations_after_planning += after_planning.mig_n[path];

            CHECK(E.store->residency(0, 0) == Residency::SSD,
                  "read-only planner leaves the expert in SSD");
            CHECK(migrations_after_planning ==
                      measured.mig_n[SSD_RAM] + measured.mig_n[RAM_SSD],
                  "planning performs no additional migrations");
            CHECK(E.store->stats().pinned_now == 0,
                  "all measurement acquisitions are released before planning");
        }
    }



    SECTION("Adaptive RAM admission: SSD staging and recovery");
    {
        Config cfg;
        cfg.exec_tier = Residency::SSD;
        cfg.ram_budget = 2 * FP;

        CHECK(E.init_experts(cfg),
              "adaptive SSD staging store initializes");

        if (E.store) {
            E.store->set_adaptive_ram_cap(FP);

            CHECK(E.store->promote(0, 0, Residency::RAM),
                  "first expert fits adaptive RAM ceiling");

            auto held = E.store->acquire(0, 0);
            CHECK(held.ok, "existing RAM resident can be pinned");

            CHECK(!E.store->promote(0, 1, Residency::RAM),
                  "second expert is rejected by adaptive ceiling");

            CHECK(E.store->residency(0, 0) == Residency::RAM,
                  "adaptive rejection preserves existing resident");

            CHECK(E.store->ram_used() == FP,
                  "adaptive rejection does not evict RAM residents");

            CHECK(E.store->ram_total() == 2 * FP,
                  "adaptive ceiling does not change static capacity");

            CHECK(E.store->stats().adaptive_admission_reject[SSD_RAM] == 1,
                  "SSD-to-RAM adaptive rejection is counted");

            CHECK(E.store->stats().admission_reject[SSD_RAM] == 1,
                  "generic SSD-to-RAM rejection counter remains correct");

            if (held.ok) E.store->release(0, 0);

            E.store->set_adaptive_ram_cap(2 * FP);

            CHECK(E.store->promote(0, 1, Residency::RAM),
                  "SSD-to-RAM admission recovers after cap expansion");

            CHECK(E.store->ram_used() == 2 * FP,
                  "recovered admission fills static RAM capacity");

            CHECK(E.store->check_invariants().empty(),
                  "adaptive SSD staging preserves invariants");
        }
    }

    SECTION("Adaptive RAM admission: VRAM demotion and recovery");
    {
        Config cfg;
        cfg.exec_tier = Residency::SSD;
        cfg.ram_budget = FP;
        cfg.vram_budget = FP;

        CHECK(E.init_experts(cfg),
              "adaptive VRAM demotion store initializes");

        if (E.store) {
            CHECK(E.store->promote(0, 0, Residency::VRAM),
                  "expert is staged into VRAM");

            CHECK(E.store->ram_used() == 0,
                  "VRAM resident does not consume RAM pool");

            E.store->set_adaptive_ram_cap(0);

            CHECK(!E.store->migrate_no_eviction(0, 0, Residency::RAM),
                  "adaptive ceiling blocks VRAM-to-RAM demotion");

            CHECK(E.store->residency(0, 0) == Residency::VRAM,
                  "blocked demotion preserves VRAM source");

            CHECK(E.store->stats().adaptive_admission_reject[VRAM_RAM] == 1,
                  "VRAM-to-RAM adaptive rejection is counted");

            E.store->set_adaptive_ram_cap(FP);

            CHECK(E.store->migrate_no_eviction(0, 0, Residency::RAM),
                  "VRAM-to-RAM admission recovers after cap expansion");

            CHECK(E.store->residency(0, 0) == Residency::RAM,
                  "recovered demotion places expert in RAM");

            CHECK(E.store->check_invariants().empty(),
                  "adaptive VRAM demotion preserves invariants");
        }
    }

    SECTION("Adaptive cap reclamation: LRU SSD-to-RAM");
    {
        Config cfg;
        cfg.exec_tier = Residency::SSD;
        cfg.ram_budget = 2 * FP;

        CHECK(E.init_experts(cfg), "LRU adaptive staging store initializes");

        if (E.store) {
            E.store->set_adaptive_ram_cap(FP);

            CHECK(E.store->promote(0, 0, Residency::RAM),
                  "first expert enters RAM");

            CHECK(E.store->promote(0, 1, Residency::RAM),
                  "second expert reclaims RAM under adaptive cap");

            CHECK(E.store->residency(0, 0) == Residency::SSD &&
                  E.store->residency(0, 1) == Residency::RAM,
                  "LRU evicts the older unpinned resident");

            CHECK(E.store->ram_used() == FP,
                  "LRU staging stays within adaptive capacity");

            CHECK(E.store->check_invariants().empty(),
                  "LRU adaptive staging preserves invariants");
        }
    }

    SECTION("Adaptive cap reclamation: LRU VRAM-to-RAM demotion");
    {
        Config cfg;
        cfg.exec_tier = Residency::SSD;
        cfg.ram_budget = 2 * FP;
        cfg.vram_budget = FP;

        CHECK(E.init_experts(cfg), "LRU adaptive demotion store initializes");

        if (E.store) {
            CHECK(E.store->promote(0, 1, Residency::VRAM),
                  "incoming expert is placed in VRAM");

            E.store->set_adaptive_ram_cap(FP);

            CHECK(E.store->promote(0, 0, Residency::RAM),
                  "RAM resident is placed under adaptive cap");

            CHECK(E.store->demote(0, 1, Residency::RAM),
                  "LRU demotion reclaims RAM under adaptive cap");

            CHECK(E.store->residency(0, 0) == Residency::SSD &&
                  E.store->residency(0, 1) == Residency::RAM,
                  "demotion evicts the older RAM resident");

            CHECK(E.store->ram_used() == FP,
                  "demotion stays within adaptive capacity");

            CHECK(E.store->check_invariants().empty(),
                  "LRU adaptive demotion preserves invariants");
        }
    }

    SECTION("Adaptive cap reclamation: frequency SSD-to-RAM");
    {
        Config cfg;
        cfg.exec_tier = Residency::RAM;
        cfg.ram_budget = 2 * FP;
        cfg.expert_policy = ExpertPolicy::FREQUENCY;
        cfg.expert_aging_interval = 0;

        CHECK(E.init_experts(cfg), "frequency adaptive staging store initializes");

        if (E.store) {
            E.store->set_adaptive_ram_cap(FP);

            CHECK(E.store->promote(0, 0, Residency::RAM),
                  "initial RAM resident is placed");

            auto incoming = E.store->acquire(0, 1);
            CHECK(incoming.ok && incoming.served_from == Residency::RAM,
                  "higher-priority incoming expert is admitted into RAM");
            if (incoming.ok) E.store->release(0, 1);

            CHECK(E.store->residency(0, 0) == Residency::SSD &&
                  E.store->residency(0, 1) == Residency::RAM,
                  "frequency admission reclaims the lower-priority resident");

            CHECK(E.store->ram_used() == FP,
                  "frequency staging stays within adaptive capacity");

            CHECK(E.store->check_invariants().empty(),
                  "frequency adaptive staging preserves invariants");
        }
    }

    SECTION("Adaptive cap reclamation: frequency VRAM-to-RAM demotion");
    {
        Config cfg;
        cfg.exec_tier = Residency::SSD;
        cfg.ram_budget = 2 * FP;
        cfg.vram_budget = FP;
        cfg.expert_policy = ExpertPolicy::FREQUENCY;
        cfg.expert_aging_interval = 0;

        CHECK(E.init_experts(cfg), "frequency adaptive demotion store initializes");

        if (E.store) {
            CHECK(E.store->promote(0, 1, Residency::VRAM),
                  "incoming expert is placed in VRAM");

            E.store->set_adaptive_ram_cap(FP);

            CHECK(E.store->promote(0, 0, Residency::RAM),
                  "initial RAM resident is placed under adaptive cap");

            auto incoming = E.store->acquire(0, 1);
            CHECK(incoming.ok, "VRAM expert receives a frequency-counted request");
            if (incoming.ok) E.store->release(0, 1);

            CHECK(E.store->demote(0, 1, Residency::RAM),
                  "higher-priority VRAM expert reclaims RAM");

            CHECK(E.store->residency(0, 0) == Residency::SSD &&
                  E.store->residency(0, 1) == Residency::RAM,
                  "frequency demotion replaces the lower-priority resident");

            CHECK(E.store->ram_used() == FP,
                  "frequency demotion stays within adaptive capacity");

            CHECK(E.store->check_invariants().empty(),
                  "frequency adaptive demotion preserves invariants");
        }
    }

    SECTION("18. frequency RAM admission: score, rejection, pins and aging");
    {
        Config cfg;
        cfg.exec_tier = Residency::RAM;
        cfg.ram_budget = 2 * FP;
        cfg.expert_policy = ExpertPolicy::FREQUENCY;
        cfg.expert_aging_interval = 0;

        CHECK(E.init_experts(cfg), "frequency-policy store initializes");

        if (E.store) {
            CHECK(E.store->promote(0, 0, Residency::RAM),
                  "first resident is explicitly placed in RAM");
            CHECK(E.store->promote(0, 1, Residency::RAM),
                  "second resident fills the RAM budget");

            auto warm = E.store->acquire(0, 0);
            CHECK(warm.ok && warm.served_from == Residency::RAM,
                  "resident 0 receives a frequency-counted acquire");
            if (warm.ok) E.store->release(0, 0);

            auto warm_again = E.store->acquire(0, 0);
            CHECK(warm_again.ok, "resident 0 receives a second acquire");
            if (warm_again.ok) E.store->release(0, 0);

            auto warm_victim = E.store->acquire(0, 1);
            CHECK(warm_victim.ok, "resident 1 receives one acquire");
            if (warm_victim.ok) E.store->release(0, 1);

            // Candidate and resident 1 initially have equal scores.
            // Its higher score should admit it by evicting resident 1.
            auto candidate_first = E.store->acquire(0, 2);
            CHECK(candidate_first.ok,
                  "first candidate request falls back safely when admission is denied");
            if (candidate_first.ok) E.store->release(0, 2);

            auto candidate_second = E.store->acquire(0, 2);
            CHECK(candidate_second.ok,
                  "higher-frequency candidate is acquired");
            if (candidate_second.ok) E.store->release(0, 2);

            CHECK(E.store->residency(0, 0) == Residency::RAM,
                  "more frequently used resident remains in RAM");
            CHECK(E.store->residency(0, 1) == Residency::SSD,
                  "zero-frequency resident is selected as the victim");
            CHECK(E.store->residency(0, 2) == Residency::RAM,
                  "higher-frequency candidate is admitted");

            // Equal score is not enough to displace a resident.
            auto equal_candidate = E.store->acquire(0, 3);
            CHECK(equal_candidate.ok,
                  "equal-score candidate safely falls back to SSD");
            if (equal_candidate.ok) E.store->release(0, 3);
            CHECK(E.store->residency(0, 0) == Residency::RAM &&
                  E.store->residency(0, 2) == Residency::RAM &&
                  E.store->residency(0, 3) == Residency::SSD,
                  "rejected admission performs no partial eviction");

            // Pin both RAM residents. No pinned descriptor may be selected.
            auto pin0 = E.store->acquire(0, 0);
            auto pin2 = E.store->acquire(0, 2);
            CHECK(pin0.ok && pin2.ok,
                  "both RAM residents can be pinned");

            auto pinned_candidate = E.store->acquire(0, 4);
            CHECK(pinned_candidate.ok,
                  "candidate falls back when every resident is pinned");
            if (pinned_candidate.ok) E.store->release(0, 4);

            CHECK(E.store->residency(0, 0) == Residency::RAM &&
                  E.store->residency(0, 2) == Residency::RAM,
                  "pinned RAM residents are not evicted");

            if (pin0.ok) E.store->release(0, 0);
            if (pin2.ok) E.store->release(0, 2);
            CHECK(E.store->check_invariants().empty(),
                  "frequency admission preserves store invariants");
        }

        Config aging;
        aging.exec_tier = Residency::SSD;
        aging.expert_policy = ExpertPolicy::FREQUENCY;
        aging.expert_aging_interval = 2;
        CHECK(E.init_experts(aging), "aging test store initializes");

        if (E.store) {
            auto a = E.store->acquire(0, 0);
            if (a.ok) E.store->release(0, 0);
            auto b = E.store->acquire(0, 0);
            if (b.ok) E.store->release(0, 0);

            CHECK(E.store->descriptor(0, 0).policy_frequency == 1,
                  "frequency counters are halved after the configured interval");
        }
    }



    SECTION("18A. frequency VRAM admission: hot protection and cold rejection");
    {
        Config cfg;
        cfg.exec_tier = Residency::VRAM;
        cfg.vram_budget = 2 * FP;
        cfg.ram_budget = 3 * FP;
        cfg.expert_policy = ExpertPolicy::FREQUENCY;
        cfg.expert_aging_interval = 0;

        CHECK(E.init_experts(cfg), "frequency VRAM store initializes");

        if (E.store) {
            CHECK(E.store->promote(0, 0, Residency::VRAM) &&
                  E.store->promote(0, 1, Residency::VRAM),
                  "two initial experts fill VRAM");

            auto request = [&](int expert, int count) {
                bool all_ok = true;
                for (int i = 0; i < count; ++i) {
                    auto v = E.store->acquire(0, expert);
                    all_ok &= v.ok;
                    if (v.ok) E.store->release(0, expert);
                }
                return all_ok;
            };

            CHECK(request(0, 10), "expert 0 receives ten requests");
            CHECK(request(1, 3), "expert 1 receives three requests");

            CHECK(request(2, 4),
                  "incoming expert remains serviceable while building frequency");

            CHECK(E.store->descriptor(0, 0).policy_frequency == 10 &&
                  E.store->descriptor(0, 1).policy_frequency == 3 &&
                  E.store->descriptor(0, 2).policy_frequency == 4,
                  "request frequencies are counted as expected");

            CHECK(E.store->residency(0, 0) == Residency::VRAM,
                  "hot expert 0 remains in VRAM");
            CHECK(E.store->residency(0, 1) == Residency::RAM,
                  "less frequently used expert 1 is demoted to RAM");
            CHECK(E.store->residency(0, 2) == Residency::VRAM,
                  "higher-frequency expert 2 is admitted to VRAM");

            CHECK(request(3, 1),
                  "low-frequency incoming expert remains serviceable");

            CHECK(E.store->residency(0, 0) == Residency::VRAM &&
                  E.store->residency(0, 2) == Residency::VRAM,
                  "rejected candidate does not evict existing VRAM residents");
            CHECK(E.store->residency(0, 3) == Residency::RAM,
                  "rejected candidate retains its valid RAM copy");

            CHECK(E.store->vram_used() == 2 * FP &&
                  E.store->check_invariants().empty(),
                  "frequency VRAM admission preserves capacity and invariants");
        }
    }


   SECTION("18B. frequency RAM demotion: protect warm residents");
   {
       Config cfg;
       cfg.exec_tier = Residency::SSD;
       cfg.ram_budget = FP;
       cfg.vram_budget = FP;
       cfg.expert_policy = ExpertPolicy::FREQUENCY;
       cfg.expert_aging_interval = 0;

       CHECK(E.init_experts(cfg),
             "frequency demotion store initializes");

       if (E.store) {
           CHECK(E.store->promote(0, 1, Residency::VRAM),
                 "expert 1 is placed in VRAM");

           CHECK(E.store->promote(0, 0, Residency::RAM),
                 "expert 0 fills the RAM tier");

           auto request = [&](int expert, int count) {
               for (int i = 0; i < count; ++i) {
                   auto acquired = E.store->acquire(0, expert);
                   CHECK(acquired.ok,
                         "expert %d request %d succeeds",
                         expert, i + 1);
                   if (acquired.ok) E.store->release(0, expert);
               }
           };

           request(0, 10);

           CHECK(E.store->descriptor(0, 0).policy_frequency == 10,
                 "RAM resident has frequency 10");

           CHECK(E.store->descriptor(0, 1).policy_frequency == 0,
                 "VRAM resident initially has frequency 0");

           CHECK(!E.store->demote(0, 1, Residency::RAM),
                 "cold VRAM expert cannot displace warm RAM resident");

           CHECK(E.store->residency(0, 0) == Residency::RAM &&
                 E.store->residency(0, 1) == Residency::VRAM,
                 "rejected demotion preserves both existing residents");

           request(1, 11);

           CHECK(E.store->descriptor(0, 1).policy_frequency == 11,
                 "VRAM expert becomes more frequently used");

           CHECK(E.store->demote(0, 1, Residency::RAM),
                 "hotter VRAM expert can displace colder RAM resident");

           CHECK(E.store->residency(0, 0) == Residency::SSD &&
                 E.store->residency(0, 1) == Residency::RAM,
                 "higher-priority expert replaces lower-priority RAM resident");

           CHECK(E.store->ram_used() <= E.store->ram_total() &&
                 E.store->vram_used() <= E.store->vram_total(),
                 "demotion respects both tier capacities");

           CHECK(E.store->check_invariants().empty(),
                 "frequency demotion preserves store invariants");
       }
   }


   SECTION("18C. frequency VRAM eviction: fallback to SSD");
   {
       Config cfg;
       cfg.exec_tier = Residency::SSD;
       cfg.ram_budget = 3 * FP;
       cfg.vram_budget = FP;
       cfg.expert_policy = ExpertPolicy::FREQUENCY;
       cfg.expert_aging_interval = 0;

       CHECK(E.init_experts(cfg),
             "frequency SSD fallback store initializes");

       if (E.store) {
           auto request = [&](int expert, int count) {
               bool all_ok = true;
               for (int i = 0; i < count; ++i) {
                   auto v = E.store->acquire(0, expert);
                   all_ok &= v.ok;
                   if (v.ok) E.store->release(0, expert);
               }
               return all_ok;
           };

           CHECK(request(2, 5), "incoming expert receives five requests");
           CHECK(E.store->promote(0, 2, Residency::RAM),
                 "incoming expert is staged in RAM");

           CHECK(request(1, 1), "VRAM victim receives one request");
           CHECK(E.store->promote(0, 1, Residency::VRAM),
                 "low-frequency victim is placed in VRAM");

           CHECK(request(0, 10), "warm RAM expert 0 receives ten requests");
           CHECK(E.store->promote(0, 0, Residency::RAM),
                 "warm expert 0 fills a RAM slot");

           CHECK(request(3, 10), "warm RAM expert 3 receives ten requests");
           CHECK(E.store->promote(0, 3, Residency::RAM),
                 "warm expert 3 fills the final RAM slot");

           CHECK(E.store->ram_used() == 3 * FP &&
                 E.store->vram_used() == FP,
                 "RAM and VRAM are both full before admission");

           CHECK(E.store->descriptor(0, 0).policy_frequency == 10 &&
                 E.store->descriptor(0, 1).policy_frequency == 1 &&
                 E.store->descriptor(0, 2).policy_frequency == 5 &&
                 E.store->descriptor(0, 3).policy_frequency == 10,
                 "resident frequencies match the intended priority order");

           CHECK(E.store->promote(0, 2, Residency::VRAM),
                 "higher-priority incoming expert is admitted to VRAM");

           CHECK(E.store->residency(0, 0) == Residency::RAM &&
                 E.store->residency(0, 3) == Residency::RAM,
                 "warm RAM residents are preserved");

           CHECK(E.store->residency(0, 1) == Residency::SSD,
                 "VRAM victim falls back to SSD when RAM rejects demotion");

           CHECK(E.store->residency(0, 2) == Residency::VRAM,
                 "incoming expert occupies the freed VRAM slot");

           CHECK(E.store->ram_used() == 2 * FP &&
                 E.store->vram_used() == FP,
                 "tier accounting reflects SSD fallback");

           CHECK(E.store->check_invariants().empty(),
                 "automatic SSD fallback preserves store invariants");
       }
   }

   SECTION("19. deterministic hot/cold workload: LRU vs frequency");
    {
        auto run_hot_cold = [&](ExpertPolicy policy, const char* name) {
            Config cfg;
            cfg.exec_tier = Residency::RAM;
            cfg.ram_budget = 2 * FP;
            cfg.expert_policy = policy;
            cfg.expert_aging_interval = 0;

            CHECK(E.init_experts(cfg), "%s store initializes", name);
            if (!E.store) return;

            CHECK(E.store->promote(0, 0, Residency::RAM),
                  "%s places hot expert 0 in RAM", name);
            CHECK(E.store->promote(0, 1, Residency::RAM),
                  "%s places hot expert 1 in RAM", name);

            auto request = [&](int expert) {
                auto acquired = E.store->acquire(0, expert);
                CHECK(acquired.ok, "%s acquires expert %d", name, expert);
                if (acquired.ok) {
                    CHECK(E.store->release(0, expert),
                          "%s releases expert %d", name, expert);
                }
            };

            // Establish two frequently requested experts.
            for (int i = 0; i < 10; ++i) request(0);
            for (int i = 0; i < 10; ++i) request(1);

            // Introduce six cold experts, each requested exactly once.
            for (int expert = 2; expert < 8; ++expert) {
                request(expert);
            }

            if (policy == ExpertPolicy::FREQUENCY) {
                CHECK(E.store->residency(0, 0) == Residency::RAM,
                      "frequency retains hot expert 0");
                CHECK(E.store->residency(0, 1) == Residency::RAM,
                      "frequency retains hot expert 1");

                for (int expert = 2; expert < 8; ++expert) {
                    CHECK(E.store->residency(0, expert) == Residency::SSD,
                          "frequency rejects cold expert %d", expert);
                }
            } else {
                CHECK(E.store->residency(0, 0) == Residency::SSD,
                      "LRU evicts older hot expert 0");
                CHECK(E.store->residency(0, 1) == Residency::SSD,
                      "LRU evicts hot expert 1");

                for (int expert = 2; expert < 6; ++expert) {
                    CHECK(E.store->residency(0, expert) == Residency::SSD,
                          "LRU later evicts cold expert %d", expert);
                }

                CHECK(E.store->residency(0, 6) == Residency::RAM,
                      "LRU retains recently requested expert 6");
                CHECK(E.store->residency(0, 7) == Residency::RAM,
                      "LRU retains most recently requested expert 7");
            }

            CHECK(E.store->ram_used() <= 2 * FP,
                  "%s respects the two-expert RAM budget", name);
            CHECK(E.store->check_invariants().empty(),
                  "%s preserves store invariants", name);
        };

        run_hot_cold(ExpertPolicy::LRU, "LRU");
        run_hot_cold(ExpertPolicy::FREQUENCY, "frequency");
    }



    SECTION("20. frequency aging sensitivity and adaptation time");
    {
        const uint64_t intervals[] = {
            0, 256, 512, 1024, 2048, 4096, 16384
        };

        const uint64_t NEVER = UINT64_MAX;

        fprintf(stderr,
                "\nFREQUENCY AGING ADAPTATION RESULTS\n"
                "interval | E2 first RAM (P2/total) | "
                "E3 first RAM (P2/total) | "
                "both RAM (P2/total) | "
                "SSD->RAM | RAM->SSD | fallbacks\n");

        for (uint64_t interval : intervals) {
            Config cfg;
            cfg.exec_tier = Residency::RAM;
            cfg.ram_budget = 2 * FP;
            cfg.expert_policy = ExpertPolicy::FREQUENCY;
            cfg.expert_aging_interval = interval;

            CHECK(E.init_experts(cfg),
                  "aging interval %llu store initializes",
                  (unsigned long long)interval);

            if (!E.store) continue;

            CHECK(E.store->promote(0, 0, Residency::RAM),
                  "initial hot expert 0 placed in RAM");
            CHECK(E.store->promote(0, 1, Residency::RAM),
                  "initial hot expert 1 placed in RAM");

            bool requests_ok = true;
            uint64_t request_count = 0;
            uint64_t phase2_requests = 0;

            uint64_t first_e2_ram = NEVER;
            uint64_t first_e3_ram = NEVER;
            uint64_t first_both_ram = NEVER;

            auto request = [&](int expert, bool track_phase2) {
                if (track_phase2)
                    ++phase2_requests;

                auto acquired = E.store->acquire(0, expert);

                if (!acquired.ok) {
                    requests_ok = false;
                    return;
                }

                ++request_count;

                if (!E.store->release(0, expert))
                    requests_ok = false;

                if (!track_phase2)
                    return;

                const bool e2_ram =
                    E.store->residency(0, 2) == Residency::RAM;
                const bool e3_ram =
                    E.store->residency(0, 3) == Residency::RAM;

                if (e2_ram && first_e2_ram == NEVER)
                    first_e2_ram = phase2_requests;

                if (e3_ram && first_e3_ram == NEVER)
                    first_e3_ram = phase2_requests;

                if (e2_ram && e3_ram && first_both_ram == NEVER)
                    first_both_ram = phase2_requests;
            };

            // Phase 1: establish historical popularity.
            for (int i = 0; i < 4096; ++i)
                request(0, false);

            for (int i = 0; i < 4096; ++i)
                request(1, false);

            // Phase 2: workload shift.
            // Alternate the two new hot experts.
            for (int i = 0; i < 4095; ++i) {
                request(2, true);
                request(3, true);
            }

            CHECK(requests_ok,
                  "all acquisitions and releases succeed at interval %llu",
                  (unsigned long long)interval);

            CHECK(request_count == 16382,
                  "expected 16382 successful requests, got %llu",
                  (unsigned long long)request_count);

            CHECK(phase2_requests == 8190,
                  "expected 8190 phase-2 requests, got %llu",
                  (unsigned long long)phase2_requests);

            const Stats measured = E.store->stats();

            CHECK(measured.acquires == 16382 &&
                  measured.releases == 16382,
                  "acquire/release accounting matches workload");

            CHECK(E.store->ram_used() <= 2 * FP,
                  "aging interval %llu respects RAM capacity",
                  (unsigned long long)interval);

            CHECK(E.store->check_invariants().empty(),
                  "aging interval %llu preserves store invariants",
                  (unsigned long long)interval);

            auto print_step = [&](uint64_t step) {
                if (step == NEVER) {
                    fprintf(stderr, "NA");
                } else {
                    fprintf(stderr, "%llu/%llu",
                            (unsigned long long)step,
                            (unsigned long long)(8192 + step));
                }
            };

            fprintf(stderr, "%8llu | ",
                    (unsigned long long)interval);

            print_step(first_e2_ram);
            fprintf(stderr, " | ");

            print_step(first_e3_ram);
            fprintf(stderr, " | ");

            print_step(first_both_ram);

            fprintf(stderr, " | %llu | %llu | %llu | RAM={",
                    (unsigned long long)measured.mig_n[SSD_RAM],
                    (unsigned long long)measured.mig_n[RAM_SSD],
                    (unsigned long long)measured.fallbacks);

            bool first = true;

            for (int expert = 0; expert < 8; ++expert) {
                if (E.store->residency(0, expert) == Residency::RAM) {
                    fprintf(stderr, "%s%d",
                            first ? "" : ",", expert);
                    first = false;
                }
            }

            const uint64_t f0 =
                E.store->descriptor(0, 0).policy_frequency;
            const uint64_t f1 =
                E.store->descriptor(0, 1).policy_frequency;
            const uint64_t f2 =
                E.store->descriptor(0, 2).policy_frequency;
            const uint64_t f3 =
                E.store->descriptor(0, 3).policy_frequency;

            fprintf(stderr,
                    "} | freq={%llu,%llu,%llu,%llu}\n",
                    (unsigned long long)f0,
                    (unsigned long long)f1,
                    (unsigned long long)f2,
                    (unsigned long long)f3);
        }
    }

    fprintf(stderr, "\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
