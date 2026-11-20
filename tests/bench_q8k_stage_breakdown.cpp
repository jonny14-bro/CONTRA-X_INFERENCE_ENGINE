
#define main q8k_original_main
#include "bench_q8k_avxvnni_q6_256_replay.cpp"
#undef main
#include <functional>

static void quantize_all(
    const Record& r,
    const float* activation,
    std::vector<Q8K>& q8
) {
    const int blocks = r.k / 256;

    for (int t = 0; t < r.nb; ++t) {
        for (int b = 0; b < blocks; ++b) {
            quantize_q8k(
                activation + static_cast<size_t>(t) * r.k
                           + static_cast<size_t>(b) * 256,
                q8[static_cast<size_t>(t) * blocks + b]
            );
        }
    }
}

static void decode_all(
    const Record& r,
    const uint8_t* weights,
    int rows,
    std::vector<PredecodedQ5>& q5,
    std::vector<PredecodedQ6>& q6
) {
    const int blocks = r.k / 256;
    const int block_bytes = r.type == 13 ? 176 : 210;

#pragma omp parallel for schedule(dynamic, 16)
    for (int row = 0; row < rows; ++row) {
        const uint8_t* wr =
            weights + static_cast<size_t>(row) * r.row_bytes;

        for (int b = 0; b < blocks; ++b) {
            const uint8_t* wb =
                wr + static_cast<size_t>(b) * block_bytes;

            const size_t index =
                static_cast<size_t>(row) * blocks + b;

            if (r.type == 13)
                decode_q5_predecoded(wb, q5[index]);
            else
                decode_q6_predecoded(wb, q6[index]);
        }
    }
}

static void dot_cached(
    const Record& r,
    int rows,
    const std::vector<Q8K>& q8,
    const std::vector<PredecodedQ5>& q5,
    const std::vector<PredecodedQ6>& q6,
    std::vector<float>& output
) {
    const int blocks = r.k / 256;

    std::fill(output.begin(), output.end(), 0.0f);

#pragma omp parallel for schedule(dynamic, 16)
    for (int row = 0; row < rows; ++row) {
        float acc[64] = {};

        for (int b = 0; b < blocks; ++b) {
            const size_t wi =
                static_cast<size_t>(row) * blocks + b;

            for (int t = 0; t < r.nb; ++t) {
                const Q8K& x =
                    q8[static_cast<size_t>(t) * blocks + b];

                if (r.type == 13)
                    acc[t] += dot_q5_predecoded(q5[wi], x);
                else
                    acc[t] += dot_q6_predecoded(q6[wi], x);
            }
        }

        for (int t = 0; t < r.nb; ++t)
            output[static_cast<size_t>(t) * rows + row] = acc[t];
    }
}

static double measure_median(
    const std::function<void()>& fn,
    int warmups = 2,
    int repetitions = 9
) {
    for (int i = 0; i < warmups; ++i)
        fn();

    std::vector<double> samples;
    samples.reserve(repetitions);

    using Clock = std::chrono::steady_clock;

    for (int i = 0; i < repetitions; ++i) {
        const auto start = Clock::now();
        fn();
        const auto end = Clock::now();

        samples.push_back(
            std::chrono::duration<double, std::micro>(
                end - start
            ).count()
        );
    }

    return median_us(samples);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(
            stderr,
            "Usage: %s weights.bin activations.f32 descriptors.tsv "
            "record_limit max_rows threads\n",
            argv[0]
        );
        return 2;
    }

    try {
        const int record_limit = std::atoi(argv[4]);
        const int max_rows = std::atoi(argv[5]);
        const int threads = std::atoi(argv[6]);

        if (record_limit < 0 || max_rows < 0 || threads < 1)
            throw std::runtime_error("Invalid arguments");

        Mapping weights(argv[1]);
        Mapping activations(argv[2]);
        const auto records = load_records(argv[3], record_limit);

        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        init_types();

        double total_quant = 0;
        double total_decode = 0;
        double total_cached_dot = 0;
        double total_repeated = 0;
        double total_cached_pipeline = 0;

        std::printf("===== Q8K STAGE BREAKDOWN =====\n");
        std::printf("Records: %zu Threads: %d\n\n",
                    records.size(), threads);

        for (const Record& r : records) {
            const int rows =
                max_rows > 0 ? std::min(max_rows, r.rows) : r.rows;

            const int blocks = r.k / 256;
            const size_t qcount =
                static_cast<size_t>(r.nb) * blocks;
            const size_t wcount =
                static_cast<size_t>(rows) * blocks;

            const uint8_t* wp = weights.data + r.weight_offset;
            const float* xp = reinterpret_cast<const float*>(
                activations.data + r.activation_offset
            );

            const uint64_t weight_bytes =
                static_cast<uint64_t>(rows) * r.row_bytes;
            const uint64_t activation_bytes =
                static_cast<uint64_t>(r.nb) * r.k * sizeof(float);

            if (r.weight_offset > weights.size ||
                weight_bytes > weights.size - r.weight_offset ||
                r.activation_offset > activations.size ||
                activation_bytes > activations.size - r.activation_offset)
                throw std::runtime_error("Input bounds error");

            std::vector<Q8K> q8(qcount);
            std::vector<PredecodedQ5> q5;
            std::vector<PredecodedQ6> q6;

            if (r.type == 13)
                q5.resize(wcount);
            else
                q6.resize(wcount);

            std::vector<float> output(
                static_cast<size_t>(r.nb) * rows
            );
            std::vector<float> reference;

            const double quant_us = measure_median([&] {
                quantize_all(r, xp, q8);
            });

            quantize_all(r, xp, q8);

            const double decode_us = measure_median([&] {
                decode_all(r, wp, rows, q5, q6);
            });

            decode_all(r, wp, rows, q5, q6);

            const double cached_dot_us = measure_median([&] {
                dot_cached(r, rows, q8, q5, q6, output);
            });

            const double repeated_us = measure_median([&] {
                run_candidate_predecoded(r, wp, xp, rows, output);
            });

            const double cached_pipeline_us = measure_median([&] {
                quantize_all(r, xp, q8);
                dot_cached(r, rows, q8, q5, q6, output);
            });

            // Numerical validation against the established implementation.
            run_candidate_predecoded(r, wp, xp, rows, reference);
            dot_cached(r, rows, q8, q5, q6, output);

            double max_delta = 0;
            uint64_t mismatches = 0;

            for (size_t i = 0; i < output.size(); ++i) {
                const double delta = std::fabs(
                    static_cast<double>(output[i]) - reference[i]
                );
                max_delta = std::max(max_delta, delta);
                if (delta > 2.0e-5)
                    ++mismatches;
            }

            total_quant += quant_us;
            total_decode += decode_us;
            total_cached_dot += cached_dot_us;
            total_repeated += repeated_us;
            total_cached_pipeline += cached_pipeline_us;

            std::printf(
                "record=%d op=%s type=Q%d nb=%d rows=%d "
                "quant_us=%.2f decode_us=%.2f "
                "cached_dot_us=%.2f repeated_us=%.2f "
                "cached_pipeline_us=%.2f "
                "parity_max=%.9g mismatches=%llu\n",
                r.id, r.operation.c_str(),
                r.type == 13 ? 5 : 6,
                r.nb, rows,
                quant_us, decode_us,
                cached_dot_us, repeated_us,
                cached_pipeline_us, max_delta,
                static_cast<unsigned long long>(mismatches)
            );

            if (mismatches)
                throw std::runtime_error("Cached dot parity failure");
        }

        std::printf("\n===== STAGE TOTALS: SUM OF RECORD MEDIANS =====\n");
        std::printf("quantization_us=%.2f\n", total_quant);
        std::printf("weight_decode_us=%.2f\n", total_decode);
        std::printf("cached_dot_us=%.2f\n", total_cached_dot);
        std::printf("repeated_predecode_us=%.2f\n", total_repeated);
        std::printf("cached_pipeline_us=%.2f\n", total_cached_pipeline);
        std::printf("STAGE BREAKDOWN: COMPLETED\n");

    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    return 0;
}
