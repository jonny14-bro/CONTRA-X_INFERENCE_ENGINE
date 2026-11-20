
// tests/test_pwea_client.cpp

#include "../pwea_client.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    const std::string socket_path =
        argc > 1 ? argv[1] : "/tmp/pwea_prediction.sock";

    const std::string prompt =
        "Explain binary search in Python and provide its complexity.";

    pwea::Client client(socket_path);
    pwea::Prediction prediction;
    std::string error;

    if (!client.predict(prompt, prediction, error)) {
        std::fprintf(stderr, "PWEA CLIENT ERROR: %s\n", error.c_str());
        return 1;
    }

    std::printf("PWEA CLIENT TEST PASSED\n");
    std::printf("Layers: %zu\n", pwea::N_LAYER);
    std::printf("Experts per layer: %zu\n", pwea::N_EXPERT);
    std::printf("Total values: %zu\n", pwea::OUTPUT_SIZE);
    std::printf("Payload bytes: %zu\n", pwea::PAYLOAD_BYTES);

    for (size_t layer : {0U, 1U, 24U, 47U}) {
        double sum = 0.0;

        for (size_t expert = 0; expert < pwea::N_EXPERT; ++expert)
            sum += prediction.at(layer, expert);

        std::printf(
            "Layer %zu sum: %.15f\n",
            layer,
            sum
        );
    }

    return 0;
}
