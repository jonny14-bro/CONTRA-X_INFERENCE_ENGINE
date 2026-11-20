
#ifndef PWEA_CLIENT_H
#define PWEA_CLIENT_H

#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <unistd.h>

namespace pwea {

constexpr size_t N_LAYER = 48;
constexpr size_t N_EXPERT = 128;
constexpr size_t OUTPUT_SIZE = N_LAYER * N_EXPERT;
constexpr size_t PAYLOAD_BYTES = OUTPUT_SIZE * sizeof(double);
constexpr uint32_t MAX_FRAME_BYTES = 1024 * 1024;

struct Prediction {
    std::array<double, OUTPUT_SIZE> values{};

    double at(size_t layer, size_t expert) const {
        return values.at(layer * N_EXPERT + expert);
    }
};

class Client {
public:
    explicit Client(
        std::string socket_path = "/tmp/pwea_prediction.sock",
        int timeout_seconds = 30
    )
        : socket_path_(std::move(socket_path)),
          timeout_seconds_(timeout_seconds) {}

    bool predict(
        const std::string& prompt,
        Prediction& output,
        std::string& error
    ) const {
        error.clear();

        if (prompt.empty()) {
            error = "Empty prompt";
            return false;
        }

        if (socket_path_.empty() ||
            socket_path_.size() >= sizeof(sockaddr_un::sun_path)) {
            error = "Invalid Unix socket path";
            return false;
        }

        const uint64_t request_number =
            next_request_id_.fetch_add(1);

        const std::string request_id =
            "cpp-" + std::to_string(request_number);

        const std::string request =
            "{\"id\":\"" + request_id +
            "\",\"prompt\":\"" + json_escape(prompt) + "\"}";

        if (request.size() > MAX_FRAME_BYTES) {
            error = "Request exceeds maximum frame size";
            return false;
        }

        int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);

        if (fd < 0) {
            error = system_error("socket");
            return false;
        }

        struct timeval timeout {};
        timeout.tv_sec = timeout_seconds_;

        if (::setsockopt(
                fd, SOL_SOCKET, SO_RCVTIMEO,
                &timeout, sizeof(timeout)) < 0 ||
            ::setsockopt(
                fd, SOL_SOCKET, SO_SNDTIMEO,
                &timeout, sizeof(timeout)) < 0) {
            error = system_error("setsockopt");
            ::close(fd);
            return false;
        }

        sockaddr_un address {};
        address.sun_family = AF_UNIX;
        std::memcpy(
            address.sun_path,
            socket_path_.c_str(),
            socket_path_.size() + 1
        );

        if (::connect(
                fd,
                reinterpret_cast<sockaddr*>(&address),
                sizeof(address)) < 0) {
            error = system_error("connect");
            ::close(fd);
            return false;
        }

        bool success = false;

        do {
            uint32_t request_size =
                static_cast<uint32_t>(request.size());

            uint32_t network_size = host_to_network(request_size);

            if (!send_all(fd, &network_size, sizeof(network_size)) ||
                !send_all(fd, request.data(), request.size())) {
                error = system_error("send request");
                break;
            }

            uint32_t response_size_network = 0;

            if (!recv_all(
                    fd,
                    &response_size_network,
                    sizeof(response_size_network))) {
                error = "Cannot receive response frame length";
                break;
            }

            uint32_t response_size =
                network_to_host(response_size_network);

            if (response_size == 0 ||
                response_size > MAX_FRAME_BYTES) {
                error = "Invalid response frame length";
                break;
            }

            std::string header(response_size, '\0');

            if (!recv_all(fd, header.data(), header.size())) {
                error = "Incomplete response header";
                break;
            }

            std::string returned_id;
            std::string status;

            if (!json_string(header, "id", returned_id) ||
                !json_string(header, "status", status)) {
                error = "Invalid response header";
                break;
            }

            if (returned_id != request_id) {
                error = "Response request ID mismatch";
                break;
            }

            if (status != "ok") {
                std::string detail;
                json_string(header, "error", detail);
                error = detail.empty()
                    ? "Prediction service returned an error"
                    : detail;
                break;
            }

            std::string dtype;

            if (!json_string(header, "dtype", dtype) ||
                dtype != "float64-le") {
                error = "Unexpected prediction data type";
                break;
            }

            std::vector<uint64_t> shape;

            if (!json_array_uint(header, "shape", shape) ||
                shape.size() != 2 ||
                shape[0] != N_LAYER ||
                shape[1] != N_EXPERT) {
                error = "Unexpected prediction dimensions";
                break;
            }

            uint64_t payload_size = 0;

            if (!json_uint(header, "payload_bytes", payload_size) ||
                payload_size != PAYLOAD_BYTES) {
                error = "Unexpected prediction payload size";
                break;
            }

            std::array<unsigned char, PAYLOAD_BYTES> payload{};

            if (!recv_all(fd, payload.data(), payload.size())) {
                error = "Incomplete prediction payload";
                break;
            }

            // This client currently targets little-endian Linux systems.
            const uint16_t endian_probe = 1;

            if (*reinterpret_cast<const unsigned char*>(
                    &endian_probe) != 1) {
                error = "Host must be little-endian";
                break;
            }

            Prediction candidate;

            for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
                std::memcpy(
                    &candidate.values[i],
                    payload.data() + i * sizeof(double),
                    sizeof(double)
                );
            }

            if (!validate(candidate, error)) {
                break;
            }

            output = candidate;
            success = true;

        } while (false);

        ::close(fd);
        return success;
    }

private:
    std::string socket_path_;
    int timeout_seconds_;

    inline static std::atomic<uint64_t> next_request_id_{1};

    static uint32_t host_to_network(uint32_t value) {
        return ((value & 0x000000ffU) << 24) |
               ((value & 0x0000ff00U) << 8)  |
               ((value & 0x00ff0000U) >> 8)  |
               ((value & 0xff000000U) >> 24);
    }

    static uint32_t network_to_host(uint32_t value) {
        return host_to_network(value);
    }

    static bool send_all(int fd, const void* data, size_t size) {
        const char* ptr = static_cast<const char*>(data);

        while (size > 0) {
            ssize_t sent = ::send(fd, ptr, size, MSG_NOSIGNAL);

            if (sent < 0 && errno == EINTR)
                continue;

            if (sent <= 0)
                return false;

            ptr += sent;
            size -= static_cast<size_t>(sent);
        }

        return true;
    }

    static bool recv_all(int fd, void* data, size_t size) {
        char* ptr = static_cast<char*>(data);

        while (size > 0) {
            ssize_t received = ::recv(fd, ptr, size, 0);

            if (received < 0 && errno == EINTR)
                continue;

            if (received <= 0)
                return false;

            ptr += received;
            size -= static_cast<size_t>(received);
        }

        return true;
    }

    static std::string system_error(const char* operation) {
        return std::string(operation) + ": " + std::strerror(errno);
    }

    static std::string json_escape(const std::string& input) {
        std::string output;
        output.reserve(input.size());

        static const char hex[] = "0123456789abcdef";

        for (unsigned char c : input) {
            switch (c) {
                case '"': output += "\\\""; break;
                case '\\': output += "\\\\"; break;
                case '\b': output += "\\b"; break;
                case '\f': output += "\\f"; break;
                case '\n': output += "\\n"; break;
                case '\r': output += "\\r"; break;
                case '\t': output += "\\t"; break;
                default:
                    if (c < 0x20) {
                        output += "\\u00";
                        output += hex[c >> 4];
                        output += hex[c & 0x0f];
                    } else {
                        output += static_cast<char>(c);
                    }
            }
        }

        return output;
    }

    static size_t json_value_start(
        const std::string& json,
        const std::string& key
    ) {
        const std::string marker = "\"" + key + "\"";
        size_t pos = json.find(marker);

        if (pos == std::string::npos)
            return pos;

        pos = json.find(':', pos + marker.size());

        if (pos == std::string::npos)
            return pos;

        ++pos;

        while (pos < json.size() &&
               (json[pos] == ' ' ||
                json[pos] == '\t' ||
                json[pos] == '\n' ||
                json[pos] == '\r')) {
            ++pos;
        }

        return pos;
    }

    static bool json_string(
        const std::string& json,
        const std::string& key,
        std::string& result
    ) {
        size_t pos = json_value_start(json, key);

        if (pos == std::string::npos ||
            pos >= json.size() ||
            json[pos] != '"') {
            return false;
        }

        ++pos;
        result.clear();

        while (pos < json.size()) {
            char c = json[pos++];

            if (c == '"')
                return true;

            if (c == '\\') {
                if (pos >= json.size())
                    return false;

                char escaped = json[pos++];

                switch (escaped) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    default: return false;
                }
            } else {
                result += c;
            }
        }

        return false;
    }

    static bool json_uint(
        const std::string& json,
        const std::string& key,
        uint64_t& result
    ) {
        size_t pos = json_value_start(json, key);

        if (pos == std::string::npos ||
            pos >= json.size() ||
            json[pos] < '0' ||
            json[pos] > '9') {
            return false;
        }

        result = 0;

        while (pos < json.size() &&
               json[pos] >= '0' &&
               json[pos] <= '9') {
            unsigned digit = json[pos++] - '0';

            if (result > (UINT64_MAX - digit) / 10)
                return false;

            result = result * 10 + digit;
        }

        return true;
    }

    static bool json_array_uint(
        const std::string& json,
        const std::string& key,
        std::vector<uint64_t>& result
    ) {
        size_t pos = json_value_start(json, key);

        if (pos == std::string::npos ||
            pos >= json.size() ||
            json[pos++] != '[') {
            return false;
        }

        result.clear();

        while (pos < json.size()) {
            while (pos < json.size() &&
                   (json[pos] == ' ' || json[pos] == ',')) {
                ++pos;
            }

            if (pos < json.size() && json[pos] == ']')
                return !result.empty();

            if (pos >= json.size() ||
                json[pos] < '0' ||
                json[pos] > '9') {
                return false;
            }

            uint64_t value = 0;

            while (pos < json.size() &&
                   json[pos] >= '0' &&
                   json[pos] <= '9') {
                unsigned digit = json[pos++] - '0';

                if (value > (UINT64_MAX - digit) / 10)
                    return false;

                value = value * 10 + digit;
            }

            result.push_back(value);
        }

        return false;
    }

    static bool validate(
        const Prediction& prediction,
        std::string& error
    ) {
        for (size_t layer = 0; layer < N_LAYER; ++layer) {
            double sum = 0.0;

            for (size_t expert = 0; expert < N_EXPERT; ++expert) {
                double value = prediction.at(layer, expert);

                if (!std::isfinite(value) || value < 0.0) {
                    error = "Prediction contains invalid values";
                    return false;
                }

                sum += value;
            }

            if (std::abs(sum - 1.0) > 1e-10) {
                error = "Prediction layer does not sum to one";
                return false;
            }
        }

        return true;
    }
};

} // namespace pwea

#endif
