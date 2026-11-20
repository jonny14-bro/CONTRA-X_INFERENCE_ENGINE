#include "../os_memory.h"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace eahmmf::osmem;

static void print_gib(const char* name, uint64_t bytes) {
    std::printf("%-28s %.3f GiB\n",
        name, double(bytes) / (1024.0 * 1024.0 * 1024.0));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 2;
    }

    const int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        perror("fstat");
        close(fd);
        return 1;
    }

    const size_t length = static_cast<size_t>(st.st_size);

    const auto before = read_system_memory();

    void* mapping = mmap(
        nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);

    close(fd);

    if (mapping == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    const auto residency =
        inspect_mmap_residency(mapping, length);

    const auto after = read_system_memory();

    std::printf("Model: %s\n", argv[1]);
    print_gib("Mapping size:", length);

    if (!residency.valid) {
        std::fprintf(stderr, "mincore failed: errno=%d\n",
                     residency.error_number);
        munmap(mapping, length);
        return 1;
    }

    std::printf("Page size:                  %zu bytes\n",
                residency.page_size);
    std::printf("Pages mapped:               %zu\n",
                residency.pages_total);
    std::printf("Pages resident:             %zu\n",
                residency.pages_resident);
    print_gib("Resident estimate:",
              residency.bytes_resident_estimate);
    std::printf("Resident percentage:        %.3f%%\n",
                residency.resident_fraction * 100.0);

    std::puts("\nSystem memory:");
    print_gib("MemAvailable before:",
              before.available_bytes);
    print_gib("MemAvailable after:",
              after.available_bytes);
    print_gib("SwapFree before:",
              before.swap_free_bytes);
    print_gib("SwapFree after:",
              after.swap_free_bytes);

    munmap(mapping, length);
    return 0;
}
