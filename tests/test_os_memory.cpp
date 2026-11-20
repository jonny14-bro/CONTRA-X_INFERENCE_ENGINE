#include "../os_memory.h"

#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <unistd.h>

using namespace eahmmf::osmem;

static int checks = 0;
static int failures = 0;

#define CHECK(expr, label) do { \
    ++checks; \
    if (!(expr)) { \
        ++failures; \
        std::fprintf(stderr, "FAIL: %s\n", label); \
    } else { \
        std::fprintf(stderr, "PASS: %s\n", label); \
    } \
} while (0)

int main() {
    std::puts("[OS memory snapshot]");

    const auto system = read_system_memory();
    CHECK(system.page_size > 0, "page size available");
    CHECK(system.meminfo_valid, "/proc/meminfo parsed");
    CHECK(system.total_bytes > 0, "physical RAM total detected");
    CHECK(system.available_bytes > 0, "MemAvailable detected");
    CHECK(system.vmstat_valid, "/proc/vmstat parsed");

    const auto process = read_process_memory(getpid());
    CHECK(process.valid, "current process status parsed");
    CHECK(process.rss_bytes > 0, "current process RSS detected");

    const auto cgroup = read_cgroup_memory(getpid());
    CHECK(cgroup.valid, "current process cgroup discovered");
    if (cgroup.valid)
        std::printf("cgroup=%s current=%llu swap=%llu\n",
                    cgroup.path.c_str(),
                    (unsigned long long)cgroup.current_bytes,
                    (unsigned long long)cgroup.swap_current_bytes);

    std::puts("[mincore observation]");

    const size_t page = static_cast<size_t>(system.page_size);
    const size_t length = page * 4;

    void* mapping = mmap(
        nullptr, length, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    CHECK(mapping != MAP_FAILED, "anonymous mapping created");

    if (mapping != MAP_FAILED) {
        volatile unsigned char* bytes =
            static_cast<volatile unsigned char*>(mapping);

        bytes[0] = 1;
        bytes[page * 3] = 2;

        const auto residency = inspect_mmap_residency(mapping, length);
        CHECK(residency.valid, "mincore snapshot succeeds");
        CHECK(residency.pages_total == 4, "four mapped pages counted");
        CHECK(residency.pages_resident >= 2,
              "touched pages reported resident");

        munmap(mapping, length);
    }

    std::puts("[lock budget accounting]");

    LockedMemoryBudget no_lock_budget(0);
    void* test_page = mmap(
        nullptr, page, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    CHECK(test_page != MAP_FAILED, "lock test mapping created");

    if (test_page != MAP_FAILED) {
        const auto result = no_lock_budget.lock(test_page, page);
        CHECK(!result.ok, "zero-byte lock budget rejects lock");
        CHECK(result.budget_rejected, "budget rejection is explicit");
        CHECK(no_lock_budget.used_bytes() == 0,
              "rejected lock does not change accounting");
        munmap(test_page, page);
    }

    const auto admission = check_admission(
        system, cgroup, page, 0, 0);
    CHECK(admission.system_headroom_ok,
          "one-page request fits system headroom");

    std::printf("\nOS memory tests: %d checks, %d failures\n",
                checks, failures);

    return failures ? 1 : 0;
}
