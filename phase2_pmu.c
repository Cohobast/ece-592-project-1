#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>


/*


This program:

1. Builds a randomized dependent pointer-chase.
2. Runs the SAME workload twice.
3. Measures 4 counters in each run.

GROUP 1
-------
1. CPU cycles
2. Instructions
3. L1D read accesses
4. L1D read misses

GROUP 2
-------
5. LLC read accesses
6. LLC read misses
7. CPU-specific L2 event #1
8. CPU-specific L2 event #2

IMPORTANT:

The two L2 raw event configurations are CPU-specific.

You MUST determine their semantics and encodings using:
    perf list
    perf list --details
    CPU/vendor PMU documentation

before claiming they correspond to L2 accesses/hits/misses.

If RAW L2 events are not configured, the program will still run
the six portable counters.
============================================================
*/


/* ---------------------------------------------------------
   RAW L2 EVENT CONFIGURATION

   Leave as 0 until you identify appropriate events.

   Example ONLY:
       #define RAW_L2_EVENT_1 0x1234

   DO NOT copy an encoding from another CPU.
--------------------------------------------------------- */

#define RAW_L2_EVENT_1 0x0
#define RAW_L2_EVENT_2 0x0

#define RAW_L2_NAME_1 "L2_RAW_EVENT_1"
#define RAW_L2_NAME_2 "L2_RAW_EVENT_2"


/* Prevent compiler from eliminating the pointer chase. */
static volatile uint32_t final_index_sink = 0;




static long perf_event_open(
    struct perf_event_attr *attr,
    pid_t pid,
    int cpu,
    int group_fd,
    unsigned long flags
)
{
    return syscall(
        SYS_perf_event_open,
        attr,
        pid,
        cpu,
        group_fd,
        flags
    );
}




static int open_hardware_event(
    uint64_t config,
    int group_fd,
    int leader
)
{
    struct perf_event_attr pe;

    memset(&pe, 0, sizeof(pe));

    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = config;

    pe.disabled = leader ? 1 : 0;

    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    pe.read_format =
        PERF_FORMAT_GROUP |
        PERF_FORMAT_TOTAL_TIME_ENABLED |
        PERF_FORMAT_TOTAL_TIME_RUNNING;

    return (int)perf_event_open(
        &pe,
        0,
        -1,
        group_fd,
        0
    );
}


static int open_cache_event(
    uint64_t cache,
    uint64_t operation,
    uint64_t result,
    int group_fd,
    int leader
)
{
    struct perf_event_attr pe;

    memset(&pe, 0, sizeof(pe));

    pe.type = PERF_TYPE_HW_CACHE;
    pe.size = sizeof(pe);

    pe.config =
        cache |
        (operation << 8) |
        (result << 16);

    pe.disabled = leader ? 1 : 0;

    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    pe.read_format =
        PERF_FORMAT_GROUP |
        PERF_FORMAT_TOTAL_TIME_ENABLED |
        PERF_FORMAT_TOTAL_TIME_RUNNING;

    return (int)perf_event_open(
        &pe,
        0,
        -1,
        group_fd,
        0
    );
}


static int open_raw_event(
    uint64_t config,
    int group_fd,
    int leader
)
{
    struct perf_event_attr pe;

    memset(&pe, 0, sizeof(pe));

    pe.type = PERF_TYPE_RAW;
    pe.size = sizeof(pe);
    pe.config = config;

    pe.disabled = leader ? 1 : 0;

    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    pe.read_format =
        PERF_FORMAT_GROUP |
        PERF_FORMAT_TOTAL_TIME_ENABLED |
        PERF_FORMAT_TOTAL_TIME_RUNNING;

    return (int)perf_event_open(
        &pe,
        0,
        -1,
        group_fd,
        0
    );
}


static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    *state = x;

    return x;
}



static uint32_t *build_random_cycle(
    size_t nodes,
    uint32_t seed
)
{
    uint32_t *order = NULL;
    uint32_t *next = NULL;

    if (nodes < 2) {
        fprintf(stderr, "Need at least 2 nodes.\n");
        return NULL;
    }

    if (posix_memalign(
            (void **)&order,
            64,
            nodes * sizeof(uint32_t)
        ) != 0) {

        perror("posix_memalign order");
        return NULL;
    }

    if (posix_memalign(
            (void **)&next,
            64,
            nodes * sizeof(uint32_t)
        ) != 0) {

        perror("posix_memalign next");
        free(order);
        return NULL;
    }

    for (size_t i = 0; i < nodes; i++) {
        order[i] = (uint32_t)i;
    }

    uint32_t rng = seed;

    for (size_t i = nodes - 1; i > 0; i--) {

        size_t j =
            (size_t)(
                xorshift32(&rng) %
                (uint32_t)(i + 1)
            );

        uint32_t temp = order[i];
        order[i] = order[j];
        order[j] = temp;
    }

    for (size_t i = 0; i < nodes - 1; i++) {
        next[order[i]] = order[i + 1];
    }

    next[order[nodes - 1]] = order[0];

    free(order);

    return next;
}



static uint32_t run_pointer_chase(
    uint32_t *next,
    uint64_t accesses
)
{
    uint32_t index = 0;

    for (uint64_t i = 0; i < accesses; i++) {
        index = next[index];
    }

    final_index_sink = index;

    return index;
}




static void warm_chain(
    uint32_t *next,
    size_t nodes,
    int passes
)
{
    uint32_t index = 0;

    for (int pass = 0; pass < passes; pass++) {

        for (size_t i = 0; i < nodes; i++) {
            index = next[index];
        }
    }

    final_index_sink = index;
}


struct group_read {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t value[4];
};



static double scaled_value(
    uint64_t raw,
    uint64_t time_enabled,
    uint64_t time_running
)
{
    if (time_running == 0) {
        return 0.0;
    }

    return
        (double)raw *
        (double)time_enabled /
        (double)time_running;
}


static void print_group(
    const char *title,
    struct group_read *r,
    const char *names[],
    int count,
    uint64_t accesses
)
{
    printf("\n==================================================\n");
    printf("%s\n", title);
    printf("==================================================\n");

    printf(
        "time_enabled = %" PRIu64 "\n",
        r->time_enabled
    );

    printf(
        "time_running = %" PRIu64 "\n",
        r->time_running
    );

    if (r->time_enabled > 0) {

        double running_fraction =
            (double)r->time_running /
            (double)r->time_enabled;

        printf(
            "running fraction = %.4f\n",
            running_fraction
        );

        if (running_fraction < 0.95) {

            printf(
                "WARNING: counters were multiplexed "
                "significantly.\n"
            );
        }
    }

    printf("\n");

    for (int i = 0; i < count; i++) {

        double scaled = scaled_value(
            r->value[i],
            r->time_enabled,
            r->time_running
        );

        printf(
            "%-28s raw=%" PRIu64
            " scaled=%.2f"
            " per_access=%.6f\n",
            names[i],
            r->value[i],
            scaled,
            scaled / (double)accesses
        );
    }
}



static int run_group1(
    uint32_t *next,
    uint64_t accesses
)
{
    int fd[4];

    fd[0] = open_hardware_event(
        PERF_COUNT_HW_CPU_CYCLES,
        -1,
        1
    );

    if (fd[0] < 0) {
        perror("cycles");
        return -1;
    }

    fd[1] = open_hardware_event(
        PERF_COUNT_HW_INSTRUCTIONS,
        fd[0],
        0
    );

    fd[2] = open_cache_event(
        PERF_COUNT_HW_CACHE_L1D,
        PERF_COUNT_HW_CACHE_OP_READ,
        PERF_COUNT_HW_CACHE_RESULT_ACCESS,
        fd[0],
        0
    );

    fd[3] = open_cache_event(
        PERF_COUNT_HW_CACHE_L1D,
        PERF_COUNT_HW_CACHE_OP_READ,
        PERF_COUNT_HW_CACHE_RESULT_MISS,
        fd[0],
        0
    );

    for (int i = 1; i < 4; i++) {

        if (fd[i] < 0) {

            fprintf(
                stderr,
                "Group 1 event %d failed: %s\n",
                i,
                strerror(errno)
            );

            return -1;
        }
    }

    struct group_read r;

    memset(&r, 0, sizeof(r));

    ioctl(
        fd[0],
        PERF_EVENT_IOC_RESET,
        PERF_IOC_FLAG_GROUP
    );

    ioctl(
        fd[0],
        PERF_EVENT_IOC_ENABLE,
        PERF_IOC_FLAG_GROUP
    );

    run_pointer_chase(
        next,
        accesses
    );

    ioctl(
        fd[0],
        PERF_EVENT_IOC_DISABLE,
        PERF_IOC_FLAG_GROUP
    );

    ssize_t bytes = read(
        fd[0],
        &r,
        sizeof(r)
    );

    if (bytes < 0) {
        perror("read group1");
        return -1;
    }

    const char *names[4] = {
        "cycles",
        "instructions",
        "L1D_read_access",
        "L1D_read_miss"
    };

    print_group(
        "PMU GROUP 1",
        &r,
        names,
        4,
        accesses
    );

    if (r.value[2] != 0) {

        double miss_rate =
            (double)r.value[3] /
            (double)r.value[2];

        printf(
            "\nL1D load miss rate = %.6f (%.2f%%)\n",
            miss_rate,
            miss_rate * 100.0
        );
    }

    for (int i = 0; i < 4; i++) {
        close(fd[i]);
    }

    return 0;
}


static int run_group2(
    uint32_t *next,
    uint64_t accesses
)
{
    int fd[4];

    int event_count = 0;

    const char *names[4];

    fd[event_count] = open_cache_event(
        PERF_COUNT_HW_CACHE_LL,
        PERF_COUNT_HW_CACHE_OP_READ,
        PERF_COUNT_HW_CACHE_RESULT_ACCESS,
        -1,
        1
    );

    if (fd[event_count] < 0) {
        perror("LLC read access");
        return -1;
    }

    names[event_count] = "LLC_read_access";

    int leader_fd = fd[event_count];

    event_count++;


    fd[event_count] = open_cache_event(
        PERF_COUNT_HW_CACHE_LL,
        PERF_COUNT_HW_CACHE_OP_READ,
        PERF_COUNT_HW_CACHE_RESULT_MISS,
        leader_fd,
        0
    );

    if (fd[event_count] < 0) {

        perror("LLC read miss");

        close(leader_fd);

        return -1;
    }

    names[event_count] = "LLC_read_miss";

    event_count++;



    if (RAW_L2_EVENT_1 != 0) {

        fd[event_count] = open_raw_event(
            RAW_L2_EVENT_1,
            leader_fd,
            0
        );

        if (fd[event_count] >= 0) {

            names[event_count] =
                RAW_L2_NAME_1;

            event_count++;
        }

        else {

            fprintf(
                stderr,
                "Warning: %s unavailable.\n",
                RAW_L2_NAME_1
            );
        }
    }


    if (
        RAW_L2_EVENT_2 != 0 &&
        event_count < 4
    ) {

        fd[event_count] = open_raw_event(
            RAW_L2_EVENT_2,
            leader_fd,
            0
        );

        if (fd[event_count] >= 0) {

            names[event_count] =
                RAW_L2_NAME_2;

            event_count++;
        }

        else {

            fprintf(
                stderr,
                "Warning: %s unavailable.\n",
                RAW_L2_NAME_2
            );
        }
    }


    struct group_read r;

    memset(&r, 0, sizeof(r));

    ioctl(
        leader_fd,
        PERF_EVENT_IOC_RESET,
        PERF_IOC_FLAG_GROUP
    );

    ioctl(
        leader_fd,
        PERF_EVENT_IOC_ENABLE,
        PERF_IOC_FLAG_GROUP
    );

    run_pointer_chase(
        next,
        accesses
    );

    ioctl(
        leader_fd,
        PERF_EVENT_IOC_DISABLE,
        PERF_IOC_FLAG_GROUP
    );


    ssize_t bytes = read(
        leader_fd,
        &r,
        sizeof(r)
    );

    if (bytes < 0) {
        perror("read group2");
        return -1;
    }


    print_group(
        "PMU GROUP 2",
        &r,
        names,
        event_count,
        accesses
    );


    if (event_count >= 2 &&
        r.value[0] != 0) {

        double llc_miss_rate =
            (double)r.value[1] /
            (double)r.value[0];

        printf(
            "\nLLC load miss rate = %.6f (%.2f%%)\n",
            llc_miss_rate,
            llc_miss_rate * 100.0
        );
    }


    for (int i = 0; i < event_count; i++) {
        close(fd[i]);
    }

    return 0;
}



int main(
    int argc,
    char **argv
)
{
    if (argc != 4) {

        fprintf(
            stderr,
            "\nUsage:\n"
            "  %s FOOTPRINT_BYTES ACCESSES SEED\n\n",
            argv[0]
        );

        fprintf(
            stderr,
            "Example:\n"
            "  %s 65536 100000000 12345\n\n",
            argv[0]
        );

        return 1;
    }


    uint64_t footprint_bytes =
        strtoull(
            argv[1],
            NULL,
            0
        );

    uint64_t accesses =
        strtoull(
            argv[2],
            NULL,
            0
        );

    uint32_t seed =
        (uint32_t)strtoul(
            argv[3],
            NULL,
            0
        );


    if (
        footprint_bytes <
        2 * sizeof(uint32_t)
    ) {

        fprintf(
            stderr,
            "Footprint too small.\n"
        );

        return 1;
    }


    size_t nodes =
        footprint_bytes /
        sizeof(uint32_t);


    printf("\n");
    printf("==========================================\n");
    printf("Phase II PMU cache experiment\n");
    printf("==========================================\n");

    printf(
        "Requested footprint : %" PRIu64 " bytes\n",
        footprint_bytes
    );

    printf(
        "Nodes               : %zu\n",
        nodes
    );

    printf(
        "Actual footprint    : %zu bytes\n",
        nodes * sizeof(uint32_t)
    );

    printf(
        "Dependent accesses  : %" PRIu64 "\n",
        accesses
    );

    printf(
        "Seed                : %" PRIu32 "\n",
        seed
    );


    uint32_t *next =
        build_random_cycle(
            nodes,
            seed
        );

    if (next == NULL) {
        return 1;
    }


    /*
    Warm-up is OUTSIDE PMU measurement.
    */

    warm_chain(
        next,
        nodes,
        4
    );


    printf(
        "\nRunning PMU group 1...\n"
    );

    if (
        run_group1(
            next,
            accesses
        ) != 0
    ) {

        free(next);

        return 1;
    }


    /*
    Re-warm before the identical second run.
    */

    warm_chain(
        next,
        nodes,
        4
    );


    printf(
        "\nRunning PMU group 2...\n"
    );

    if (
        run_group2(
            next,
            accesses
        ) != 0
    ) {

        free(next);

        return 1;
    }


    printf(
        "\nFinal pointer index = %" PRIu32 "\n",
        final_index_sink
    );


    free(next);

    return 0;
}
