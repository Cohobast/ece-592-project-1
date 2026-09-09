/* Cache Line Size / Associativity Microbenchmark
* ECE 592 Project 1 - Homework I, Section 8.2 (items 3 and 4)
*
* Purpose:
* This program implements the two remaining Phase-I, timing-only
* reverse-engineering experiments that are not covered by
* cache_capacity_x86.c:
*
*   1. LINE / BLOCK SIZE ("stride" experiment)
*      For a fixed, controlled footprint, sweep the byte spacing
*      ("stride") between successive nodes of a randomized dependent
*      pointer-chase. Repeating this sweep for several candidate
*      strides at several footprints produces the "family of curves"
*      described in the handout (Figure 3): the spatial-locality
*      granularity at which neighboring accesses stop sharing a
*      fetched cache line shows up as a change in the working-set
*      size at which the latency curve for that stride departs from
*      the curves of smaller strides.
*
*   2. ASSOCIATIVITY ("conflict-set" experiment)
*      For a fixed candidate "set stride" (a byte increment believed
*      to map different addresses to the same cache set/index), build
*      a randomized dependent cycle over K congruent addresses and
*      sweep K (the number of simultaneously live conflicting lines).
*      A sudden increase in per-access latency after adding one more
*      congruent address is evidence for the number of ways at that
*      level.
*
* Both experiments reuse the same measurement discipline as the
* capacity benchmark:
*   - One randomized pointer-chase cycle per configuration; the next
*     address depends on the value returned by the previous load, so
*     ordinary memory-level parallelism cannot hide the result.
*   - Latency is measured as a BATCH of dependent accesses between one
*     serialized RDTSC/RDTSCP start/stop pair, divided by the number
*     of accesses in the batch. Do not time one access at a time; the
*     fence/timer overhead is tens of nanoseconds, larger than an L1
*     hit, and would swamp the signal (see Section 6.F of the
*     handout).
*   - At least 1,000,000 timed samples per configuration, full
*     distribution preserved (mean, median, stddev, Q1, Q3, P5, P95,
*     min, max, Tukey-IQR outlier count).
*   - No performance counters, no cache-topology files, and no
*     published/vendor cache specifications are read anywhere in this
*     file. This program is Phase-I timing-only.
*
* Build (Phase-I baseline, optimizations disabled per the handout):
*   gcc -O0 -g -std=c11 -Wall -Wextra -fno-omit-frame-pointer \
*       -o cache_line_assoc_x86 cache_line_assoc_x86.c
*
* Inspect the critical loop before trusting any result:
*   gcc -O0 -g -std=c11 -Wall -Wextra -fno-omit-frame-pointer \
*       -S -o cache_line_assoc_x86.s cache_line_assoc_x86.c
*   objdump -d -S ./cache_line_assoc_x86 > cache_line_assoc_x86.source.dis
*
* Pin before running (topology-identification commands only, Phase I):
*   taskset -c <logical_cpu> ./cache_line_assoc_x86 ...
* On Hazel:
*   srun --cpu-bind=verbose --cpu-bind=cores ./cache_line_assoc_x86 ...
*
* Usage:
*   Line-size / stride experiment:
*     cache_line_assoc_x86 stride single STRIDE_BYTES FOOTPRINT_BYTES SAMPLES STEPS SEED PREFIX
*     cache_line_assoc_x86 stride sweep  STRIDE_LIST MIN_FOOTPRINT MAX_FOOTPRINT SAMPLES STEPS SEED PREFIX
*       STRIDE_LIST is a comma-separated list of candidate byte jumps,
*       e.g. "8,16,32,64,128,256,512". Run this sweep once per stride
*       family; overlay the resulting curves to find the spatial
*       transition (Figure 3).
*
*   Associativity / conflict-set experiment:
*     cache_line_assoc_x86 assoc single SET_STRIDE_BYTES WAYS SAMPLES STEPS SEED PREFIX
*     cache_line_assoc_x86 assoc sweep  SET_STRIDE_BYTES MIN_WAYS MAX_WAYS SAMPLES STEPS SEED PREFIX
*       SET_STRIDE_BYTES is a candidate address increment hypothesized
*       to map to the same cache set/index (e.g. a candidate capacity
*       guess, a page-color stride, or 2^k values you are searching
*       over). Sweep WAYS from 1 upward; the eviction/latency jump
*       after WAYS exceeds the true number of ways is the signature.
*       Recommend choosing STEPS as a multiple of WAYS so each timed
*       batch corresponds to a whole number of passes through the
*       conflict cycle.
*
* Output (per configuration):
*   <prefix>_stride_summary.csv / <prefix>_assoc_summary.csv
*       one row per (stride, footprint) or (set_stride, ways) point
*   <prefix>_metadata.txt
*       experiment configuration, compiler/timer/affinity information
*   <prefix>_*_raw.csv
*       every individual timed sample for that configuration
*
* This file targets x86-64 only. A separate AArch64 implementation is
* required for the Arm/Ampere lab machine (Thunderbird), following the
* same experimental logic with the CNTVCT_EL0 generic timer in place
* of RDTSC/RDTSCP.
*/

#define _GNU_SOURCE

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#endif

#if !defined(__x86_64__) && !defined(_M_X64)
#error "This source file is for x86-64 only. Use a separate AArch64 implementation on Arm."
#endif

#define ALIGNMENT_BYTES 4096u
#define MIN_ASSIGNMENT_SAMPLES 1000000ULL
#define TIMER_OVERHEAD_SAMPLES 10000ULL
#define MAX_STRIDE_LIST_ENTRIES 64u

/* Pointer-chase node. Placed at caller-chosen byte offsets inside a
 * raw buffer (NOT necessarily contiguous struct-array elements), so
 * that arbitrary stride/set-stride spacing can be tested without the
 * compiler's own struct layout getting in the way. */
struct node {
    struct node *next;
};

/* Prevents the final pointer-chase result from becoming unused. */
static volatile struct node *sink_node;

typedef struct {
    double mean, median, stddev;
    double q1, q3, p5, p95;
    double min, max;
    uint64_t outliers;
    uint64_t n;
} stats_t;

typedef struct {
    uint64_t min_ticks;
    double mean_ticks;
    double median_ticks;
} overhead_stats_t;

/* Address of the i-th node inside a raw buffer, spaced by 'stride'
 * bytes. 'stride' must be >= sizeof(struct node). */
#define NODE_AT(buffer, stride, i) \
    ((struct node *)((buffer) + (size_t)(i) * (size_t)(stride)))

/* Compiler barrier only; this is not a CPU fence. */
static inline void compiler_barrier(void) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
#endif
}

/* Serialized x86 TSC timing primitives (Intel SDM: RDTSC is not
 * serializing; LFENCE is used to order it. RDTSCP additionally waits
 * for prior instructions to execute before reading the counter). */
static inline uint64_t x86_tsc_start(void) {
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}

static inline uint64_t x86_tsc_stop(void) {
    unsigned aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

/* Deterministic PRNG used for reproducible shuffling. */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Build one randomized closed cycle over 'node_count' nodes located
 * at byte offsets i*stride inside 'buffer', i = 0..node_count-1. The
 * randomization defeats simple stride/stream prefetchers so the
 * measured latency reflects true dependent-load cost rather than
 * prefetch-hidden throughput. */
static void build_strided_random_cycle(unsigned char *buffer, size_t stride,
                                       size_t node_count, uint32_t seed) {
    if (node_count < 2) {
        fprintf(stderr, "Need at least two nodes to form a cycle.\n");
        exit(EXIT_FAILURE);
    }
    if (stride < sizeof(struct node)) {
        fprintf(stderr,
                "Stride %zu bytes is smaller than sizeof(struct node)=%zu.\n",
                stride, sizeof(struct node));
        exit(EXIT_FAILURE);
    }

    size_t *order = malloc(node_count * sizeof(*order));
    if (!order) {
        perror("malloc order");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < node_count; i++) order[i] = i;

    uint32_t state = seed ? seed : 1u;
    for (size_t i = node_count - 1; i > 0; i--) {
        size_t j = (size_t)(xorshift32(&state) % (uint32_t)(i + 1));
        size_t temp = order[i];
        order[i] = order[j];
        order[j] = temp;
    }

    for (size_t i = 0; i < node_count; i++) {
        size_t current = order[i];
        size_t next = order[(i + 1) % node_count];
        NODE_AT(buffer, stride, current)->next = NODE_AT(buffer, stride, next);
    }

    free(order);
}

/* Untimed warm-up. The cursor is preserved for the next traversal. */
static void warm_chain(struct node **cursor, uint64_t steps) {
    struct node *p = *cursor;
    for (uint64_t i = 0; i < steps; i++) p = p->next;
    sink_node = p;
    *cursor = p;
}

/* One call = one timed sample. A sample times a batch of dependent
 * loads and returns TSC ticks/access. This is the primary latency
 * method required by the handout (Section 6.F): batching amortizes
 * fence/timer overhead toward zero and preserves the true dependency
 * chain, unlike a loop of independent loads. */
static double chase_batch(struct node **cursor, uint64_t steps) {
    struct node *p = *cursor;

    compiler_barrier();
    uint64_t t0 = x86_tsc_start();

    for (uint64_t i = 0; i < steps; i++) {
        p = p->next;
    }

    uint64_t t1 = x86_tsc_stop();
    compiler_barrier();

    sink_node = p;
    *cursor = p;
    return (double)(t1 - t0) / (double)steps;
}

/* Empty timed region for timer/barrier-overhead characterization. */
static uint64_t timer_once(void) {
    compiler_barrier();
    uint64_t t0 = x86_tsc_start();
    uint64_t t1 = x86_tsc_stop();
    compiler_barrier();
    return t1 - t0;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Linear-interpolated quantile of an already sorted array. */
static double quantile_linear(const double *sorted, size_t n, double q) {
    if (n == 0) return NAN;
    if (n == 1) return sorted[0];

    double position = q * (double)(n - 1);
    size_t lower = (size_t)floor(position);
    size_t upper = (size_t)ceil(position);
    double fraction = position - (double)lower;

    if (lower == upper) return sorted[lower];
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

/* Statistics over the complete timed-sample distribution. */
static stats_t compute_stats(const double *samples, size_t n) {
    stats_t s;
    memset(&s, 0, sizeof(s));
    s.n = (uint64_t)n;
    if (n == 0) return s;

    double mean = 0.0, m2 = 0.0;
    double min_value = samples[0], max_value = samples[0];

    for (size_t i = 0; i < n; i++) {
        double x = samples[i];
        if (x < min_value) min_value = x;
        if (x > max_value) max_value = x;

        double delta = x - mean;
        mean += delta / (double)(i + 1);
        double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double *sorted = malloc(n * sizeof(*sorted));
    if (!sorted) {
        perror("malloc sorted samples");
        exit(EXIT_FAILURE);
    }

    memcpy(sorted, samples, n * sizeof(*sorted));
    qsort(sorted, n, sizeof(*sorted), cmp_double);

    s.mean = mean;
    s.stddev = (n > 1) ? sqrt(m2 / (double)(n - 1)) : 0.0;
    s.median = quantile_linear(sorted, n, 0.50);
    s.q1 = quantile_linear(sorted, n, 0.25);
    s.q3 = quantile_linear(sorted, n, 0.75);
    s.p5 = quantile_linear(sorted, n, 0.05);
    s.p95 = quantile_linear(sorted, n, 0.95);
    s.min = min_value;
    s.max = max_value;

    /* Tukey 1.5*IQR outlier count. */
    double iqr = s.q3 - s.q1;
    double lower_limit = s.q1 - 1.5 * iqr;
    double upper_limit = s.q3 + 1.5 * iqr;

    for (size_t i = 0; i < n; i++) {
        if (samples[i] < lower_limit || samples[i] > upper_limit) s.outliers++;
    }

    free(sorted);
    return s;
}

static overhead_stats_t measure_timer_overhead(void) {
    double *values = malloc((size_t)TIMER_OVERHEAD_SAMPLES * sizeof(*values));
    if (!values) {
        perror("malloc timer overhead");
        exit(EXIT_FAILURE);
    }

    uint64_t minimum = UINT64_MAX;
    long double sum = 0.0L;

    for (uint64_t i = 0; i < TIMER_OVERHEAD_SAMPLES; i++) {
        uint64_t value = timer_once();
        values[i] = (double)value;
        if (value < minimum) minimum = value;
        sum += (long double)value;
    }

    qsort(values, (size_t)TIMER_OVERHEAD_SAMPLES, sizeof(*values), cmp_double);

    overhead_stats_t result;
    result.min_ticks = minimum;
    result.mean_ticks = (double)(sum / (long double)TIMER_OVERHEAD_SAMPLES);
    result.median_ticks = quantile_linear(values, (size_t)TIMER_OVERHEAD_SAMPLES, 0.50);

    free(values);
    return result;
}

static uint64_t parse_u64(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }

    return (uint64_t)value;
}

/* Parse a comma-separated list of byte strides, e.g. "8,16,32,64". */
static size_t parse_stride_list(const char *text, size_t *out, size_t max_out) {
    size_t count = 0;
    const char *cursor = text;

    while (*cursor != '\0') {
        while (*cursor == ',' || isspace((unsigned char)*cursor)) cursor++;
        if (*cursor == '\0') break;

        char *end = NULL;
        errno = 0;
        unsigned long long value = strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor) {
            fprintf(stderr, "Invalid stride list entry near: %s\n", cursor);
            exit(EXIT_FAILURE);
        }
        if (count == max_out) {
            fprintf(stderr, "Too many strides in STRIDE_LIST (max %zu).\n", max_out);
            exit(EXIT_FAILURE);
        }
        out[count++] = (size_t)value;
        cursor = end;
    }

    if (count == 0) {
        fprintf(stderr, "STRIDE_LIST must contain at least one value.\n");
        exit(EXIT_FAILURE);
    }
    return count;
}

/* Page-aligned raw byte buffer. Node placement inside it is done by
 * the caller via NODE_AT so arbitrary stride/set-stride spacing can
 * be tested. */
static unsigned char *allocate_bytes(size_t bytes) {
    unsigned char *buffer = NULL;

#ifdef _WIN32
    buffer = (unsigned char *)_aligned_malloc(bytes, ALIGNMENT_BYTES);
    if (!buffer) {
        fprintf(stderr, "Aligned allocation failed for %zu bytes.\n", bytes);
        exit(EXIT_FAILURE);
    }
#else
    int result = posix_memalign((void **)&buffer, ALIGNMENT_BYTES, bytes);
    if (result != 0 || !buffer) {
        fprintf(stderr, "posix_memalign failed for %zu bytes (error=%d).\n", bytes, result);
        exit(EXIT_FAILURE);
    }
#endif

    return buffer;
}

static void free_bytes(unsigned char *buffer) {
#ifdef _WIN32
    _aligned_free(buffer);
#else
    free(buffer);
#endif
}

/* Verify that a Linux run is pinned to exactly one logical CPU. */
static void print_affinity_information(FILE *metadata) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);

    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        fprintf(metadata, "affinity_check=failed\n");
        fprintf(stderr, "WARNING: sched_getaffinity failed. Verify taskset manually.\n");
        return;
    }

    int count = CPU_COUNT(&set);
    fprintf(metadata, "allowed_logical_cpu_count=%d\n", count);
    fprintf(metadata, "allowed_logical_cpus=");

    int first = 1;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &set)) {
            fprintf(metadata, "%s%d", first ? "" : ",", cpu);
            first = 0;
        }
    }
    fprintf(metadata, "\n");

    if (count != 1) {
        fprintf(stderr,
                "\nWARNING:\n"
                "This process is currently allowed to run on %d logical CPUs.\n"
                "For final assignment measurements, launch with:\n\n"
                "    taskset -c <chosen_cpu> ./cache_line_assoc_x86 ...\n\n",
                count);
    }
#else
    fprintf(metadata, "affinity_check=not_available_on_this_platform\n");
#endif
}

/* Record run-level metadata used for reproducibility. */
static void write_global_metadata(FILE *metadata, int argc, char **argv,
                                  const char *experiment, uint64_t sample_count,
                                  uint64_t steps, uint32_t seed,
                                  const overhead_stats_t *overhead) {
    time_t now = time(NULL);
    struct tm *utc = gmtime(&now);
    char timestamp[64] = "unknown";

    if (utc) {
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", utc);
    }

    fprintf(metadata, "timestamp_utc=%s\n", timestamp);
    fprintf(metadata, "experiment=%s\n", experiment);
    fprintf(metadata, "architecture=x86-64\n");
    fprintf(metadata, "timer=serialized_rdtsc_rdtscp\n");
    fprintf(metadata, "latency_unit=tsc_ticks_per_access\n");
    fprintf(metadata, "alignment_bytes=%u\n", ALIGNMENT_BYTES);
    fprintf(metadata, "timed_samples_per_point=%" PRIu64 "\n", sample_count);
    fprintf(metadata, "dependent_accesses_per_timed_sample=%" PRIu64 "\n", steps);
    fprintf(metadata, "seed=%" PRIu32 "\n", seed);
    fprintf(metadata, "timer_overhead_samples=%llu\n",
            (unsigned long long)TIMER_OVERHEAD_SAMPLES);
    fprintf(metadata, "timer_overhead_min_tsc_ticks=%" PRIu64 "\n", overhead->min_ticks);
    fprintf(metadata, "timer_overhead_mean_tsc_ticks=%.6f\n", overhead->mean_ticks);
    fprintf(metadata, "timer_overhead_median_tsc_ticks=%.6f\n", overhead->median_ticks);

#ifdef __linux__
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[sizeof(hostname) - 1] = '\0';
        fprintf(metadata, "hostname=%s\n", hostname);
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size > 0) fprintf(metadata, "page_size_bytes=%ld\n", page_size);
#endif

    fprintf(metadata, "command_line=");
    for (int i = 0; i < argc; i++) {
        fprintf(metadata, "%s%s", i == 0 ? "" : " ", argv[i]);
    }
    fprintf(metadata, "\n");

    print_affinity_information(metadata);

    fprintf(metadata, "assignment_sample_count_status=%s\n",
            sample_count < MIN_ASSIGNMENT_SAMPLES
                ? "TEST_ONLY_BELOW_1000000"
                : "MEETS_1000000_MINIMUM");
}

/* Save every timed sample so the full distribution can be reconstructed. */
static void write_raw_csv(const char *path, const double *samples, uint64_t sample_count) {
    FILE *file = fopen(path, "w");
    if (!file) {
        perror(path);
        exit(EXIT_FAILURE);
    }

    setvbuf(file, NULL, _IOFBF, 1024 * 1024);
    fprintf(file, "sample,latency_tsc_ticks_per_access\n");

    for (uint64_t i = 0; i < sample_count; i++) {
        fprintf(file, "%" PRIu64 ",%.9f\n", i, samples[i]);
    }

    fclose(file);
}

/* -------------------------------------------------------------------
 * Experiment 1: Line / block size (stride sweep)
 *
 * Footprint is held to the value requested for this point; the number
 * of distinct nodes touched is footprint/stride. Nodes are visited in
 * a randomized dependent cycle. Run this at several candidate strides
 * across the same footprint sweep to produce the family-of-curves
 * plot (Figure 3): the stride at which the footprint-vs-latency curve
 * stops matching the finer-grained strides, and starts requiring a
 * larger nominal footprint to reach the same transition, is evidence
 * for the true cache line size.
 * ------------------------------------------------------------------- */
static stats_t run_stride_point(size_t stride, size_t requested_footprint,
                                uint64_t sample_count, uint64_t steps,
                                uint32_t seed, const char *prefix, FILE *summary) {
    size_t node_count = requested_footprint / stride;
    if (node_count < 2) {
        fprintf(stderr,
                "Skipping stride=%zu footprint=%zu: fewer than 2 nodes result.\n",
                stride, requested_footprint);
        stats_t empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }

    size_t actual_footprint = node_count * stride;
    unsigned char *buffer = allocate_bytes(actual_footprint);

    /* First-touch allocation after the process has been externally pinned. */
    memset(buffer, 0, actual_footprint);

    build_strided_random_cycle(buffer, stride, node_count, seed);

    struct node *cursor = NODE_AT(buffer, stride, 0);

    uint64_t warm_steps = node_count > UINT64_MAX / 4ULL
                              ? UINT64_MAX
                              : (uint64_t)node_count * 4ULL;
    warm_chain(&cursor, warm_steps);

    if (sample_count > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Sample buffer is too large.\n");
        exit(EXIT_FAILURE);
    }

    double *samples = malloc((size_t)sample_count * sizeof(*samples));
    if (!samples) {
        fprintf(stderr, "Could not allocate sample buffer for %" PRIu64 " samples.\n",
                sample_count);
        exit(EXIT_FAILURE);
    }

    /* Critical timing loop: intentionally no file I/O here. */
    for (uint64_t sample = 0; sample < sample_count; sample++) {
        samples[sample] = chase_batch(&cursor, steps);
    }

    stats_t statistics = compute_stats(samples, (size_t)sample_count);

    char raw_path[1024];
    int written = snprintf(raw_path, sizeof(raw_path),
                           "%s_stride%zu_footprint%zu_raw.csv",
                           prefix, stride, actual_footprint);
    if (written < 0 || (size_t)written >= sizeof(raw_path)) {
        fprintf(stderr, "Output path is too long.\n");
        free(samples);
        free_bytes(buffer);
        exit(EXIT_FAILURE);
    }

    write_raw_csv(raw_path, samples, sample_count);

    fprintf(summary,
            "%zu,%zu,%zu,%zu,"
            "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
            "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
            "%" PRIu64 ",%" PRIu32 ",%s\n",
            stride, requested_footprint, actual_footprint, node_count,
            sample_count, steps, sample_count * steps,
            statistics.mean, statistics.median, statistics.stddev,
            statistics.q1, statistics.q3, statistics.p5, statistics.p95,
            statistics.min, statistics.max, statistics.outliers,
            seed, raw_path);
    fflush(summary);

    fprintf(stderr,
            "stride=%zu footprint=%zu nodes=%zu mean=%.4f median=%.4f "
            "p5=%.4f p95=%.4f raw=%s\n",
            stride, actual_footprint, node_count,
            statistics.mean, statistics.median,
            statistics.p5, statistics.p95, raw_path);

    free(samples);
    free_bytes(buffer);
    return statistics;
}

/* -------------------------------------------------------------------
 * Experiment 2: Associativity (conflict-set sweep)
 *
 * 'set_stride' is a candidate byte increment hypothesized to map
 * different addresses to the same cache set/index. 'ways' congruent
 * addresses are linked into one randomized dependent cycle and
 * repeatedly traversed. If ways <= true associativity, all 'ways'
 * lines can remain simultaneously resident and steady-state latency
 * stays low; once ways exceeds the true associativity, addresses
 * begin evicting one another and latency rises sharply. The buffer is
 * allocated once for the whole sweep (sized for the maximum WAYS
 * requested) so that low-order address bits stay identical across
 * every K in the sweep -- only the number of live congruent lines
 * changes.
 * ------------------------------------------------------------------- */
static stats_t run_assoc_point(unsigned char *buffer, size_t set_stride, size_t ways,
                               uint64_t sample_count, uint64_t steps,
                               uint32_t seed, const char *prefix, FILE *summary) {
    if (ways < 1) {
        fprintf(stderr, "WAYS must be at least 1.\n");
        exit(EXIT_FAILURE);
    }

    if (ways == 1) {
        /* A one-node "cycle" (self-loop) is a degenerate case; use it
         * only as a same-line-reload baseline and skip pointer-cycle
         * construction that requires >= 2 nodes. */
        struct node *only = NODE_AT(buffer, set_stride, 0);
        memset(only, 0, sizeof(*only));
        only->next = only;

        struct node *cursor = only;
        uint64_t warm_steps = 64;
        warm_chain(&cursor, warm_steps);

        double *samples = malloc((size_t)sample_count * sizeof(*samples));
        if (!samples) {
            fprintf(stderr, "Could not allocate sample buffer.\n");
            exit(EXIT_FAILURE);
        }
        for (uint64_t sample = 0; sample < sample_count; sample++) {
            samples[sample] = chase_batch(&cursor, steps);
        }

        stats_t statistics = compute_stats(samples, (size_t)sample_count);

        char raw_path[1024];
        int written = snprintf(raw_path, sizeof(raw_path),
                               "%s_setstride%zu_ways%zu_raw.csv",
                               prefix, set_stride, ways);
        if (written < 0 || (size_t)written >= sizeof(raw_path)) {
            fprintf(stderr, "Output path is too long.\n");
            free(samples);
            exit(EXIT_FAILURE);
        }
        write_raw_csv(raw_path, samples, sample_count);

        fprintf(summary,
                "%zu,%zu,"
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
                "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
                "%" PRIu64 ",%" PRIu32 ",%s\n",
                set_stride, ways,
                sample_count, steps, sample_count * steps,
                statistics.mean, statistics.median, statistics.stddev,
                statistics.q1, statistics.q3, statistics.p5, statistics.p95,
                statistics.min, statistics.max, statistics.outliers,
                seed, raw_path);
        fflush(summary);

        fprintf(stderr,
                "set_stride=%zu ways=%zu mean=%.4f median=%.4f p5=%.4f p95=%.4f raw=%s\n",
                set_stride, ways, statistics.mean, statistics.median,
                statistics.p5, statistics.p95, raw_path);

        free(samples);
        return statistics;
    }

    /* Re-touch and rebuild the cycle over exactly the first 'ways'
     * congruent addresses so their absolute addresses (and therefore
     * the address bits presumed to select the cache set) stay fixed
     * across every point in the sweep. */
    memset(buffer, 0, ways * set_stride);
    build_strided_random_cycle(buffer, set_stride, ways, seed);

    struct node *cursor = NODE_AT(buffer, set_stride, 0);

    /* Warm through several full passes of the conflict cycle so the
     * measurement reflects steady-state conflict behavior rather than
     * first-touch cold misses. */
    uint64_t warm_steps = (uint64_t)ways * 8ULL;
    warm_chain(&cursor, warm_steps);

    if (steps % ways != 0) {
        fprintf(stderr,
                "NOTE: STEPS=%" PRIu64 " is not a multiple of WAYS=%zu; "
                "each timed batch will not cover a whole number of passes "
                "through the conflict cycle.\n",
                steps, ways);
    }

    if (sample_count > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Sample buffer is too large.\n");
        exit(EXIT_FAILURE);
    }

    double *samples = malloc((size_t)sample_count * sizeof(*samples));
    if (!samples) {
        fprintf(stderr, "Could not allocate sample buffer for %" PRIu64 " samples.\n",
                sample_count);
        exit(EXIT_FAILURE);
    }

    /* Critical timing loop: intentionally no file I/O here. */
    for (uint64_t sample = 0; sample < sample_count; sample++) {
        samples[sample] = chase_batch(&cursor, steps);
    }

    stats_t statistics = compute_stats(samples, (size_t)sample_count);

    char raw_path[1024];
    int written = snprintf(raw_path, sizeof(raw_path),
                           "%s_setstride%zu_ways%zu_raw.csv",
                           prefix, set_stride, ways);
    if (written < 0 || (size_t)written >= sizeof(raw_path)) {
        fprintf(stderr, "Output path is too long.\n");
        free(samples);
        exit(EXIT_FAILURE);
    }

    write_raw_csv(raw_path, samples, sample_count);

    fprintf(summary,
            "%zu,%zu,"
            "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
            "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
            "%" PRIu64 ",%" PRIu32 ",%s\n",
            set_stride, ways,
            sample_count, steps, sample_count * steps,
            statistics.mean, statistics.median, statistics.stddev,
            statistics.q1, statistics.q3, statistics.p5, statistics.p95,
            statistics.min, statistics.max, statistics.outliers,
            seed, raw_path);
    fflush(summary);

    fprintf(stderr,
            "set_stride=%zu ways=%zu mean=%.4f median=%.4f p5=%.4f p95=%.4f raw=%s\n",
            set_stride, ways, statistics.mean, statistics.median,
            statistics.p5, statistics.p95, raw_path);

    free(samples);
    return statistics;
}

static void usage(const char *program) {
    fprintf(stderr,
            "\nUSAGE:\n\n"
            "Line / block size (stride) experiment:\n"
            "  %s stride single STRIDE_BYTES FOOTPRINT_BYTES SAMPLES STEPS SEED PREFIX\n"
            "  %s stride sweep  STRIDE_LIST MIN_FOOTPRINT MAX_FOOTPRINT SAMPLES STEPS SEED PREFIX\n"
            "      STRIDE_LIST is comma-separated, e.g. \"8,16,32,64,128,256,512\"\n"
            "      Footprint is swept coarsely by powers of two from MIN to MAX.\n\n"
            "Associativity (conflict-set) experiment:\n"
            "  %s assoc single SET_STRIDE_BYTES WAYS SAMPLES STEPS SEED PREFIX\n"
            "  %s assoc sweep  SET_STRIDE_BYTES MIN_WAYS MAX_WAYS SAMPLES STEPS SEED PREFIX\n"
            "      WAYS is swept by 1 from MIN_WAYS to MAX_WAYS.\n\n"
            "Examples (functionality/debug only -- below the 1,000,000-sample minimum):\n"
            "  %s stride single 64 4096 1000 128 12345 test/line\n"
            "  %s stride sweep \"8,16,32,64,128,256\" 512 65536 1000 128 12345 test/line\n"
            "  %s assoc single 4096 8 1000 128 12345 test/assoc\n"
            "  %s assoc sweep 4096 1 16 1000 128 12345 test/assoc\n\n"
            "Assignment-scale examples:\n"
            "  %s stride sweep \"8,16,32,64,128,256,512\" 1024 4194304 1000000 128 12345 data/line\n"
            "  %s assoc sweep 32768 1 24 1000000 128 12345 data/assoc\n\n",
            program, program, program, program,
            program, program, program, program,
            program, program);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    int is_stride = strcmp(argv[1], "stride") == 0;
    int is_assoc = strcmp(argv[1], "assoc") == 0;
    int is_single = strcmp(argv[2], "single") == 0;
    int is_sweep = strcmp(argv[2], "sweep") == 0;

    if ((!is_stride && !is_assoc) || (!is_single && !is_sweep)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    if ((is_stride && is_single && argc != 9) ||
        (is_stride && is_sweep && argc != 10) ||
        (is_assoc && is_single && argc != 9) ||
        (is_assoc && is_sweep && argc != 10)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    uint64_t sample_count, steps;
    uint32_t seed;
    const char *prefix;

    overhead_stats_t overhead = measure_timer_overhead();

    if (is_stride) {
        size_t strides[MAX_STRIDE_LIST_ENTRIES];
        size_t stride_count;
        size_t single_footprint = 0, min_footprint = 0, max_footprint = 0;

        if (is_single) {
            strides[0] = (size_t)parse_u64(argv[3], "STRIDE_BYTES");
            stride_count = 1;
            single_footprint = (size_t)parse_u64(argv[4], "FOOTPRINT_BYTES");
            sample_count = parse_u64(argv[5], "SAMPLES");
            steps = parse_u64(argv[6], "STEPS");
            seed = (uint32_t)parse_u64(argv[7], "SEED");
            prefix = argv[8];
        } else {
            stride_count = parse_stride_list(argv[3], strides, MAX_STRIDE_LIST_ENTRIES);
            min_footprint = (size_t)parse_u64(argv[4], "MIN_FOOTPRINT");
            max_footprint = (size_t)parse_u64(argv[5], "MAX_FOOTPRINT");
            sample_count = parse_u64(argv[6], "SAMPLES");
            steps = parse_u64(argv[7], "STEPS");
            seed = (uint32_t)parse_u64(argv[8], "SEED");
            prefix = argv[9];
        }

        if (sample_count == 0 || steps == 0) {
            fprintf(stderr, "SAMPLES and STEPS must both be greater than zero.\n");
            return EXIT_FAILURE;
        }
        if (sample_count < MIN_ASSIGNMENT_SAMPLES) {
            fprintf(stderr,
                    "\nWARNING:\nSAMPLES=%" PRIu64 " is below the assignment minimum of %llu.\n"
                    "Use this only as a functionality/debugging run, not as final project data.\n\n",
                    sample_count, (unsigned long long)MIN_ASSIGNMENT_SAMPLES);
        }

        char metadata_path[1024], summary_path[1024];
        snprintf(metadata_path, sizeof(metadata_path), "%s_metadata.txt", prefix);
        snprintf(summary_path, sizeof(summary_path), "%s_stride_summary.csv", prefix);

        FILE *metadata = fopen(metadata_path, "w");
        if (!metadata) { perror(metadata_path); return EXIT_FAILURE; }
        write_global_metadata(metadata, argc, argv, "line_size_stride_sweep",
                              sample_count, steps, seed, &overhead);
        fclose(metadata);

        FILE *summary = fopen(summary_path, "w");
        if (!summary) { perror(summary_path); return EXIT_FAILURE; }
        fprintf(summary,
                "stride_bytes,requested_footprint_bytes,actual_footprint_bytes,nodes,"
                "timed_samples,dependent_accesses_per_sample,total_dependent_accesses,"
                "mean_tsc_ticks_per_access,median_tsc_ticks_per_access,stddev,"
                "q1,q3,p5,p95,min,max,outliers,seed,raw_file\n");

        if (is_single) {
            run_stride_point(strides[0], single_footprint, sample_count, steps,
                             seed, prefix, summary);
        } else {
            if (min_footprint == 0 || max_footprint == 0 || min_footprint > max_footprint) {
                fprintf(stderr, "Require 0 < MIN_FOOTPRINT <= MAX_FOOTPRINT.\n");
                fclose(summary);
                return EXIT_FAILURE;
            }

            for (size_t s = 0; s < stride_count; s++) {
                size_t footprint = min_footprint;
                while (footprint <= max_footprint) {
                    fprintf(stderr, "[stride %zu/%zu] stride_bytes=%zu footprint_bytes=%zu\n",
                            s + 1, stride_count, strides[s], footprint);
                    run_stride_point(strides[s], footprint, sample_count, steps,
                                     seed, prefix, summary);
                    if (footprint > max_footprint / 2 || footprint > SIZE_MAX / 2) break;
                    footprint *= 2;
                }
            }
        }

        fclose(summary);
        fprintf(stderr, "\nStride experiment finished.\nSummary file:  %s\nMetadata file: %s\n",
                summary_path, metadata_path);
        return EXIT_SUCCESS;
    }

    /* is_assoc */
    size_t set_stride, single_ways = 0, min_ways = 0, max_ways = 0;

    if (is_single) {
        set_stride = (size_t)parse_u64(argv[3], "SET_STRIDE_BYTES");
        single_ways = (size_t)parse_u64(argv[4], "WAYS");
        sample_count = parse_u64(argv[5], "SAMPLES");
        steps = parse_u64(argv[6], "STEPS");
        seed = (uint32_t)parse_u64(argv[7], "SEED");
        prefix = argv[8];
    } else {
        set_stride = (size_t)parse_u64(argv[3], "SET_STRIDE_BYTES");
        min_ways = (size_t)parse_u64(argv[4], "MIN_WAYS");
        max_ways = (size_t)parse_u64(argv[5], "MAX_WAYS");
        sample_count = parse_u64(argv[6], "SAMPLES");
        steps = parse_u64(argv[7], "STEPS");
        seed = (uint32_t)parse_u64(argv[8], "SEED");
        prefix = argv[9];
    }

    if (set_stride < sizeof(struct node)) {
        fprintf(stderr, "SET_STRIDE_BYTES must be >= %zu.\n", sizeof(struct node));
        return EXIT_FAILURE;
    }
    if (sample_count == 0 || steps == 0) {
        fprintf(stderr, "SAMPLES and STEPS must both be greater than zero.\n");
        return EXIT_FAILURE;
    }
    if (sample_count < MIN_ASSIGNMENT_SAMPLES) {
        fprintf(stderr,
                "\nWARNING:\nSAMPLES=%" PRIu64 " is below the assignment minimum of %llu.\n"
                "Use this only as a functionality/debugging run, not as final project data.\n\n",
                sample_count, (unsigned long long)MIN_ASSIGNMENT_SAMPLES);
    }

    size_t max_ways_needed = is_single ? single_ways : max_ways;
    if (max_ways_needed < 1) {
        fprintf(stderr, "WAYS/MAX_WAYS must be at least 1.\n");
        return EXIT_FAILURE;
    }
    if (!is_single && (min_ways < 1 || min_ways > max_ways)) {
        fprintf(stderr, "Require 1 <= MIN_WAYS <= MAX_WAYS.\n");
        return EXIT_FAILURE;
    }

    char metadata_path[1024], summary_path[1024];
    snprintf(metadata_path, sizeof(metadata_path), "%s_metadata.txt", prefix);
    snprintf(summary_path, sizeof(summary_path), "%s_assoc_summary.csv", prefix);

    FILE *metadata = fopen(metadata_path, "w");
    if (!metadata) { perror(metadata_path); return EXIT_FAILURE; }
    write_global_metadata(metadata, argc, argv, "associativity_conflict_sweep",
                          sample_count, steps, seed, &overhead);
    fclose(metadata);

    FILE *summary = fopen(summary_path, "w");
    if (!summary) { perror(summary_path); return EXIT_FAILURE; }
    fprintf(summary,
            "set_stride_bytes,ways,"
            "timed_samples,dependent_accesses_per_sample,total_dependent_accesses,"
            "mean_tsc_ticks_per_access,median_tsc_ticks_per_access,stddev,"
            "q1,q3,p5,p95,min,max,outliers,seed,raw_file\n");

    /* Allocate one buffer sized for the largest WAYS requested so
     * every point in the sweep uses the same base address and the
     * same low-order address bits; only the number of live congruent
     * lines (the visited prefix of the buffer) changes between
     * points. */
    size_t buffer_bytes = max_ways_needed * set_stride;
    unsigned char *buffer = allocate_bytes(buffer_bytes);
    memset(buffer, 0, buffer_bytes);

    if (is_single) {
        run_assoc_point(buffer, set_stride, single_ways, sample_count, steps,
                        seed, prefix, summary);
    } else {
        for (size_t ways = min_ways; ways <= max_ways; ways++) {
            fprintf(stderr, "[ways %zu/%zu] set_stride_bytes=%zu\n",
                    ways, max_ways, set_stride);
            run_assoc_point(buffer, set_stride, ways, sample_count, steps,
                            seed, prefix, summary);
        }
    }

    free_bytes(buffer);
    fclose(summary);

    fprintf(stderr, "\nAssociativity experiment finished.\nSummary file:  %s\nMetadata file: %s\n",
            summary_path, metadata_path);
    return EXIT_SUCCESS;
}
