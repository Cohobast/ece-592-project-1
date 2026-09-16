/*
 * phase2_verify.c  (v2)
 *
 * Phase II ONLY. Do not run this before your Phase-I timing-only results
 * are frozen and committed (tag e.g. phase1-timing-only first).
 *
 * Changes from v1, after a group of us reviewed both a from-scratch attempt
 * and this file side by side against a real failure on Sunbird:
 *
 *   1. FIXED: `capacity <footprint> <steps>` previously read the wrong argv
 *      index and silently ignored the requested step count (off-by-one --
 *      the capacity mode has one fewer required positional arg than
 *      stride/assoc, and the optional-steps check wasn't adjusted for it).
 *
 *   2. FIXED: results are now looked up BY NAME after each group read,
 *      never by a positional index that silently shifts if an earlier
 *      event failed to open. A mismatched ratio computed from shifted
 *      indices would look plausible and be wrong -- exactly the failure
 *      mode to avoid in a report.
 *
 *   3. CHANGED: counters are no longer requested as one 8-event group.
 *      They're split into small groups (2 events each) that each rerun
 *      the identical workload. This trades wall-clock time (each group
 *      reruns the chase) for a much lower chance of the group failing to
 *      schedule or being heavily multiplexed on machines with few usable
 *      general-purpose PMU counters per logical CPU (e.g. under SMT).
 *      IMPORTANT: if a bare `perf stat -e cycles sleep 1` also shows
 *      0.00% on your machine, splitting groups will not fix it -- that
 *      points at perf_event_paranoid, the NMI watchdog reserving a
 *      counter, or a virtualized/PMU-passthrough-disabled host. Check
 *      those before assuming group size was the whole problem.
 *
 *   4. ADDED: optional raw L2 event pair (PERF_TYPE_RAW), left at 0x0
 *      until you fill in the vendor-documented encoding for the specific
 *      machine under test (e.g. via `perf list --details` /
 *      `l2_rqsts.demand_data_rd_hit` and its miss counterpart on Intel).
 *      Do NOT copy an encoding from one machine's datasheet onto another.
 *      Generic PERF_TYPE_HW_CACHE has no L2 entry in the kernel's
 *      cache-id enum (only L1D, L1I, LL, DTLB, ITLB, BPU, NODE) -- there
 *      is no portable "generic L2" event to fall back on.
 *
 * Build (x86-64 example):
 *   gcc -O0 -g -std=c11 -Wall -Wextra -fno-omit-frame-pointer \
 *       -o phase2_verify phase2_verify.c
 *
 * Run (pin to one logical CPU, exactly like Phase I):
 *   taskset -c 4 ./phase2_verify capacity  4096  1000000
 *   taskset -c 4 ./phase2_verify stride    64    4194304
 *   taskset -c 4 ./phase2_verify assoc     8     1048576   4096
 *
 * Mode-specific positional args:
 *   capacity <footprint_bytes> <steps>
 *   stride   <stride_bytes>    <footprint_bytes> [steps]
 *   assoc    <n_conflicting_lines> <candidate_set_stride_bytes> <line_size_bytes> [steps]
 *
 * Before trusting a raw L2 pair on a new machine, fill in RAW_L2_EVENT_1/2
 * below and rebuild -- leave them at 0x0 to skip that group entirely.
 *
 * CSV columns (one row per invocation):
 *   mode,param1,param2,param3,n_samples,elapsed_ns,
 *   cycles,instructions,cache_refs,cache_misses,
 *   l1d_loads,l1d_misses,ll_loads,ll_misses,
 *   raw_l2_1,raw_l2_2,
 *   any_group_multiplexed,cache_miss_rate,l1d_miss_rate,ll_miss_rate
 *
 * A value is emitted as -1 if that event was never successfully opened
 * AND recorded on this machine -- check stderr for exactly which named
 * event failed and why (permissions vs. unsupported event).
 */

#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pmu_counters.h"
#include "pmu_events_hwcache.h"

#define DEFAULT_N_SAMPLES 1000000ULL

/* --------------------------------------------------------------------
 * RAW L2 EVENT CONFIGURATION -- leave at 0x0 until you have looked up
 * the correct encoding for the specific machine under test via
 * `perf list --details` and vendor documentation. Set both to nonzero
 * to enable the raw-L2 group; leave either at 0x0 and that group is
 * skipped (fields print as -1).
 * -------------------------------------------------------------------- */
#define RAW_L2_EVENT_1 0x0ULL
#define RAW_L2_EVENT_2 0x0ULL
#define RAW_L2_NAME_1  "raw_l2_1"
#define RAW_L2_NAME_2  "raw_l2_2"

struct node {
    struct node *next;
};

static volatile struct node *g_sink;

/* -------- name -> value lookup table, filled in as groups are read -------
 * This replaces the v1 approach of assuming a fixed positional index per
 * event. Every group read calls record_result() once per successfully
 * opened event in that group; lookups later go by name and return -1 if
 * that name was never recorded (event failed to open, or its group failed
 * to schedule at all). */
#define MAX_RESULTS 16
typedef struct {
    const char *name;
    double scaled;
    uint64_t raw;
    int multiplexed;
} named_result_t;

static named_result_t g_results[MAX_RESULTS];
static int g_n_results = 0;
static int g_any_multiplexed = 0;

static void record_result(const char *name, uint64_t raw,
                           const pmu_read_buf_t *buf) {
    if (g_n_results >= MAX_RESULTS) return;
    g_results[g_n_results].name = name;
    g_results[g_n_results].raw = raw;
    g_results[g_n_results].scaled = pmu_scale(raw, buf);
    g_results[g_n_results].multiplexed = pmu_was_multiplexed(buf);
    if (g_results[g_n_results].multiplexed) g_any_multiplexed = 1;
    g_n_results++;
}

static double lookup(const char *name) {
    for (int i = 0; i < g_n_results; i++) {
        if (strcmp(g_results[i].name, name) == 0) return g_results[i].scaled;
    }
    return -1.0;
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

static struct node *build_cycle(size_t n, size_t stride_bytes, uint32_t seed) {
    size_t node_span = stride_bytes > sizeof(struct node) ? stride_bytes
                                                           : sizeof(struct node);
    unsigned char *pool = malloc(n * node_span + node_span);
    if (!pool) { perror("malloc"); exit(1); }

    size_t *order = malloc(n * sizeof(*order));
    if (!order) { perror("malloc order"); exit(1); }
    for (size_t i = 0; i < n; i++) order[i] = i;

    uint32_t state = seed ? seed : 1u;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(xorshift32(&state) % (uint32_t)(i + 1));
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }

    struct node **slot = malloc(n * sizeof(*slot));
    if (!slot) { perror("malloc slot"); exit(1); }
    for (size_t i = 0; i < n; i++)
        slot[i] = (struct node *)(pool + i * node_span);

    for (size_t i = 0; i < n; i++)
        slot[order[i]]->next = slot[order[(i + 1) % n]];

    free(order);
    struct node *head = slot[0];
    free(slot);
    return head;
}

static void chase(struct node *p, uint64_t steps) {
    for (uint64_t i = 0; i < steps; i++) {
        p = p->next;
    }
    g_sink = p;
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Run one small group against a fresh re-warmed pass of the SAME cycle,
 * and record whatever opened successfully by name. `events` is an array
 * of {name, type, config} triples; `n` is how many to attempt (<=4 is a
 * safe default group size). Returns the number that actually opened. */
typedef struct {
    const char *name;
    uint32_t type;
    uint64_t config;
} event_spec_t;

static int run_group(struct node *cycle_head, size_t n_nodes, uint64_t steps,
                      int cpu, const event_spec_t *events, int n) {
    pmu_group_t g;
    memset(&g, 0, sizeof(g));
    for (int i = 0; i < n; i++) {
        pmu_group_add(&g, events[i].name, events[i].type, events[i].config, cpu);
    }
    if (g.n_events == 0) {
        pmu_group_close(&g);
        return 0;
    }

    /* Re-warm before every group so each group sees an equally warm chain,
     * not a chain left in whatever state the previous group's run left it in. */
    chase(cycle_head, n_nodes * 4);

    pmu_group_reset_enable(&g);
    chase(cycle_head, steps);
    pmu_group_disable(&g);

    pmu_read_buf_t buf;
    memset(&buf, 0, sizeof(buf));
    if (pmu_group_read(&g, &buf) != 0) {
        pmu_group_close(&g);
        return 0;
    }

    for (int i = 0; i < g.n_events; i++) {
        record_result(g.names[i], buf.value[i], &buf);
    }
    pmu_group_close(&g);
    return g.n_events;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage:\n"
                "  %s capacity <footprint_bytes> <steps>\n"
                "  %s stride   <stride_bytes> <footprint_bytes> [steps]\n"
                "  %s assoc    <n_conflicting_lines> <candidate_set_stride_bytes> "
                "<line_size_bytes> [steps]\n",
                argv[0], argv[0], argv[0]);
        return 2;
    }

    const char *mode = argv[1];
    long p1 = 0, p2 = 0, p3 = 0;
    uint64_t steps = DEFAULT_N_SAMPLES;
    size_t n_nodes = 0;
    struct node *cycle_head = NULL;

    if (strcmp(mode, "capacity") == 0) {
        if (argc < 3) { fprintf(stderr, "capacity needs footprint_bytes [steps]\n"); return 2; }
        p1 = atol(argv[2]);                         /* footprint bytes */
        steps = (argc > 3) ? (uint64_t)atoll(argv[3]) : DEFAULT_N_SAMPLES;
        n_nodes = (size_t)p1 / sizeof(struct node);
        if (n_nodes < 2) n_nodes = 2;
        cycle_head = build_cycle(n_nodes, sizeof(struct node), 12345u);
    } else if (strcmp(mode, "stride") == 0) {
        if (argc < 4) { fprintf(stderr, "stride needs stride_bytes footprint_bytes [steps]\n"); return 2; }
        p1 = atol(argv[2]);               /* stride bytes */
        p2 = atol(argv[3]);               /* footprint bytes */
        steps = (argc > 4) ? (uint64_t)atoll(argv[4]) : DEFAULT_N_SAMPLES;
        n_nodes = (size_t)p2 / (size_t)p1;
        if (n_nodes < 2) n_nodes = 2;
        cycle_head = build_cycle(n_nodes, (size_t)p1, 6789u);
    } else if (strcmp(mode, "assoc") == 0) {
        if (argc < 5) { fprintf(stderr, "assoc needs n_conflicting_lines set_stride_bytes line_size_bytes [steps]\n"); return 2; }
        p1 = atol(argv[2]);               /* number of conflicting lines */
        p2 = atol(argv[3]);               /* candidate set-index stride */
        p3 = atol(argv[4]);               /* line size (from frozen Phase-I result) */
        steps = (argc > 5) ? (uint64_t)atoll(argv[5]) : DEFAULT_N_SAMPLES;
        n_nodes = (size_t)p1;
        if (n_nodes < 2) n_nodes = 2;
        cycle_head = build_cycle(n_nodes, (size_t)p2, 42u);
    } else {
        fprintf(stderr, "unknown mode '%s'\n", mode);
        return 2;
    }

    int cpu = sched_getcpu();
    double t0 = now_ns();

    /* Not `static const`: PMU_CFG_* expand to calls to an inline function
     * (pmu_hwcache_config), not compile-time constants, so these must be
     * ordinary local arrays initialized at runtime. They're tiny and built
     * once per invocation, so this costs nothing measurable. */
    /* Group A: portable, cheap, almost always available. */
    const event_spec_t group_a[] = {
        {"cycles",       PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
    };
    /* Group B: overall cache references/misses. */
    const event_spec_t group_b[] = {
        {"cache_refs",   PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
        {"cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
    };
    /* Group C: L1D specific. */
    const event_spec_t group_c[] = {
        {"l1d_loads",  PERF_TYPE_HW_CACHE, PMU_CFG_L1D_LOADS},
        {"l1d_misses", PERF_TYPE_HW_CACHE, PMU_CFG_L1D_LOAD_MISSES},
    };
    /* Group D: last-level-cache specific. */
    const event_spec_t group_d[] = {
        {"ll_loads",  PERF_TYPE_HW_CACHE, PMU_CFG_LL_LOADS},
        {"ll_misses", PERF_TYPE_HW_CACHE, PMU_CFG_LL_LOAD_MISSES},
    };
    /* Group E: raw, machine-specific L2 pair -- only attempted if both
     * are configured to something nonzero above. */
    const event_spec_t group_e[] = {
        {RAW_L2_NAME_1, PERF_TYPE_RAW, RAW_L2_EVENT_1},
        {RAW_L2_NAME_2, PERF_TYPE_RAW, RAW_L2_EVENT_2},
    };

    run_group(cycle_head, n_nodes, steps, cpu, group_a, 2);
    run_group(cycle_head, n_nodes, steps, cpu, group_b, 2);
    run_group(cycle_head, n_nodes, steps, cpu, group_c, 2);
    run_group(cycle_head, n_nodes, steps, cpu, group_d, 2);
    if (RAW_L2_EVENT_1 != 0 && RAW_L2_EVENT_2 != 0) {
        run_group(cycle_head, n_nodes, steps, cpu, group_e, 2);
    }

    double t1 = now_ns();

    if (g_n_results == 0) {
        fprintf(stderr, "no PMU events could be opened on this machine -- "
                        "check perf_event_paranoid, whether this host is "
                        "virtualized, and try a bare `perf stat -e cycles "
                        "sleep 1` before assuming this program is at fault\n");
        return 1;
    }

    double cycles       = lookup("cycles");
    double instructions = lookup("instructions");
    double cache_refs   = lookup("cache_refs");
    double cache_misses = lookup("cache_misses");
    double l1d_loads    = lookup("l1d_loads");
    double l1d_misses   = lookup("l1d_misses");
    double ll_loads     = lookup("ll_loads");
    double ll_misses    = lookup("ll_misses");
    double raw_l2_1     = lookup(RAW_L2_NAME_1);
    double raw_l2_2     = lookup(RAW_L2_NAME_2);

    double cache_miss_rate = (cache_refs > 0 && cache_misses >= 0)
        ? cache_misses / cache_refs : -1.0;
    double l1d_miss_rate = (l1d_loads > 0 && l1d_misses >= 0)
        ? l1d_misses / l1d_loads : -1.0;
    double ll_miss_rate = (ll_loads > 0 && ll_misses >= 0)
        ? ll_misses / ll_loads : -1.0;

    printf("mode,param1,param2,param3,n_samples,elapsed_ns,"
           "cycles,instructions,cache_refs,cache_misses,"
           "l1d_loads,l1d_misses,ll_loads,ll_misses,raw_l2_1,raw_l2_2,"
           "any_group_multiplexed,cache_miss_rate,l1d_miss_rate,ll_miss_rate\n");
    printf("%s,%ld,%ld,%ld,%" PRIu64 ",%.1f,"
           "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,"
           "%d,%.6f,%.6f,%.6f\n",
           mode, p1, p2, p3, steps, t1 - t0,
           cycles, instructions, cache_refs, cache_misses,
           l1d_loads, l1d_misses, ll_loads, ll_misses, raw_l2_1, raw_l2_2,
           g_any_multiplexed, cache_miss_rate, l1d_miss_rate, ll_miss_rate);

    return 0;
}
