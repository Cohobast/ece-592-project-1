/*
 * pmu_counters.h
 *
 * Minimal, dependency-free wrapper around perf_event_open(2) for grouping
 * and reading hardware performance counters from a Phase-I-style benchmark.
 *
 * Phase II only: do not call anything in this header until your Phase-I
 * timing-only results are frozen and committed.
 *
 * Design notes:
 *  - Counters are opened as a group (the first fd is the group leader).
 *    Grouping is what guarantees the events are multiplexed together (or,
 *    ideally, not multiplexed at all if the PMU has enough physical
 *    counters) so that ratios computed across events are meaningful.
 *  - We request PERF_FORMAT_TOTAL_TIME_ENABLED / _RUNNING so you can detect
 *    multiplexing after the fact (time_running < time_enabled means the
 *    group did not run continuously and the raw counts must be scaled or
 *    treated with caution, per the homework's caveat about not letting
 *    multiplexing obscure a cache transition).
 *  - PERF_TYPE_HARDWARE generic events (cycles, instructions,
 *    cache-references, cache-misses) are portable across x86 and Arm.
 *    Level-specific events (L1D loads/misses, LLC loads/misses) use
 *    PERF_TYPE_HW_CACHE and are documented separately in
 *    pmu_events_hwcache.h because their config encoding is a 3-field
 *    bitfield (cache id, op id, result id) rather than a flat enum.
 */

#ifndef PMU_COUNTERS_H
#define PMU_COUNTERS_H

#include <errno.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define PMU_MAX_EVENTS 8

typedef struct {
    int fd[PMU_MAX_EVENTS];
    int n_events;
    const char *names[PMU_MAX_EVENTS];
} pmu_group_t;

typedef struct {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t value[PMU_MAX_EVENTS];
} pmu_read_buf_t;

static inline long pmu_perf_event_open(struct perf_event_attr *attr, pid_t pid,
                                        int cpu, int group_fd,
                                        unsigned long flags) {
    return syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/*
 * Open one event and add it to the group. `type`/`config` follow the
 * perf_event_open semantics (e.g. PERF_TYPE_HARDWARE / PERF_COUNT_HW_CYCLES,
 * or PERF_TYPE_HW_CACHE / an encoded cache config -- see
 * pmu_hwcache_config() below). `pinned_cpu` should match the logical CPU
 * you taskset the benchmark to; pass -1 to let it follow the calling
 * thread (not recommended -- pin explicitly, consistent with the rest of
 * the homework's affinity requirements).
 *
 * Returns 0 on success, -1 on failure (errno set, message printed to
 * stderr identifying which named event failed to open -- this is the
 * detail you should record rather than silently substituting a different
 * event, per the assignment's Phase-II instructions).
 */
static inline int pmu_group_add(pmu_group_t *g, const char *name,
                                 uint32_t type, uint64_t config,
                                 int pinned_cpu) {
    if (g->n_events >= PMU_MAX_EVENTS) {
        fprintf(stderr, "pmu_group_add: group full, cannot add %s\n", name);
        return -1;
    }

    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                      PERF_FORMAT_TOTAL_TIME_RUNNING;

    int is_leader = (g->n_events == 0);
    pe.disabled = is_leader ? 1 : 0;
    int group_fd = is_leader ? -1 : g->fd[0];

    int fd = (int)pmu_perf_event_open(&pe, 0 /* this process */, pinned_cpu,
                                       group_fd, 0);
    if (fd < 0) {
        fprintf(stderr, "pmu_group_add: failed to open '%s' (type=%u "
                        "config=0x%" PRIx64 "): %s\n",
                name, type, config, strerror(errno));
        return -1;
    }

    g->fd[g->n_events] = fd;
    g->names[g->n_events] = name;
    g->n_events++;
    return 0;
}

static inline void pmu_group_reset_enable(pmu_group_t *g) {
    ioctl(g->fd[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(g->fd[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

static inline void pmu_group_disable(pmu_group_t *g) {
    ioctl(g->fd[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
}

/*
 * Read the group. Returns 0 on success. buf->nr should equal g->n_events;
 * mismatch is treated as an error (do not silently trust a short read).
 * Caller should check time_running vs time_enabled to detect multiplexing.
 */
static inline int pmu_group_read(pmu_group_t *g, pmu_read_buf_t *buf) {
    size_t want = sizeof(uint64_t) * 3 +
                  sizeof(uint64_t) * (size_t)g->n_events;
    ssize_t got = read(g->fd[0], buf, want);
    if (got != (ssize_t)want) {
        fprintf(stderr, "pmu_group_read: short/failed read (%zd of %zu): %s\n",
                got, want, strerror(errno));
        return -1;
    }
    if (buf->nr != (uint64_t)g->n_events) {
        fprintf(stderr, "pmu_group_read: expected %d events, kernel reports %"
                        PRIu64 "\n", g->n_events, buf->nr);
        return -1;
    }
    return 0;
}

static inline void pmu_group_close(pmu_group_t *g) {
    for (int i = 0; i < g->n_events; i++) {
        if (g->fd[i] >= 0) close(g->fd[i]);
    }
}

/* Convenience: was this read affected by multiplexing? */
static inline int pmu_was_multiplexed(const pmu_read_buf_t *buf) {
    return buf->time_running < buf->time_enabled;
}

/*
 * Scale a raw count for multiplexing, the same way `perf stat` does:
 * scaled = raw * (time_enabled / time_running). If time_running == 0,
 * the event never ran; report that explicitly instead of dividing by zero.
 */
static inline double pmu_scale(uint64_t raw, const pmu_read_buf_t *buf) {
    if (buf->time_running == 0) return 0.0;
    return (double)raw * ((double)buf->time_enabled / (double)buf->time_running);
}

#endif /* PMU_COUNTERS_H */
