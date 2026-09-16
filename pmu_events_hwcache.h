/*
 * pmu_events_hwcache.h
 *
 * PERF_TYPE_HW_CACHE config values are a 3-byte-packed bitfield:
 *   config = cache_id | (op_id << 8) | (result_id << 16)
 *
 * These are still "generic" Linux events -- the kernel maps them to a
 * vendor-specific raw event underneath -- so availability and exact
 * semantics still differ by CPU. Always cross-check with
 *   perf list | grep -Ei 'cache|L1|LLC|TLB'
 * on the target machine before trusting a value, and record which
 * generic event actually resolved to which raw event if `perf stat -vv`
 * is available.
 *
 * This header only builds the config words; it does not claim every
 * event is supported on every machine. Call pmu_group_add() with the
 * returned config and check its return value -- a failure to open means
 * that event is not exposed on this CPU, which is itself a fact to record
 * per the homework's "do not assume event names are identical across
 * Intel, AMD, and Arm" instruction.
 */

#ifndef PMU_EVENTS_HWCACHE_H
#define PMU_EVENTS_HWCACHE_H

#include <linux/perf_event.h>
#include <stdint.h>

static inline uint64_t pmu_hwcache_config(uint64_t cache_id, uint64_t op_id,
                                           uint64_t result_id) {
    return cache_id | (op_id << 8) | (result_id << 16);
}

/* L1 data cache: loads and load-misses. */
#define PMU_CFG_L1D_LOADS \
    pmu_hwcache_config(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, \
                        PERF_COUNT_HW_CACHE_RESULT_ACCESS)
#define PMU_CFG_L1D_LOAD_MISSES \
    pmu_hwcache_config(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, \
                        PERF_COUNT_HW_CACHE_RESULT_MISS)

/* Last-level cache: loads and load-misses (kernel's LL mapping -- verify
 * against `perf list` that this actually resolves to your LLC and not
 * some other level on the machine under test). */
#define PMU_CFG_LL_LOADS \
    pmu_hwcache_config(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, \
                        PERF_COUNT_HW_CACHE_RESULT_ACCESS)
#define PMU_CFG_LL_LOAD_MISSES \
    pmu_hwcache_config(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, \
                        PERF_COUNT_HW_CACHE_RESULT_MISS)

/* Data TLB. */
#define PMU_CFG_DTLB_LOADS \
    pmu_hwcache_config(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, \
                        PERF_COUNT_HW_CACHE_RESULT_ACCESS)
#define PMU_CFG_DTLB_LOAD_MISSES \
    pmu_hwcache_config(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, \
                        PERF_COUNT_HW_CACHE_RESULT_MISS)

#endif /* PMU_EVENTS_HWCACHE_H */
