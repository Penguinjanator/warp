/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * threads.h — minimal fork-join pool for row-parallel kernels.
 *
 * No OpenMP: Apple's clang does not ship it, and a build dependency for
 * one parallel-for is not worth it. Workers are created once and parked on
 * a condvar; waste_parallel_for splits [0,n) across them and blocks until
 * all have finished. Splitting is by row, so results are bit-identical to
 * the serial version regardless of thread count.
 *
 * Windows needs nothing here: MinGW implements pthreads over winpthreads,
 * so the pool below compiles and runs unchanged. Only the CPU count was
 * POSIX-only, and that now comes from platform.h.
 *
 * Placement is the OS's unless a caller names a cpuset. It is worth naming
 * on a machine whose cores are not interchangeable: on a two-CCD Ryzen,
 * six threads on one CCD measured 16-25% faster than the same six split
 * across both, at identical bytes read (issue #23), because the split pair
 * reaches the other die's L3 over Infinity Fabric and the barrier waits
 * for whoever is slowest. LEARNED §47 is the same shape with P-cores and
 * E-cores instead of dies. Neither is a default — §47 measured "cap the
 * pool" as a 25% win on one model and a 34% loss on another — so this is
 * a switch, and the engine still chooses nothing on its own.
 */

#ifndef WASTE_THREADS_H
#define WASTE_THREADS_H

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

typedef void (*waste_range_fn)(int begin, int end, void *arg);

typedef struct {
    pthread_t th[64];
    pthread_mutex_t mu;
    /* Two start signals, not one. A fast job must not wake the efficiency
     * cores at all: they would take no chunk, but they would still have to
     * be scheduled before they could say so, and with the barrier waiting
     * on a count they decrement, their wake-up latency lands on the
     * critical path of every dispatch. The first version of this had one
     * condvar and measured exactly nothing, for that reason. */
    pthread_cond_t start, start_slow, done;
    waste_range_fn fn;
    void *arg;
    int n, nthreads, next_chunk, chunk, active, epoch, stop;
    /* How many of the pool's threads are meant for the fast cores, and
     * how many a given job wants. `n_fast` counts the calling thread,
     * which is a worker too; `job_workers` is what the current job set,
     * and a worker whose index is past it wakes, takes nothing and goes
     * back to sleep. See waste_parallel_for_fast. */
    int n_fast, job_workers;
    /* The cpuset every participant binds to, and a generation so a thread
     * knows whether it has already done so. 0 = placement is the OS's. */
    waste_cpumask cpus;
    int aff_gen;
} waste_pool;

static waste_pool g_pool;
/* The pool is shared by every model in this translation unit.  One lock
 * makes first-use initialization deterministic; the other owns a complete
 * submitted job, because the fields in waste_pool are the job descriptor
 * itself and cannot represent two callers at once. */
static pthread_mutex_t g_pool_init_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_pool_run_mu = PTHREAD_MUTEX_INITIALIZER;

/* Outside the pool because waste_pool_shutdown memsets the pool and the
 * per-thread "already bound" marks survive it: a second pool that reused
 * generation 1 would be skipped by every thread the first one bound, and
 * silently keep the old cpuset. Counting up costs nothing and cannot. */
static int g_aff_gen;

/* Bind this thread to the pool's cpuset, at most once per thread.
 *
 * The pool's own threads do it when they start. The calling thread does it
 * on its first parallel region rather than at pool init, because it is a
 * worker too (see the tail of waste_parallel_for) and because it is the
 * host's thread, not the engine's: restricting it when it is handed to a
 * kernel is defensible, restricting it because a model was opened on it is
 * not. It is also the only way this works in a server, where the thread
 * that opens the model is rarely the thread that decodes on it.
 *
 * Leaving the caller out was measured as pointless: with six threads it is
 * one of the six, so an unbound caller on the far die is exactly the
 * straggler the option exists to remove. */
static inline void waste__bind_self(void)
{
    static _Thread_local int seen;      /* per thread, per translation unit */
    const int gen = g_pool.aff_gen;
    if (!gen || seen == gen) return;
    seen = gen;
    waste_bind_thread_cpus(&g_pool.cpus);
}

/* Each worker knows its own index so a job can decline the slow ones. */
typedef struct { waste_pool *P; int idx; } waste_worker;
static waste_worker g_workers[64];

static void *waste__worker(void *p)
{
    waste_worker *W = (waste_worker *)p;
    waste_pool *P = W->P;
    int seen = 0;
    waste__bind_self();
    /* Which signal this worker sleeps on for the rest of its life. */
    const int slow = P->n_fast && W->idx >= P->n_fast - 1;
    pthread_cond_t *sig = slow ? &P->start_slow : &P->start;
    for (;;) {
        pthread_mutex_lock(&P->mu);
        while (!P->stop && P->epoch == seen) pthread_cond_wait(sig, &P->mu);
        if (P->stop) { pthread_mutex_unlock(&P->mu); return NULL; }
        seen = P->epoch;
        const int mine = W->idx < P->job_workers - 1;
        if (!mine) { pthread_mutex_unlock(&P->mu); continue; }
        pthread_mutex_unlock(&P->mu);

        for (;;) {
            pthread_mutex_lock(&P->mu);
            const int b = P->next_chunk;
            if (b >= P->n) { pthread_mutex_unlock(&P->mu); break; }
            P->next_chunk = b + P->chunk;
            pthread_mutex_unlock(&P->mu);
            int e = b + P->chunk;
            if (e > P->n) e = P->n;
            P->fn(b, e, P->arg);
        }

        pthread_mutex_lock(&P->mu);
        if (--P->active == 0) pthread_cond_signal(&P->done);
        pthread_mutex_unlock(&P->mu);
    }
}

static inline int waste_hw_threads(void) { return waste_cpu_count(); }

/* How many the pool ended up with, which is not always what was asked for:
 * a cpuset sizes it, the cap trims it, and a pthread_create that failed
 * leaves it short. 0 before the first waste_pool_init. */
static inline int waste_pool_threads(void) { return g_pool.nthreads; }

/* nthreads 0 = one per CPU the pool may use; cpus NULL = wherever the OS
 * puts them. Both are decided by the first caller in the process: the pool
 * is shared, and a second model opened with different wishes gets the
 * first model's pool. */
static inline void waste_pool_init(int nthreads, const waste_cpumask *cpus)
{
    pthread_mutex_lock(&g_pool_init_mu);
    if (g_pool.nthreads) { pthread_mutex_unlock(&g_pool_init_mu); return; }
    if (cpus) {
        g_pool.cpus = *cpus;
        g_pool.aff_gen = ++g_aff_gen;
        /* A cpuset without a thread count means the cpuset: 24 threads
         * over the 6 CPUs a user just asked for is not what they asked
         * for, and it is worse than either half of the choice. */
        if (nthreads <= 0) nthreads = waste_cpumask_count(cpus);
    }
    if (nthreads <= 0) nthreads = waste_hw_threads();
    if (nthreads > 64) nthreads = 64;
    if (nthreads < 1) nthreads = 1;
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.start, NULL);
    pthread_cond_init(&g_pool.start_slow, NULL);
    pthread_cond_init(&g_pool.done, NULL);

    /* The fast group. It is the first `n_fast` participants — the calling
     * thread plus workers 0..n_fast-2 — and on macOS they are created at
     * a quality of service the scheduler answers with a performance core.
     * There is no affinity API on this platform, so the class is the whole
     * mechanism: QOS_CLASS_USER_INTERACTIVE prefers the P cluster, and
     * the E-core threads simply do not participate in a fast job, which is
     * what leaves the cluster to the ones that do.
     *
     * A cpuset overrides it: a caller who has named CPUs has said what it
     * wants and the two mechanisms would fight. */
    {
        const char *e = getenv("WASTE_PCORES");
        int f = e ? atoi(e) : waste_perf_cpu_count();
        if (cpus || f <= 0 || f >= nthreads) f = 0;   /* nothing to split */
        g_pool.n_fast = f;
    }

    /* Count only workers that actually exist.  Waiting for a failed
     * pthread_create as though it were alive otherwise hangs forever. */
    g_pool.nthreads = 1;
    g_pool.job_workers = nthreads;
    for (int i = 0; i < nthreads - 1; i++) {
        pthread_attr_t attr, *ap = NULL;
#if defined(__APPLE__)
        if (g_pool.n_fast && pthread_attr_init(&attr) == 0) {
            pthread_attr_set_qos_class_np(&attr,
                i < g_pool.n_fast - 1 ? QOS_CLASS_USER_INTERACTIVE
                                      : QOS_CLASS_UTILITY, 0);
            ap = &attr;
        }
#else
        (void)attr;
#endif
        g_workers[i].P = &g_pool;
        g_workers[i].idx = i;
        const int rc = pthread_create(&g_pool.th[i], ap, waste__worker,
                                      &g_workers[i]);
#if defined(__APPLE__)
        if (ap) pthread_attr_destroy(ap);
#endif
        if (rc) break;
        g_pool.nthreads++;
    }
    if (g_pool.n_fast > g_pool.nthreads) g_pool.n_fast = g_pool.nthreads;
    pthread_mutex_unlock(&g_pool_init_mu);
}

/* How many participants a fast job gets, or the whole pool where the
 * machine has no split to make. */
static inline int waste_pool_fast(void)
{ return g_pool.n_fast ? g_pool.n_fast : g_pool.nthreads; }

/* Runs fn over [0,n) split into chunks; every chunk boundary is a multiple
 * of min_chunk. Blocks until complete.
 *
 * `workers` is how many participants to cut the work for. Everything the
 * engine does went through the whole pool until LEARNED §47, which found
 * that the barrier makes an efficiency core a straggler for a kernel wide
 * enough to be limited by its slowest lane — and that the same machine
 * wants every core for the kernel next to it. So the count is per call
 * site now, and the two available answers are the pool and the fast
 * group. Results do not depend on it: the split is by row either way. */
static inline void waste_parallel_for_n(int n, int min_chunk, waste_range_fn fn,
                                        void *arg, int workers)
{
    /* Before the serial path too: a pool of one still runs the kernel, and
     * on the CPUs it was told to. A thread-local check, so this costs a
     * load per dispatch and a syscall once per thread. */
    waste__bind_self();
    if (workers > g_pool.nthreads) workers = g_pool.nthreads;
    if (workers < 1) workers = 1;
    if (workers <= 1 || n <= min_chunk) { fn(0, n, arg); return; }
    int chunk = (n + workers - 1) / workers;
    if (chunk < min_chunk) chunk = min_chunk;
    /* Round up to a whole number of min_chunk units: callers that block
     * their data (the VQ tile) need every range to start on a boundary. */
    chunk = ((chunk + min_chunk - 1) / min_chunk) * min_chunk;

    /* Keep the descriptor stable until every worker has left this job.
     * Distinct waste_ctx instances may be called concurrently even though
     * they reuse this process-wide pool. */
    pthread_mutex_lock(&g_pool_run_mu);
    pthread_mutex_lock(&g_pool.mu);
    g_pool.fn = fn; g_pool.arg = arg; g_pool.n = n; g_pool.chunk = chunk;
    g_pool.next_chunk = 0;
    g_pool.job_workers = workers;
    g_pool.active = workers - 1;
    g_pool.epoch++;
    pthread_cond_broadcast(&g_pool.start);
    if (workers > g_pool.n_fast) pthread_cond_broadcast(&g_pool.start_slow);
    pthread_mutex_unlock(&g_pool.mu);

    /* the calling thread is a worker too */
    for (;;) {
        pthread_mutex_lock(&g_pool.mu);
        const int b = g_pool.next_chunk;
        if (b >= n) { pthread_mutex_unlock(&g_pool.mu); break; }
        g_pool.next_chunk = b + chunk;
        pthread_mutex_unlock(&g_pool.mu);
        int e = b + chunk;
        if (e > n) e = n;
        fn(b, e, arg);
    }

    pthread_mutex_lock(&g_pool.mu);
    while (g_pool.active > 0) pthread_cond_wait(&g_pool.done, &g_pool.mu);
    pthread_mutex_unlock(&g_pool.mu);
    pthread_mutex_unlock(&g_pool_run_mu);
}

static inline void waste_parallel_for(int n, int min_chunk, waste_range_fn fn,
                                      void *arg)
{ waste_parallel_for_n(n, min_chunk, fn, arg, g_pool.nthreads); }

/* For a kernel whose lanes are wide and whose barrier therefore waits for
 * the slowest core in it. Falls back to the whole pool on a machine with
 * one kind of core, so a call site can use it unconditionally. */
static inline void waste_parallel_for_fast(int n, int min_chunk,
                                           waste_range_fn fn, void *arg)
{ waste_parallel_for_n(n, min_chunk, fn, arg, waste_pool_fast()); }

static inline void waste_pool_shutdown(void)
{
    if (!g_pool.nthreads) return;
    pthread_mutex_lock(&g_pool.mu);
    g_pool.job_workers = g_pool.nthreads;
    g_pool.stop = 1;
    /* Both signals: a slow worker never hears the fast one, and a shutdown
     * that only wakes half the pool joins forever. */
    pthread_cond_broadcast(&g_pool.start);
    pthread_cond_broadcast(&g_pool.start_slow);
    pthread_mutex_unlock(&g_pool.mu);
    for (int i = 0; i < g_pool.nthreads - 1; i++) pthread_join(g_pool.th[i], NULL);
    memset(&g_pool, 0, sizeof g_pool);
}

#endif /* WASTE_THREADS_H */
