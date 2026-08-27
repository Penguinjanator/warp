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
#include <stdatomic.h>
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
    /* epoch, next_chunk and active are read and written outside the mutex.
     *
     * A parked thread costs 54 us to wake on this machine, and a decode
     * step dispatches ~150 times — so the mutex was never the problem, the
     * *parking* was. Workers now spin on `epoch` for a bounded time before
     * they park, the caller publishes it with a release store, and the
     * chunk cursor and the completion count are atomics so a spinning
     * worker never has to take a lock at all. The condvars stay for
     * whoever did park: the epoch store happens before the broadcast and
     * the parked side re-checks under the mutex, so no wake-up is lost. */
    /* epoch and job_workers travel together in one word.
     *
     * They cannot be separate atomics. A worker that is not in the current
     * job reads job_workers, decides it is not wanted, and goes back to
     * spinning — and the dispatcher never waits for it, so that read races
     * the *next* job's write. Reading a newer job_workers against an older
     * epoch made a non-participant join a job it had not been counted in,
     * drive `active` to zero early, and let the caller return while its
     * arg was still being read. One acquire load of a packed pair cannot
     * see a mismatched half. */
    _Atomic uint64_t job;            /* epoch << 32 | workers             */
    _Atomic int next_chunk, active;
    _Atomic int stop;
    int n, nthreads, chunk;
    /* How many of the pool's threads are meant for the fast cores, and
     * how many a given job wants. `n_fast` counts the calling thread,
     * which is a worker too; `job_workers` is what the current job set,
     * and a worker whose index is past it wakes, takes nothing and goes
     * back to sleep. See waste_parallel_for_fast. */
    int n_fast;
    /* Resolved once, at init, from the environment: a lazy static inside a
     * helper is written by whichever worker gets there first and read by
     * all the others, which is a race even when every writer stores the
     * same value. */
    long spin, spin_slow;
    size_t wide_min;
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

/* How long a worker spins before parking, in iterations of the relax loop.
 *
 * Measured rather than guessed: a parked thread costs 54 us to wake on this
 * machine and a decode step dispatches about 150 times, so parking was 12%
 * of a token. Spinning removes that at the cost of burning a core while
 * idle, which is why it is bounded and why WASTE_SPIN can turn it off — on
 * a machine that is doing something else, or on battery, the trade is not
 * obviously the same one. 0 restores the pure condvar pool. */
static inline long waste__env_spin(void)
{
    const char *e = getenv("WASTE_SPIN");
    long v = e ? atol(e) : 20000;
    return v < 0 ? 0 : v;
}

static inline long waste__env_spin_slow(void)
{
    const char *e = getenv("WASTE_SPIN_SLOW");
    return (e && *e != '0') ? waste__env_spin() : 0;
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
    /* A slow worker parks straight away: the whole point of the fast/slow
     * split is that a short job must not pay for the efficiency cores, and
     * a spinning E-core would put itself back on exactly the critical path
     * the split removes. */
    const long spin = slow ? P->spin_slow : P->spin;
    uint64_t job = 0;
    for (;;) {
        for (long i = 0; i < spin; i++) {
            job = atomic_load_explicit(&P->job, memory_order_acquire);
            if ((uint32_t)(job >> 32) != (uint32_t)seen ||
                atomic_load_explicit(&P->stop, memory_order_relaxed))
                goto awake;
            waste_cpu_relax();
        }
        pthread_mutex_lock(&P->mu);
        for (;;) {
            job = atomic_load_explicit(&P->job, memory_order_acquire);
            if (atomic_load_explicit(&P->stop, memory_order_relaxed) ||
                (uint32_t)(job >> 32) != (uint32_t)seen) break;
            pthread_cond_wait(sig, &P->mu);
        }
        pthread_mutex_unlock(&P->mu);
    awake:
        if (atomic_load_explicit(&P->stop, memory_order_relaxed)) return NULL;
        seen = (int)(uint32_t)(job >> 32);
        if (W->idx >= (int)(uint32_t)job - 1) continue;   /* not in this job */

        for (;;) {
            const int b = atomic_fetch_add_explicit(&P->next_chunk, P->chunk,
                                                    memory_order_relaxed);
            if (b >= P->n) break;
            int e = b + P->chunk;
            if (e > P->n) e = P->n;
            P->fn(b, e, P->arg);
        }

        /* The signal costs a lock, so pay it only when someone is waiting:
         * the caller spins first and only takes the mutex if it has to. */
        if (atomic_fetch_sub_explicit(&P->active, 1,
                                      memory_order_acq_rel) == 1) {
            pthread_mutex_lock(&P->mu);
            pthread_cond_broadcast(&P->done);
            pthread_mutex_unlock(&P->mu);
        }
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
    g_pool.spin = waste__env_spin();
    g_pool.spin_slow = waste__env_spin_slow();
    {
        const char *e = getenv("WASTE_WIDE_MIN");
        g_pool.wide_min = e ? (size_t)strtoull(e, NULL, 0) : (size_t)(4u << 20);
    }
    g_pool.nthreads = 1;
    /* The job word stays 0 until the first dispatch, and a worker starts
     * with seen == 0, so it parks instead of running a job that does not
     * exist. Seeding it with epoch 1 here was a real bug and not a
     * cosmetic one: every worker woke immediately, read n and chunk before
     * anything had written them, and decremented `active` below zero,
     * which the first real dispatch then inherited. */
    for (int i = 0; i < nthreads - 1; i++) {
        pthread_attr_t attr, *ap = NULL;
#if defined(__APPLE__)
        /* Raise the fast group and leave everyone else exactly as they
         * were. Marking the rest QOS_CLASS_UTILITY was tried and is a 25%
         * regression on K3: a full-pool job — the trunk matvec, 46% of a
         * decode step — then runs twelve of its eighteen threads at a
         * class the scheduler answers with an efficiency core, so the
         * kernel that wants every core loses two thirds of them. The fast
         * group only has to be *preferred*, not the others demoted; during
         * a fast job the rest are asleep and compete for nothing anyway. */
        if (g_pool.n_fast && i < g_pool.n_fast - 1 &&
            pthread_attr_init(&attr) == 0) {
            pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
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
    /* Plain writes, published by the release store below: no worker may
     * read them without first acquiring the new job word, and the previous
     * job's participants have all decremented `active` before this call
     * returned, so none of them is still reading. */
    g_pool.fn = fn; g_pool.arg = arg; g_pool.n = n; g_pool.chunk = chunk;
    atomic_store_explicit(&g_pool.next_chunk, 0, memory_order_relaxed);
    atomic_store_explicit(&g_pool.active, workers - 1, memory_order_relaxed);
    {
        const uint64_t prev = atomic_load_explicit(&g_pool.job,
                                                   memory_order_relaxed);
        const uint64_t next = ((prev >> 32) + 1) << 32 | (uint32_t)workers;
        atomic_store_explicit(&g_pool.job, next, memory_order_release);
    }
    /* A parked worker still needs the broadcast, and it has to happen under
     * the mutex: the parked side re-checks the job word while holding it,
     * which is what makes store-then-signal safe rather than racy. */
    pthread_mutex_lock(&g_pool.mu);
    pthread_cond_broadcast(&g_pool.start);
    if (workers > g_pool.n_fast) pthread_cond_broadcast(&g_pool.start_slow);
    pthread_mutex_unlock(&g_pool.mu);

    /* the calling thread is a worker too */
    for (;;) {
        const int b = atomic_fetch_add_explicit(&g_pool.next_chunk, chunk,
                                                memory_order_relaxed);
        if (b >= n) break;
        int e = b + chunk;
        if (e > n) e = n;
        fn(b, e, arg);
    }

    /* Spin on the completion count before waiting on it, for the same
     * reason the workers spin on the epoch: the last worker to finish is
     * usually microseconds behind, and a condvar round trip is longer than
     * the wait it replaces. */
    {
        const long spin = g_pool.spin;
        for (long i = 0; i < spin; i++) {
            if (atomic_load_explicit(&g_pool.active, memory_order_acquire) <= 0)
                goto joined;
            waste_cpu_relax();
        }
        pthread_mutex_lock(&g_pool.mu);
        while (atomic_load_explicit(&g_pool.active, memory_order_acquire) > 0)
            pthread_cond_wait(&g_pool.done, &g_pool.mu);
        pthread_mutex_unlock(&g_pool.mu);
    }
joined:
    pthread_mutex_unlock(&g_pool_run_mu);
}

static inline void waste_parallel_for(int n, int min_chunk, waste_range_fn fn,
                                      void *arg)
{ waste_parallel_for_n(n, min_chunk, fn, arg, g_pool.nthreads); }

/* Bytes below which a dispatch is not worth waking the slow group for.
 *
 * The two groups do not cost the same to start. The fast group spins, so a
 * dispatch to it is ~1.5 us; the slow group is parked, and waking it
 * measured **54 us** on this machine — for a barrier that then waits for
 * it. A decode step dispatches about 150 times, so that difference is not
 * a detail, it is 12% of a token spent scheduling.
 *
 * Which means the whole pool is only worth reaching for when the job is
 * long enough to hide the wake. `work` is the caller's proxy for how long
 * that is, in bytes touched, and 0 means "no idea" — those keep the whole
 * pool, as they always had. WASTE_WIDE_MIN overrides the threshold; 0
 * sends every sized dispatch to the fast group and a huge value restores
 * the old behaviour exactly. */
static inline void waste_parallel_for_work(int n, int min_chunk,
                                           waste_range_fn fn, void *arg,
                                           size_t work)
{
    if (work && work < g_pool.wide_min)
        waste_parallel_for_n(n, min_chunk, fn, arg, waste_pool_fast());
    else
        waste_parallel_for_n(n, min_chunk, fn, arg, g_pool.nthreads);
}

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
    atomic_store_explicit(&g_pool.stop, 1, memory_order_relaxed);
    /* A spinning worker watches the job word, so move that too or the spin
     * runs to completion before the thread notices. */
    {
        const uint64_t prev = atomic_load_explicit(&g_pool.job,
                                                   memory_order_relaxed);
        atomic_store_explicit(&g_pool.job,
                              ((prev >> 32) + 1) << 32 | (uint32_t)g_pool.nthreads,
                              memory_order_release);
    }
    /* Both signals: a slow worker never hears the fast one, and a shutdown
     * that only wakes half the pool joins forever. */
    pthread_cond_broadcast(&g_pool.start);
    pthread_cond_broadcast(&g_pool.start_slow);
    pthread_mutex_unlock(&g_pool.mu);
    for (int i = 0; i < g_pool.nthreads - 1; i++) pthread_join(g_pool.th[i], NULL);
    memset(&g_pool, 0, sizeof g_pool);
}

#endif /* WASTE_THREADS_H */
