/*
 * threads.h — minimal fork-join pool for row-parallel kernels.
 *
 * No OpenMP: Apple's clang does not ship it, and a build dependency for
 * one parallel-for is not worth it. Workers are created once and parked on
 * a condvar; waste_parallel_for splits [0,n) across them and blocks until
 * all have finished. Splitting is by row, so results are bit-identical to
 * the serial version regardless of thread count.
 *
 * Windows: swap the pthreads calls for SRWLOCK + CONDITION_VARIABLE; the
 * interface stays the same.
 */

#ifndef WASTE_THREADS_H
#define WASTE_THREADS_H

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef void (*waste_range_fn)(int begin, int end, void *arg);

typedef struct {
    pthread_t th[64];
    pthread_mutex_t mu;
    pthread_cond_t start, done;
    waste_range_fn fn;
    void *arg;
    int n, nthreads, next_chunk, chunk, active, epoch, stop;
} waste_pool;

static waste_pool g_pool;

static void *waste__worker(void *p)
{
    waste_pool *P = (waste_pool *)p;
    int seen = 0;
    for (;;) {
        pthread_mutex_lock(&P->mu);
        while (!P->stop && P->epoch == seen) pthread_cond_wait(&P->start, &P->mu);
        if (P->stop) { pthread_mutex_unlock(&P->mu); return NULL; }
        seen = P->epoch;
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

static inline int waste_hw_threads(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 1 ? (int)n : 1;
}

static inline void waste_pool_init(int nthreads)
{
    if (g_pool.nthreads) return;
    if (nthreads <= 0) nthreads = waste_hw_threads();
    if (nthreads > 64) nthreads = 64;
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.start, NULL);
    pthread_cond_init(&g_pool.done, NULL);
    g_pool.nthreads = nthreads;
    for (int i = 0; i < nthreads - 1; i++)
        pthread_create(&g_pool.th[i], NULL, waste__worker, &g_pool);
}

/* Runs fn over [0,n) split into chunks; every chunk boundary is a multiple
 * of min_chunk. Blocks until complete. */
static inline void waste_parallel_for(int n, int min_chunk, waste_range_fn fn, void *arg)
{
    if (g_pool.nthreads <= 1 || n <= min_chunk) { fn(0, n, arg); return; }
    int chunk = (n + g_pool.nthreads - 1) / g_pool.nthreads;
    if (chunk < min_chunk) chunk = min_chunk;
    /* Round up to a whole number of min_chunk units: callers that block
     * their data (the VQ tile) need every range to start on a boundary. */
    chunk = ((chunk + min_chunk - 1) / min_chunk) * min_chunk;

    pthread_mutex_lock(&g_pool.mu);
    g_pool.fn = fn; g_pool.arg = arg; g_pool.n = n; g_pool.chunk = chunk;
    g_pool.next_chunk = 0;
    g_pool.active = g_pool.nthreads - 1;
    g_pool.epoch++;
    pthread_cond_broadcast(&g_pool.start);
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
}

static inline void waste_pool_shutdown(void)
{
    if (!g_pool.nthreads) return;
    pthread_mutex_lock(&g_pool.mu);
    g_pool.stop = 1;
    pthread_cond_broadcast(&g_pool.start);
    pthread_mutex_unlock(&g_pool.mu);
    for (int i = 0; i < g_pool.nthreads - 1; i++) pthread_join(g_pool.th[i], NULL);
    memset(&g_pool, 0, sizeof g_pool);
}

#endif /* WASTE_THREADS_H */
