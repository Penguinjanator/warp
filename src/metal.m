/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * metal.m — Metal backend, currently one kernel: the quantized matvec.
 *
 * The engine streams ~17 GB of expert weights per token from disk, so any
 * backend that copies host memory to device memory loses before it starts.
 * Apple Silicon has unified memory, so it does not have to: trunk tensors
 * are allocated page-aligned (waste_dio_alloc) and wrapped with
 * newBufferWithBytesNoCopy, which hands the GPU the same physical pages the
 * CPU is already reading. Nothing is copied, ever, for the weights.
 *
 * The Metal offline compiler ships with Xcode, not the Command Line Tools,
 * so the shader is compiled from source at first use. That costs ~100 ms
 * once and removes a build dependency the rest of the project does not have.
 *
 * Only mvq_rows_f32 is here. That is deliberate: it is the largest
 * dispatched compute item (every trunk projection and the output head), and
 * whether GPU offload can win at all in a per-token latency-bound engine is
 * the question one kernel answers as well as five. See BACKENDS.md for what
 * it answered.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "simd.h"
#include "waste_backend.h"

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "threads.h"

static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct P { int in; int ng; int group; int bits; uint rowbytes; uint m; };

/* One simdgroup per output row, eight rows per threadgroup.
 *
 * The 2026-07-28 version of this kernel did three things that together cost
 * it the argument in docs/BACKENDS.md, and all three are visible in what
 * replaced them:
 *
 *   - it converted a group scale from half to float *per weight*, so a
 *     7168-long row paid 7168 conversions for its 56 scales;
 *   - it reduced through threadgroup memory with a log2(n) barrier ladder,
 *     where simd_sum is one instruction and no barrier at all;
 *   - it read one byte at a time, where a uint4 load moves 32 weights.
 *
 * tools/metalbw.c measures the difference at K3's [12288 x 7168] trunk
 * shape: 53 GB/s then, ~140 GB/s now, against 78 GB/s for the exact NEON
 * path and a 284 GB/s streaming ceiling on this GPU.
 */
kernel void mvq4(device float             *y  [[buffer(0)]],
                 device const uint4       *W  [[buffer(1)]],
                 device const half        *ws [[buffer(2)]],
                 device const float       *x  [[buffer(3)]],
                 constant P               &p  [[buffer(4)]],
                 uint  tgid [[threadgroup_position_in_grid]],
                 uint  sgid [[simdgroup_index_in_threadgroup]],
                 uint  lane [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8 + sgid;
    if (row >= p.m) return;
    device const uint4 *w  = W  + (ulong)row * (p.rowbytes >> 4);
    device const half  *sc = ws + (ulong)row * p.ng;
    const uint nvec = p.rowbytes >> 4;
    const uint gshift = (p.group == 128) ? 7u : uint(log2(float(p.group)) + 0.5f);
    float acc = 0.0f;
    for (uint j = lane; j < nvec; j += 32) {
        const uint4 packed = w[j];
        const uint base = j << 5;
        float part = 0.0f;
        for (uint b = 0; b < 4; b++) {
            const uint word = packed[b];
            for (uint k = 0; k < 4; k++) {
                const uint byte = (word >> (8 * k)) & 0xffu;
                const uint i = base + b * 8 + k * 2;
                part += float(int(byte & 0xfu) - 8) * x[i]
                      + float(int(byte >> 4)  - 8) * x[i + 1];
            }
        }
        acc += float(sc[base >> gshift]) * part;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

/* Q8G: same shape, char4 loads, one scale per group. */
kernel void mvq8(device float             *y  [[buffer(0)]],
                 device const char4       *W  [[buffer(1)]],
                 device const half        *ws [[buffer(2)]],
                 device const float4      *x  [[buffer(3)]],
                 constant P               &p  [[buffer(4)]],
                 uint  tgid [[threadgroup_position_in_grid]],
                 uint  sgid [[simdgroup_index_in_threadgroup]],
                 uint  lane [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8 + sgid;
    if (row >= p.m) return;
    device const char4  *w  = W  + (ulong)row * (p.rowbytes >> 2);
    device const half   *sc = ws + (ulong)row * p.ng;
    const uint nvec = p.rowbytes >> 2;
    const uint gshift = (p.group == 128) ? 7u : uint(log2(float(p.group)) + 0.5f);
    float acc = 0.0f;
    for (uint j = lane; j < nvec; j += 32) {
        const float4 wv = float4(w[j]);
        acc += float(sc[(j << 2) >> gshift]) * dot(wv, x[j]);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

/* Q3G, the path nothing ships but the loader still accepts. Kept scalar. */
kernel void mvq3(device float             *y  [[buffer(0)]],
                 device const uchar       *W  [[buffer(1)]],
                 device const half        *ws [[buffer(2)]],
                 device const float       *x  [[buffer(3)]],
                 constant P               &p  [[buffer(4)]],
                 uint  tgid [[threadgroup_position_in_grid]],
                 uint  sgid [[simdgroup_index_in_threadgroup]],
                 uint  lane [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8 + sgid;
    if (row >= p.m) return;
    device const uchar *W0 = W  + (ulong)row * p.rowbytes;
    device const half  *sc = ws + (ulong)row * p.ng;
    const uint gshift = (p.group == 128) ? 7u : uint(log2(float(p.group)) + 0.5f);
    float acc = 0.0f;
    for (uint i = lane; i < uint(p.in); i += 32) {
        const ulong off = (ulong)i * 3;
        const int b0 = int(W0[off >> 3]), b1 = int(W0[(off >> 3) + 1]);
        const int v = ((b0 >> (off & 7)) | (b1 << (8 - (off & 7)))) & 7;
        acc += float(v - 4) * x[i] * float(sc[i >> gshift]);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}
)MSL";

/* ---- buffer cache -------------------------------------------------------
 * Keyed by host pointer. Trunk tensors live for the process, so a wrapper
 * created once is reused for every token. Linear probing over a fixed table
 * is enough: there are a few thousand tensors and lookups are per matvec,
 * not per row. */

#define MT_SLOTS 8192

typedef struct { const void *key; id<MTLBuffer> buf; } mt_slot;

static struct {
    int ready;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> mvq4, mvq8, mvq3;
    mt_slot slot[MT_SLOTS];
    id<MTLBuffer> scratch_x, scratch_y;
    size_t scratch_x_cap, scratch_y_cap;
    size_t page;
} g;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

/* WASTE_METAL_STATS=1: where the time in this file actually goes. Kept
 * because "the GPU dispatch is slow" and "the call never reached the GPU"
 * look identical from the outside. */
double waste_metal_t_gpu, waste_metal_t_copy, waste_metal_t_wrap;
unsigned long long waste_metal_n_gpu, waste_metal_n_fallback, waste_metal_bytes;
static double mt_now(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec/1e9; }

static size_t mt_hash(const void *p)
{
    uint64_t x = (uint64_t)(uintptr_t)p;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return (size_t)(x & (MT_SLOTS - 1));
}

/* A no-copy wrapper over host memory, or nil when the pointer is not page
 * aligned — in which case the caller falls back to the CPU rather than
 * silently paying for a copy of the trunk. */
static id<MTLBuffer> mt_wrap(const void *p, size_t len)
{
    if (!p || !len) return nil;
    if ((uintptr_t)p % g.page) return nil;
    size_t i = mt_hash(p);
    for (int probe = 0; probe < MT_SLOTS; probe++) {
        if (g.slot[i].key == p) return g.slot[i].buf;
        if (!g.slot[i].key) break;
        i = (i + 1) & (MT_SLOTS - 1);
    }
    const size_t pad = (len + g.page - 1) / g.page * g.page;
    id<MTLBuffer> b = [g.dev newBufferWithBytesNoCopy:(void *)p
                                               length:pad
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    if (!b) return nil;
    g.slot[i].key = p;
    g.slot[i].buf = b;
    return b;
}

/* __strong: ARC otherwise treats an id* parameter as autoreleasing and
 * refuses the address of a strong global. */
static id<MTLBuffer> mt_scratch(__strong id<MTLBuffer> *slot, size_t *cap,
                                size_t need)
{
    if (*slot && *cap >= need) return *slot;
    const size_t pad = (need + g.page - 1) / g.page * g.page;
    *slot = [g.dev newBufferWithLength:pad options:MTLResourceStorageModeShared];
    *cap = *slot ? pad : 0;
    return *slot;
}

/* ---- the kernel --------------------------------------------------------- */

struct MtParams { int in; int ng; int group; int bits; unsigned rowbytes; unsigned m; };

extern void waste_mvq_rows_f32(int b, int e, void *p);   /* CPU fallback */

static void mvq_rows_f32_metal(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    id<MTLComputePipelineState> ps = a->bits == 4 ? g.mvq4
                                   : a->bits == 8 ? g.mvq8 : g.mvq3;
    /* uint4 / char4 loads want a row that is a whole number of vectors,
     * and the caller's `in` must cover the row exactly. Anything else is
     * the CPU's — cheaper than a special case in the shader for a shape no
     * container this project converts produces. */
    const int lanes = a->bits == 4 ? 32 : a->bits == 8 ? 4 : 1;
    if (b != 0 || (a->bits != 3 && (a->rowbytes % (lanes / (a->bits == 4 ? 2 : 1)))) ||
        (a->in % lanes)) {
        waste_metal_n_fallback++;
        waste_parallel_for(e - b, 64, waste_k.mvq_rows_cpu, p);
        return;
    }
    const double t_enter = mt_now();
    pthread_mutex_lock(&g_mu);
    @autoreleasepool {
        id<MTLBuffer> bw = mt_wrap(a->W, (size_t)e * a->rowbytes);
        id<MTLBuffer> bs = mt_wrap(a->ws, (size_t)e * a->ng * sizeof(uint16_t));
        id<MTLBuffer> bx = mt_scratch(&g.scratch_x, &g.scratch_x_cap,
                                      (size_t)a->in * sizeof(float));
        id<MTLBuffer> by = mt_scratch(&g.scratch_y, &g.scratch_y_cap,
                                      (size_t)e * sizeof(float));
        if (!bw || !bs || !bx || !by) {
            waste_metal_n_fallback++;
            pthread_mutex_unlock(&g_mu);
            waste_parallel_for(e, 64, waste_k.mvq_rows_cpu, p);
            return;
        }
        waste_metal_t_wrap += mt_now() - t_enter;
        const double t_c0 = mt_now();
        memcpy([bx contents], (const float *)a->xs, (size_t)a->in * sizeof(float));
        waste_metal_t_copy += mt_now() - t_c0;

        struct MtParams pr = { a->in, a->ng, a->group, a->bits,
                               (unsigned)a->rowbytes, (unsigned)e };
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:by offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:bs offset:0 atIndex:2];
        [enc setBuffer:bx offset:0 atIndex:3];
        [enc setBytes:&pr length:sizeof pr atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)e + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        const double t_g0 = mt_now();
        [cb commit];
        [cb waitUntilCompleted];
        waste_metal_t_gpu += mt_now() - t_g0;
        waste_metal_n_gpu++;
        waste_metal_bytes += (unsigned long long)e * a->rowbytes;
        const double t_c1 = mt_now();
        memcpy(a->y, (const float *)[by contents], (size_t)e * sizeof(float));
        waste_metal_t_copy += mt_now() - t_c1;
    }
    pthread_mutex_unlock(&g_mu);
}

/* ---- registration ------------------------------------------------------- */

void waste_metal_selftest(void)
{
    if (!g.ready || g.ready < 0) return;

        @autoreleasepool {
            const unsigned M = 12288, K = 7168, G = 128, ng = K/G, rb = K/2;
            const size_t wb = (size_t)M * rb;
            const int NC = 32;
            NSMutableArray *drv = [NSMutableArray array], *noc = [NSMutableArray array];
            NSMutableArray *sdrv = [NSMutableArray array], *snoc = [NSMutableArray array];
            for (int i = 0; i < NC; i++) {
                id<MTLBuffer> a1 = [g.dev newBufferWithLength:wb options:MTLResourceStorageModeShared];
                id<MTLBuffer> a2 = [g.dev newBufferWithLength:(size_t)M*ng*2 options:MTLResourceStorageModeShared];
                memset(a1.contents, 0x37 + i, wb); memset(a2.contents, 0x38, (size_t)M*ng*2);
                [drv addObject:a1]; [sdrv addObject:a2];
                void *hp = NULL, *sp = NULL;
                if (posix_memalign(&hp, g.page ? g.page : 16384, (wb + 16383) & ~(size_t)16383) ||
                    posix_memalign(&sp, g.page ? g.page : 16384, ((size_t)M*ng*2 + 16383) & ~(size_t)16383))
                    continue;
                memset(hp, 0x37 + i, wb); memset(sp, 0x38, (size_t)M*ng*2);
                [noc addObject:[g.dev newBufferWithBytesNoCopy:hp length:(wb + 16383) & ~(size_t)16383
                                                      options:MTLResourceStorageModeShared deallocator:nil]];
                [snoc addObject:[g.dev newBufferWithBytesNoCopy:sp length:((size_t)M*ng*2 + 16383) & ~(size_t)16383
                                                       options:MTLResourceStorageModeShared deallocator:nil]];
            }
            id<MTLBuffer> X = [g.dev newBufferWithLength:K*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> Y = [g.dev newBufferWithLength:M*4 options:MTLResourceStorageModeShared];
            float *hostx = (float *)malloc(K*4); float *hosty = (float *)malloc(M*4);
            struct MtParams pr = { (int)K, (int)ng, (int)G, 4, rb, M };
            const char *nm[4] = { "driver buf, reused ", "driver buf, 32 rot ",
                                  "nocopy host, 32 rot", "nocopy + x/y memcpy" };
            for (int mode = 0; mode < 4; mode++) {
                if (mode >= 2 && noc.count < (NSUInteger)NC) continue;
                double best = 1e18, sum = 0;
                const int R = 64;
                for (int r = 0; r < R; r++) {
                    const int c = (mode == 0) ? 0 : r % NC;
                    id<MTLBuffer> BW = (mode >= 2) ? noc[c] : drv[c];
                    id<MTLBuffer> BS = (mode >= 2) ? snoc[c] : sdrv[c];
                    const double t0 = mt_now();
                    if (mode == 3) memcpy(X.contents, hostx, K*4);
                    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:g.mvq4];
                    [enc setBuffer:Y offset:0 atIndex:0]; [enc setBuffer:BW offset:0 atIndex:1];
                    [enc setBuffer:BS offset:0 atIndex:2]; [enc setBuffer:X offset:0 atIndex:3];
                    [enc setBytes:&pr length:sizeof pr atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(M/8,1,1)
                        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (mode == 3) memcpy(hosty, Y.contents, M*4);
                    const double dt = mt_now() - t0;
                    if (dt < best) best = dt; if (r >= 8) sum += dt;
                }
                fprintf(stderr, "waste: metal selftest %s  mean %.3f ms %6.1f GB/s  best %6.1f GB/s\n",
                        nm[mode], sum/(R-8)*1e3, wb/(sum/(R-8))/1e9, wb/best/1e9);
            }
            free(hostx); free(hosty);
        }
    }

const char *waste_register_metal(waste_kernels *t)
{
    pthread_mutex_lock(&g_mu);
    @autoreleasepool {
        if (g.ready) {
            const char *name = g.ready > 0 ? "Metal" : NULL;
            pthread_mutex_unlock(&g_mu);
            return name;
        }
        g.dev = MTLCreateSystemDefaultDevice();
        if (!g.dev || ![g.dev hasUnifiedMemory]) {
            g.ready = -1;
            pthread_mutex_unlock(&g_mu);
            return NULL;
        }
        NSError *err = nil;
        id<MTLLibrary> lib = [g.dev newLibraryWithSource:kSrc options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "waste: Metal shader did not compile: %s\n",
                    [[err localizedDescription] UTF8String]);
            g.ready = -1;
            pthread_mutex_unlock(&g_mu);
            return NULL;
        }
        g.mvq4 = [g.dev newComputePipelineStateWithFunction:
                     [lib newFunctionWithName:@"mvq4"] error:&err];
        g.mvq8 = [g.dev newComputePipelineStateWithFunction:
                     [lib newFunctionWithName:@"mvq8"] error:&err];
        g.mvq3 = [g.dev newComputePipelineStateWithFunction:
                     [lib newFunctionWithName:@"mvq3"] error:&err];
        g.queue = [g.dev newCommandQueue];
        if (!g.mvq4 || !g.mvq8 || !g.mvq3 || !g.queue) {
            g.ready = -1;
            pthread_mutex_unlock(&g_mu);
            return NULL;
        }
        g.page = (size_t)getpagesize();
        g.ready = 1;
    }
    /* WASTE_METAL_SELFTEST=1 runs the same kernel on a buffer this file
     * allocated, inside the process that is about to use it. It answers the
     * one question the standalone benchmark cannot: whether a dispatch that
     * measures 0.33 ms on its own measures 0.33 ms here. */
    if (getenv("WASTE_METAL_SELFTEST")) waste_metal_selftest();
    t->mvq_rows_f32 = mvq_rows_f32_metal;
    t->on_device = 1;      /* one dispatch for the whole range, not per thread */
    /* A command buffer costs about 11 us to commit and wait for on this
     * machine (tools/metalbw.m). Six threads of NEON clear a megabyte of
     * Q4G in well under that, so anything smaller stays on the pool. */
    { const char *e = getenv("WASTE_METAL_MIN_KB");
      t->device_min_bytes = (size_t)(e ? atoi(e) : 1024) << 10; }
    pthread_mutex_unlock(&g_mu);
    return "Metal";
}

void waste_metal_release_host_buffers(void)
{
    pthread_mutex_lock(&g_mu);
    @autoreleasepool {
        for (size_t i = 0; i < MT_SLOTS; i++) {
            g.slot[i].buf = nil;
            g.slot[i].key = NULL;
        }
    }
    pthread_mutex_unlock(&g_mu);
}
