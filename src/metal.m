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
#include <stdlib.h>
#include <string.h>

static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct P { int in; int ng; int group; int bits; uint rowbytes; };

/* One threadgroup per output row, threads striding the input so their loads
 * are coalesced, then a threadgroup reduction.
 *
 * The first version put one thread on a whole row. That is the obvious
 * mapping and the wrong one: adjacent threads then read addresses rowbytes
 * apart, so every load is its own cache line, and a 7168-long dot runs
 * serially in one lane.
 *
 * Group scales are applied per element rather than per group, which lets a
 * thread take any index without caring where the group boundaries fall. It
 * is the same arithmetic — sum_k s_k sum_i w_i x_i == sum_i s_{k(i)} w_i x_i
 * — for one extra half-to-float per element. */
kernel void mvq_f32(device float             *y   [[buffer(0)]],
                    device const uchar       *W   [[buffer(1)]],
                    device const ushort      *ws  [[buffer(2)]],
                    device const float       *x   [[buffer(3)]],
                    constant P               &p   [[buffer(4)]],
                    uint  row   [[threadgroup_position_in_grid]],
                    uint  tid   [[thread_position_in_threadgroup]],
                    uint  nthr  [[threads_per_threadgroup]],
                    threadgroup float *part [[threadgroup(0)]])
{
    device const uchar  *W0 = W  + (ulong)row * p.rowbytes;
    device const ushort *sc = ws + (ulong)row * p.ng;
    const int gshift = (p.group == 128) ? 7 : int(log2(float(p.group)) + 0.5f);

    float acc = 0.0f;
    if (p.bits == 8) {
        device const char *w = (device const char *)W0;
        for (int i = int(tid); i < p.in; i += int(nthr))
            acc += float(w[i]) * x[i] * float(as_type<half>(sc[i >> gshift]));
    } else if (p.bits == 4) {
        for (int i = int(tid); i < p.in; i += int(nthr)) {
            const uchar byte = W0[i >> 1];
            const int v = (i & 1) ? int(byte >> 4) - 8 : int(byte & 0x0F) - 8;
            acc += float(v) * x[i] * float(as_type<half>(sc[i >> gshift]));
        }
    } else {
        for (int i = int(tid); i < p.in; i += int(nthr)) {
            const long off = (long)i * 3;
            const int b0 = int(W0[off >> 3]), b1 = int(W0[(off >> 3) + 1]);
            const int v = ((b0 >> (off & 7)) | (b1 << (8 - (off & 7)))) & 7;
            acc += float(v - 4) * x[i] * float(as_type<half>(sc[i >> gshift]));
        }
    }

    part[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthr / 2; s > 0; s >>= 1) {
        if (tid < s) part[tid] += part[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[row] = part[0];
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
    id<MTLComputePipelineState> mvq;
    mt_slot slot[MT_SLOTS];
    id<MTLBuffer> scratch_x, scratch_y;
    size_t scratch_x_cap, scratch_y_cap;
    size_t page;
} g;

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

struct MtParams { int in; int ng; int group; int bits; unsigned rowbytes; };

extern void waste_mvq_rows_f32(int b, int e, void *p);   /* CPU fallback */

static void mvq_rows_f32_metal(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    @autoreleasepool {
        id<MTLBuffer> bw = mt_wrap(a->W, (size_t)e * a->rowbytes);
        id<MTLBuffer> bs = mt_wrap(a->ws, (size_t)e * a->ng * sizeof(uint16_t));
        if (!bw || !bs) { waste_mvq_rows_f32(b, e, p); return; }

        const float *x = (const float *)a->xs;
        id<MTLBuffer> bx = mt_scratch(&g.scratch_x, &g.scratch_x_cap,
                                      (size_t)a->in * sizeof(float));
        id<MTLBuffer> by = mt_scratch(&g.scratch_y, &g.scratch_y_cap,
                                      (size_t)e * sizeof(float));
        if (!bx || !by) { waste_mvq_rows_f32(b, e, p); return; }
        memcpy([bx contents], x, (size_t)a->in * sizeof(float));

        struct MtParams pr = { a->in, a->ng, a->group, a->bits,
                               (unsigned)a->rowbytes };
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g.mvq];
        [enc setBuffer:by offset:0 atIndex:0];
        [enc setBuffer:bw offset:0 atIndex:1];
        [enc setBuffer:bs offset:0 atIndex:2];
        [enc setBuffer:bx offset:0 atIndex:3];
        [enc setBytes:&pr length:sizeof pr atIndex:4];
        NSUInteger tg = 128;                    /* power of two, for the reduction */
        if (tg > g.mvq.maxTotalThreadsPerThreadgroup)
            tg = g.mvq.maxTotalThreadsPerThreadgroup;
        [enc setThreadgroupMemoryLength:tg * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(e - b), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(a->y + b, (const float *)[by contents] + b,
               (size_t)(e - b) * sizeof(float));
    }
}

/* ---- registration ------------------------------------------------------- */

const char *waste_register_metal(waste_kernels *t)
{
    @autoreleasepool {
        if (g.ready) return g.ready > 0 ? "Metal" : NULL;
        g.dev = MTLCreateSystemDefaultDevice();
        if (!g.dev || ![g.dev hasUnifiedMemory]) { g.ready = -1; return NULL; }
        NSError *err = nil;
        id<MTLLibrary> lib = [g.dev newLibraryWithSource:kSrc options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "waste: Metal shader did not compile: %s\n",
                    [[err localizedDescription] UTF8String]);
            g.ready = -1;
            return NULL;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"mvq_f32"];
        g.mvq = [g.dev newComputePipelineStateWithFunction:fn error:&err];
        g.queue = [g.dev newCommandQueue];
        if (!g.mvq || !g.queue) { g.ready = -1; return NULL; }
        g.page = (size_t)getpagesize();
        g.ready = 1;
    }
    t->mvq_rows_f32 = mvq_rows_f32_metal;
    t->on_device = 1;      /* one dispatch for the whole range, not per thread */
    return "Metal";
}
