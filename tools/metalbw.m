/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * metalbw.m — the gate docs/BACKENDS.md's Metal section deserves re-running.
 *
 * "Metal: it works, and it loses" (2026-07-28) measured one kernel that
 * applies a group scale per element, reduces through threadgroup memory
 * with log2(n) barriers, and puts one threadgroup on one row. It reported
 * 53 GB/s against the CPU's 195 on lm_head and concluded that the shape of
 * this engine, not the kernel, was the problem. docs/LEARNED.md §61 then
 * found the same conclusion stated from a mechanism rather than measured on
 * a coherent-memory part — which is exactly what this machine is.
 *
 * So: no engine, no integration, just the three kernels a decode step is
 * actually made of, measured on the GPU at K3's real shapes against the
 * CPU numbers tools/mvqbw.c produces.
 *
 *   clang -O2 -fobjc-arc -framework Metal -framework Foundation \
 *         -o metalbw tools/metalbw.m
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double now(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct P { uint in; uint ng; uint rowbytes; uint m; uint nv; uint st; uint en; uint st2; };

/* ---- 1. how fast can this GPU read host memory at all ---------------- */
kernel void streamsum(device const uint4 *W [[buffer(0)]],
                      device uint *out      [[buffer(1)]],
                      constant P &p         [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint gsz [[threads_per_grid]])
{
    uint4 acc = uint4(0);
    const uint n = p.rowbytes;              /* in uint4 units */
    for (uint i = gid; i < n; i += gsz) acc += W[i];
    if ((acc.x | acc.y | acc.z | acc.w) == 0xffffffffu) out[0] = 1;
}

/* ---- 2. the Q4G trunk matvec -----------------------------------------
 * One simdgroup per output row, uint4 loads strided by the simdgroup so a
 * warp's loads coalesce, the group scale applied once per group of 128
 * rather than once per weight, and simd_sum instead of a barrier ladder.
 * All three are what the 2026-07-28 kernel did the other way. */
kernel void mvq4(device float             *y  [[buffer(0)]],
                 device const uint4       *W  [[buffer(1)]],
                 device const half        *ws [[buffer(2)]],
                 device const float       *x  [[buffer(3)]],
                 constant P               &p  [[buffer(4)]],
                 uint  tgid [[threadgroup_position_in_grid]],
                 uint  tid  [[thread_index_in_threadgroup]],
                 uint  sgid [[simdgroup_index_in_threadgroup]],
                 uint  lane [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8 + sgid;               /* 8 rows per group */
    if (row >= p.m) return;
    device const uint4 *w = W + (ulong)row * (p.rowbytes >> 4);
    device const half  *sc = ws + (ulong)row * p.ng;
    const uint nvec = p.rowbytes >> 4;              /* uint4 per row */
    float acc = 0.0f;
    for (uint j = lane; j < nvec; j += 32) {
        const uint4 packed = w[j];
        const uint base = j << 5;                   /* 32 weights per uint4 */
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
        acc += float(sc[base >> 7]) * part;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
    (void)tid;
}


/* ---- 2b. the same matvec with x deinterleaved ------------------------
 * A packed byte holds an even and an odd weight, so with x split into an
 * even and an odd plane on the host the unpack becomes two vector ops and
 * the multiply becomes two dot()s — no per-nibble scalar extract at all. */
kernel void mvq4v(device float             *y  [[buffer(0)]],
                  device const uchar4      *W  [[buffer(1)]],
                  device const half        *ws [[buffer(2)]],
                  device const float4      *xe [[buffer(3)]],
                  device const float4      *xo [[buffer(5)]],
                  constant P               &p  [[buffer(4)]],
                  uint  tgid [[threadgroup_position_in_grid]],
                  uint  sgid [[simdgroup_index_in_threadgroup]],
                  uint  lane [[thread_index_in_simdgroup]])
{
    const uint row = tgid * 8 + sgid;
    if (row >= p.m) return;
    device const uchar4 *w = W + (ulong)row * (p.rowbytes >> 2);
    device const half   *sc = ws + (ulong)row * p.ng;
    const uint nvec = p.rowbytes >> 2;              /* uchar4 per row */
    float acc = 0.0f;
    for (uint j = lane; j < nvec; j += 32) {
        const uchar4 b4 = w[j];
        const float4 lo = float4(int4(b4 & uchar4(0x0f)) - 8);
        const float4 hi = float4(int4(b4 >> uchar4(4))  - 8);
        /* byte j covers weights 8j..8j+7, i.e. even indices 4j..4j+3 */
        acc += float(sc[(j << 3) >> 7]) * (dot(lo, xe[j]) + dot(hi, xo[j]));
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

/* ---- 5. VQ3R apply, 256 rows a threadgroup ---------------------------- */
kernel void vq3r_lut256(device float           *y   [[buffer(0)]],
                        device const uchar     *idx [[buffer(1)]],
                        device const float     *lut [[buffer(2)]],
                        constant P             &p   [[buffer(3)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tid  [[thread_index_in_threadgroup]])
{
    const uint blk = tgid * 4 + (tid >> 6);          /* 4 index blocks */
    const uint r   = tid & 63;
    float acc = 0.0f;
    for (uint v = 0; v < p.nv; v++) {
        device const float *b = lut + (ulong)v * p.st * p.en;
        device const uchar *ix = idx + ((ulong)blk * p.nv + v) * 64 * p.st + r * p.st;
        acc += b[ix[0]] + b[p.en + ix[1]] + b[2 * p.en + ix[2]];
    }
    y[blk * 64 + r] = acc;
}

/* ---- 6. VQ3R apply by reconstruction ---------------------------------
 * docs/LEARNED.md §50 measured on CUDA that a GPU prefers rebuilding the
 * weights to walking the table: the table is 864 KB and cannot leave L2,
 * the codebook is 24 KB and fits shared memory, and on a GPU the FLOPs are
 * free while the gathers are not. Here the codebook goes to threadgroup
 * memory as half — 12 KB for three stages — and every lookup becomes a
 * 16-byte contiguous read instead of a 4-byte divergent one. */
kernel void vq3r_dec(device float           *y   [[buffer(0)]],
                     device const uchar     *idx [[buffer(1)]],
                     device const half      *cb  [[buffer(2)]],  /* [st][en][vd] */
                     device const float     *x   [[buffer(3)]],
                     constant P             &p   [[buffer(4)]],
                     threadgroup half       *C   [[threadgroup(0)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint tid  [[thread_index_in_threadgroup]])
{
    const uint n = p.st * p.en * 8;
    for (uint i = tid; i < n; i += 256) C[i] = cb[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint blk = tgid * 4 + (tid >> 6);
    const uint r   = tid & 63;
    float acc = 0.0f;
    for (uint v = 0; v < p.nv; v++) {
        device const uchar *ix = idx + ((ulong)blk * p.nv + v) * 64 * p.st + r * p.st;
        threadgroup const half *c0 = C + (ulong)ix[0] * 8;
        threadgroup const half *c1 = C + (ulong)(p.en + ix[1]) * 8;
        threadgroup const half *c2 = C + (ulong)(2 * p.en + ix[2]) * 8;
        device const float *xv = x + v * 8;
        float t = 0.0f;
        for (uint d = 0; d < 8; d++)
            t += (float(c0[d]) + float(c1[d]) + float(c2[d])) * xv[d];
        acc += t;
    }
    y[blk * 64 + r] = acc;
}

/* ---- 8. the same apply, with the v loop split across threadgroups -----
 * The occupancy finding in §5a needed 73,728 rows in one dispatch, which
 * means every routed expert's record held before any arithmetic starts —
 * the barrier docs/LEARNED.md §44 measured as a regression on K3. But rows
 * are not the only axis. One apply loops over nv = 448 vector positions
 * and each is independent, so splitting that loop S ways multiplies the
 * threads in flight by S without touching how many experts are in hand.
 * Partial sums go to a [S][M] buffer and a second pass adds them in a
 * fixed order, so the result does not depend on S. */
kernel void vq3r_split(device float           *part [[buffer(0)]],
                       device const uchar     *idx  [[buffer(1)]],
                       device const float     *lut  [[buffer(2)]],
                       constant P             &p    [[buffer(3)]],
                       uint2 tg  [[threadgroup_position_in_grid]],
                       uint  tid [[thread_index_in_threadgroup]])
{
    const uint blk = tg.x, sp = tg.y, S = p.st2;
    const uint v0 = (p.nv * sp) / S, v1 = (p.nv * (sp + 1)) / S;
    float acc = 0.0f;
    for (uint v = v0; v < v1; v++) {
        device const float *b = lut + (ulong)v * p.st * p.en;
        device const uchar *ix = idx + ((ulong)blk * p.nv + v) * 64 * p.st + tid * p.st;
        acc += b[ix[0]] + b[p.en + ix[1]] + b[2 * p.en + ix[2]];
    }
    part[(ulong)sp * p.m + blk * 64 + tid] = acc;
}

kernel void vq3r_reduce(device float       *y    [[buffer(0)]],
                        device const float *part [[buffer(1)]],
                        constant P         &p    [[buffer(2)]],
                        uint gid [[thread_position_in_grid]])
{
    if (gid >= p.m) return;
    float a = 0.0f;
    for (uint s = 0; s < p.st2; s++) a += part[(ulong)s * p.m + gid];
    y[gid] = a;
}

/* ---- 3. the VQ3R expert apply, through the precomputed table ----------
 * lut is [v][stage][code]; idx is [rowblock][v][row][stage] with a 64-row
 * block, which is what src/model.c's vq_rows walks. One thread per row,
 * 64 rows a threadgroup, so a warp's three index bytes are 96 contiguous
 * bytes and the table block for v is shared by every thread in the group. */
kernel void vq3r_lut(device float           *y   [[buffer(0)]],
                     device const uchar     *idx [[buffer(1)]],
                     device const float     *lut [[buffer(2)]],
                     constant P             &p   [[buffer(3)]],
                     uint tgid [[threadgroup_position_in_grid]],
                     uint tid  [[thread_index_in_threadgroup]])
{
    const uint row = tgid * 64 + tid;
    float acc = 0.0f;
    for (uint v = 0; v < p.nv; v++) {
        device const float *blk = lut + (ulong)v * p.st * p.en;
        device const uchar *ix = idx + ((ulong)tgid * p.nv + v) * 64 * p.st + tid * p.st;
        acc += blk[ix[0]] + blk[p.en + ix[1]] + blk[2 * p.en + ix[2]];
    }
    y[row] = acc;
}

/* ---- 4. the same apply with the table held in threadgroup memory ------
 * The table for one v is 3 KB; a group of 64 rows reads it 64 times out of
 * device memory in kernel 3. Staging it once per v costs one barrier pair
 * per v and turns 64 device reads into 64 threadgroup reads. */
kernel void vq3r_tg(device float           *y   [[buffer(0)]],
                    device const uchar     *idx [[buffer(1)]],
                    device const float     *lut [[buffer(2)]],
                    constant P             &p   [[buffer(3)]],
                    threadgroup float      *T   [[threadgroup(0)]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint tid  [[thread_index_in_threadgroup]])
{
    const uint row = tgid * 64 + tid;
    const uint tab = p.st * p.en;                    /* 768 floats */
    float acc = 0.0f;
    for (uint v = 0; v < p.nv; v++) {
        device const float *blk = lut + (ulong)v * tab;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < tab; i += 64) T[i] = blk[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const uchar *ix = idx + ((ulong)tgid * p.nv + v) * 64 * p.st + tid * p.st;
        acc += T[ix[0]] + T[p.en + ix[1]] + T[2 * p.en + ix[2]];
    }
    y[row] = acc;
}
)MSL";

static id<MTLComputePipelineState> mkpipe(id<MTLDevice> d, id<MTLLibrary> lib, NSString *n)
{
    NSError *e = nil;
    id<MTLComputePipelineState> p =
        [d newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&e];
    if (!p) { fprintf(stderr, "pipeline %s: %s\n", n.UTF8String, e.description.UTF8String); exit(1); }
    return p;
}

typedef struct { unsigned in, ng, rowbytes, m, nv, st, en, st2; } P;

int main(int argc, char **argv)
{
    @autoreleasepool {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("device: %s   unified %d   max tg mem %lu KB\n",
           dev.name.UTF8String, (int)dev.hasUnifiedMemory,
           (unsigned long)dev.maxThreadgroupMemoryLength / 1024);
    NSError *err = nil;
    MTLCompileOptions *o = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:o error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLComputePipelineState> pStream = mkpipe(dev, lib, @"streamsum");
    id<MTLComputePipelineState> pMvq    = mkpipe(dev, lib, @"mvq4");
    id<MTLComputePipelineState> pLut    = mkpipe(dev, lib, @"vq3r_lut");
    id<MTLComputePipelineState> pTg     = mkpipe(dev, lib, @"vq3r_tg");
    id<MTLComputePipelineState> pMvqV   = mkpipe(dev, lib, @"mvq4v");
    id<MTLComputePipelineState> pLut256 = mkpipe(dev, lib, @"vq3r_lut256");
    id<MTLComputePipelineState> pDec    = mkpipe(dev, lib, @"vq3r_dec");
    id<MTLComputePipelineState> pSplit  = mkpipe(dev, lib, @"vq3r_split");
    id<MTLComputePipelineState> pRed    = mkpipe(dev, lib, @"vq3r_reduce");

    /* ---- 1. streaming bandwidth, 2 GB ---- */
    {
        const size_t bytes = 2ull << 30;
        id<MTLBuffer> W = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        memset(W.contents, 1, bytes);
        id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        P p = {0,0,(unsigned)(bytes/16),0,0,0,0,1};
        double best = 1e18;
        for (int r = 0; r < reps; r++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:pStream];
            [en setBuffer:W offset:0 atIndex:0];
            [en setBuffer:out offset:0 atIndex:1];
            [en setBytes:&p length:sizeof p atIndex:2];
            const double t0 = now();
            [en dispatchThreads:MTLSizeMake(1u<<20,1,1)
          threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
            const double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("\n1. stream 2 GB          %8.3f ms   %8.1f GB/s\n", best*1e3, bytes/best/1e9);
    }

    /* ---- 2. Q4G matvec at K3's trunk shape ---- */
    {
        const unsigned M = 12288, K = 7168, G = 128;
        const unsigned ng = K / G, rowbytes = K / 2;
        const size_t wbytes = (size_t)M * rowbytes;
        id<MTLBuffer> W = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> S = [dev newBufferWithLength:(size_t)M*ng*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> X = [dev newBufferWithLength:K*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:M*4 options:MTLResourceStorageModeShared];
        unsigned char *wp = W.contents;
        for (size_t i = 0; i < wbytes; i++) wp[i] = (unsigned char)(i * 37);
        unsigned short *sp = S.contents;
        for (size_t i = 0; i < (size_t)M*ng; i++) sp[i] = 0x3800;
        float *xp = X.contents;
        for (unsigned i = 0; i < K; i++) xp[i] = sinf(i * 0.017f) * 0.9f;
        P p = {K, ng, rowbytes, M, 0,0,0,1};
        double best = 1e18, sum = 0;
        for (int r = 0; r < reps; r++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:pMvq];
            [en setBuffer:Y offset:0 atIndex:0]; [en setBuffer:W offset:0 atIndex:1];
            [en setBuffer:S offset:0 atIndex:2]; [en setBuffer:X offset:0 atIndex:3];
            [en setBytes:&p length:sizeof p atIndex:4];
            const double t0 = now();
            [en dispatchThreadgroups:MTLSizeMake(M/8,1,1)
               threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
            const double dt = now() - t0;
            if (dt < best) best = dt; if (r) sum += dt;
        }
        const double mean = sum / (reps-1);
        printf("2. mvq4 [%u x %u] Q4G  %8.3f ms   %8.1f GB/s (best %.1f)\n",
               M, K, mean*1e3, wbytes/mean/1e9, wbytes/best/1e9);
        id<MTLBuffer> XE = [dev newBufferWithLength:K*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> XO = [dev newBufferWithLength:K*2 options:MTLResourceStorageModeShared];
        float *xe = XE.contents, *xo = XO.contents;
        for (unsigned i = 0; i < K; i += 2) { xe[i/2] = xp[i]; xo[i/2] = xp[i+1]; }
        best = 1e18; sum = 0;
        for (int r = 0; r < reps; r++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:pMvqV];
            [en setBuffer:Y offset:0 atIndex:0]; [en setBuffer:W offset:0 atIndex:1];
            [en setBuffer:S offset:0 atIndex:2]; [en setBuffer:XE offset:0 atIndex:3];
            [en setBytes:&p length:sizeof p atIndex:4];
            [en setBuffer:XO offset:0 atIndex:5];
            const double t0 = now();
            [en dispatchThreadgroups:MTLSizeMake(M/8,1,1)
               threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
            const double dt = now() - t0;
            if (dt < best) best = dt; if (r) sum += dt;
        }
        printf("2b.mvq4v deinterleaved x   %8.3f ms   %8.1f GB/s (best %.1f)\n",
               sum/(reps-1)*1e3, wbytes/(sum/(reps-1))/1e9, wbytes/best/1e9);
    }

    /* ---- 3/4. the VQ3R apply at K3's gate shape ---- */
    {
        const unsigned M = (argc > 2 ? (unsigned)atoi(argv[2]) : 3072), N = 3584, VD = 8, ST = 3, EN = 256;
        const unsigned nv = N / VD;                 /* 448 */
        const size_t ibytes = (size_t)M * nv * ST;  /* one index byte per stage */
        const size_t lbytes = (size_t)nv * ST * EN * 4;
        id<MTLBuffer> I = [dev newBufferWithLength:ibytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> L = [dev newBufferWithLength:lbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:M*4 options:MTLResourceStorageModeShared];
        unsigned char *ip = I.contents;
        for (size_t i = 0; i < ibytes; i++) ip[i] = (unsigned char)(i * 101 + (i >> 8));
        float *lp = L.contents;
        for (size_t i = 0; i < lbytes/4; i++) lp[i] = (float)((i % 251) - 125) * 0.01f;
        P p = {0,0,0,M,nv,ST,EN,1};
        /* codebook for the reconstruction arm: [st][en][vd] halves */
        id<MTLBuffer> CB = [dev newBufferWithLength:(size_t)ST*EN*VD*2
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> X2 = [dev newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
        {   __fp16 *cp = CB.contents; float *x2 = X2.contents;
            for (size_t i = 0; i < (size_t)ST*EN*VD; i++) cp[i] = (__fp16)(((i%97)-48)*0.01f);
            for (unsigned i = 0; i < N; i++) x2[i] = sinf(i*0.013f); }
        for (int which = 0; which < 4; which++) {
            id<MTLComputePipelineState> ps = which == 0 ? pLut : which == 1 ? pTg
                                           : which == 2 ? pLut256 : pDec;
            const int tgsz = which >= 2 ? 256 : 64;
            const int rows_per_tg = which >= 2 ? 256 : 64;
            double best = 1e18, sum = 0;
            for (int r = 0; r < reps; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                [en setComputePipelineState:ps];
                [en setBuffer:Y offset:0 atIndex:0]; [en setBuffer:I offset:0 atIndex:1];
                [en setBuffer:(which == 3 ? CB : L) offset:0 atIndex:2];
                if (which == 3) { [en setBuffer:X2 offset:0 atIndex:3];
                                  [en setBytes:&p length:sizeof p atIndex:4];
                                  [en setThreadgroupMemoryLength:ST*EN*VD*2 atIndex:0]; }
                else { [en setBytes:&p length:sizeof p atIndex:3];
                       if (which == 1) [en setThreadgroupMemoryLength:ST*EN*4 atIndex:0]; }
                const double t0 = now();
                [en dispatchThreadgroups:MTLSizeMake(M/rows_per_tg,1,1)
                   threadsPerThreadgroup:MTLSizeMake(tgsz,1,1)];
                [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
                const double dt = now() - t0;
                if (dt < best) best = dt; if (r) sum += dt;
            }
            const double mean = sum / (reps-1);
            const char *nm[4] = {"device lut  ","tg lut/v    ","device lut x4","codebook dec"};
            printf("%d. vq3r %-13s [%u x %u]  %8.3f ms   %8.1f GB/s index (best %.1f)\n",
                   3+which, nm[which], M, N,
                   mean*1e3, ibytes/mean/1e9, ibytes/best/1e9);
        }
    }
    /* ---- 7. does splitting one dispatch into N in-flight command buffers
     * cost the occupancy that made §5a work?
     *
     * It decides the design of a Metal MoE. One dispatch over every routed
     * expert's rows needs every record held first, which is exactly the
     * barrier docs/LEARNED.md §44 measured as a regression on K3 — the
     * hint has already queued all K reads and waiting for the last one
     * before starting the first expert stops them overlapping. If N
     * command buffers in flight aggregate on the GPU the way one big grid
     * does, the barrier is unnecessary: encode each expert as its record
     * arrives, commit without waiting, and join at the end of the layer. */
    {
        const unsigned M = 3072 * 16, N = 3584, VD = 8, ST = 3, EN = 256;
        const unsigned nv = N / VD;
        const size_t ibytes = (size_t)M * nv * ST;
        const size_t lbytes = (size_t)nv * ST * EN * 4;
        id<MTLBuffer> I = [dev newBufferWithLength:ibytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> L = [dev newBufferWithLength:lbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:M*4 options:MTLResourceStorageModeShared];
        memset(I.contents, 0x5a, ibytes);
        { float *lp = L.contents; for (size_t i = 0; i < lbytes/4; i++) lp[i] = (float)(i%251)*0.01f; }
        printf("\n7. one grid of %u rows against N command buffers of %u/N:\n", M, M);
        for (int parts = 1; parts <= 16; parts *= 2) {
            const unsigned rows = M / parts;
            P p = {0,0,0,rows,nv,ST,EN,1};
            double best = 1e18, sum = 0;
            for (int r = 0; r < reps; r++) {
                const double t0 = now();
                id<MTLCommandBuffer> last = nil;
                for (int k = 0; k < parts; k++) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                    [en setComputePipelineState:pLut];
                    [en setBuffer:Y offset:(NSUInteger)k*rows*4 atIndex:0];
                    [en setBuffer:I offset:(NSUInteger)k*rows*nv*ST atIndex:1];
                    [en setBuffer:L offset:0 atIndex:2];
                    [en setBytes:&p length:sizeof p atIndex:3];
                    [en dispatchThreadgroups:MTLSizeMake(rows/64,1,1)
                       threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                    [en endEncoding]; [cb commit];
                    last = cb;
                }
                [last waitUntilCompleted];
                const double dt = now() - t0;
                if (dt < best) best = dt; if (r) sum += dt;
            }
            const double mean = sum/(reps-1);
            printf("   %2d buffer(s) of %6u rows   %8.3f ms   %7.1f GB/s index\n",
                   parts, rows, mean*1e3, ibytes/mean/1e9);
        }
    }

    /* ---- 8. v-split ---- */
    {
        const unsigned M = 3072, N = 3584, VD = 8, ST = 3, EN = 256;
        const unsigned nv = N / VD;
        const size_t ibytes = (size_t)M * nv * ST;
        const size_t lbytes = (size_t)nv * ST * EN * 4;
        id<MTLBuffer> I = [dev newBufferWithLength:ibytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> L = [dev newBufferWithLength:lbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:M*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> PT = [dev newBufferWithLength:(size_t)M*4*64 options:MTLResourceStorageModeShared];
        memset(I.contents, 0x5a, ibytes);
        { float *lp = L.contents; for (size_t i = 0; i < lbytes/4; i++) lp[i] = (float)(i%251)*0.01f; }
        printf("\n8. one apply [%u x %u], the v loop split S ways:\n", M, N);
        for (int S = 1; S <= 32; S *= 2) {
            P p = {0,0,0,M,nv,ST,EN,(unsigned)S};
            double best = 1e18, sum = 0;
            for (int r = 0; r < reps; r++) {
                const double t0 = now();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                [en setComputePipelineState:pSplit];
                [en setBuffer:PT offset:0 atIndex:0]; [en setBuffer:I offset:0 atIndex:1];
                [en setBuffer:L offset:0 atIndex:2];
                [en setBytes:&p length:sizeof p atIndex:3];
                [en dispatchThreadgroups:MTLSizeMake(M/64,S,1)
                   threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                [en setComputePipelineState:pRed];
                [en setBuffer:Y offset:0 atIndex:0]; [en setBuffer:PT offset:0 atIndex:1];
                [en setBytes:&p length:sizeof p atIndex:2];
                [en dispatchThreads:MTLSizeMake(M,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
                const double dt = now() - t0;
                if (dt < best) best = dt; if (r) sum += dt;
            }
            const double mean = sum/(reps-1);
            printf("   S=%2d  %8.3f ms   %7.1f GB/s index (best %.1f)\n",
                   S, mean*1e3, ibytes/mean/1e9, ibytes/best/1e9);
        }
    }

    /* ---- 9. N dispatches inside ONE command buffer, concurrent encoder --
     * §7 showed N command buffers serialize. A concurrent compute encoder
     * is the other way to say "these do not depend on each other", and it
     * is the one that matters here: an expert's index lives in its own
     * cache slot, so a fused apply is N dispatches over N buffers rather
     * than one dispatch over a contiguous grid. If these overlap, the
     * engine can bind each routed expert's slot separately and still get
     * the occupancy §5a needed. */
    {
        const unsigned M = 3072 * 16, N = 3584, VD = 8, ST = 3, EN = 256;
        const unsigned nv = N / VD;
        const size_t ibytes = (size_t)M * nv * ST;
        const size_t lbytes = (size_t)nv * ST * EN * 4;
        id<MTLBuffer> I = [dev newBufferWithLength:ibytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> L = [dev newBufferWithLength:lbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:M*4 options:MTLResourceStorageModeShared];
        memset(I.contents, 0x5a, ibytes);
        { float *lp = L.contents; for (size_t i = 0; i < lbytes/4; i++) lp[i] = (float)(i%251)*0.01f; }
        printf("\n9. one command buffer, N concurrent dispatches of %u/N rows:\n", M);
        for (int parts = 1; parts <= 32; parts *= 2) {
            const unsigned rows = M / parts;
            P p = {0,0,0,rows,nv,ST,EN,1};
            double best = 1e18, sum = 0;
            for (int r = 0; r < reps; r++) {
                const double t0 = now();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> en =
                    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [en setComputePipelineState:pLut];
                [en setBuffer:L offset:0 atIndex:2];
                [en setBytes:&p length:sizeof p atIndex:3];
                for (int k = 0; k < parts; k++) {
                    [en setBuffer:Y offset:(NSUInteger)k*rows*4 atIndex:0];
                    [en setBuffer:I offset:(NSUInteger)k*rows*nv*ST atIndex:1];
                    [en dispatchThreadgroups:MTLSizeMake(rows/64,1,1)
                       threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                }
                [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
                const double dt = now() - t0;
                if (dt < best) best = dt; if (r) sum += dt;
            }
            const double mean = sum/(reps-1);
            printf("   %2d dispatch(es) of %6u rows  %8.3f ms  %7.1f GB/s index\n",
                   parts, rows, mean*1e3, ibytes/mean/1e9);
        }
    }

    /* ---- 10. what the engine does differently ---------------------------
     * §9 said sixteen concurrent dispatches reach 126 GB/s. In the engine
     * the same kernel at the same batch sizes measures 36. Three things
     * differ and this separates them: the CPU rewrites the table between
     * dispatches (it is built per token, and per expert for the down
     * projection), the CPU reads the output straight afterwards, and the
     * GPU idles between command buffers while the CPU does the SiTU and
     * the next table. */
    {
        const unsigned M = 3072 * 16, N = 3584, VD = 8, ST = 3, EN = 256;
        const unsigned nv = N / VD;
        const size_t ibytes = (size_t)M * nv * ST;
        const size_t lbytes = (size_t)nv * ST * EN * 4;
        id<MTLBuffer> I = [dev newBufferWithLength:ibytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> L = [dev newBufferWithLength:lbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:M*4 options:MTLResourceStorageModeShared];
        float *host_lut = (float *)malloc(lbytes);
        float *host_y = (float *)malloc(M*4);
        memset(I.contents, 0x5a, ibytes);
        for (size_t i = 0; i < lbytes/4; i++) host_lut[i] = (float)(i%251)*0.01f;
        memcpy(L.contents, host_lut, lbytes);
        const char *nm[4] = { "back to back            ",
                              "+ rewrite the table     ",
                              "+ read the output       ",
                              "+ 1 ms of CPU in between" };
        printf("\n10. sixteen concurrent dispatches, with the engine's habits:\n");
        for (int mode = 0; mode < 4; mode++) {
            P p = {0,0,0,3072,nv,ST,EN,1};
            double sum = 0;
            for (int r = 0; r < reps; r++) {
                const double t0 = now();
                if (mode >= 1) memcpy(L.contents, host_lut, lbytes);
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> en =
                    [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [en setComputePipelineState:pLut];
                [en setBuffer:L offset:0 atIndex:2];
                [en setBytes:&p length:sizeof p atIndex:3];
                for (int k = 0; k < 16; k++) {
                    [en setBuffer:Y offset:(NSUInteger)k*3072*4 atIndex:0];
                    [en setBuffer:I offset:(NSUInteger)k*3072*nv*ST atIndex:1];
                    [en dispatchThreadgroups:MTLSizeMake(3072/64,1,1)
                       threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                }
                [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
                if (mode >= 2) memcpy(host_y, Y.contents, M*4);
                if (mode >= 3) { const double t1 = now();
                                 volatile double z = 0;
                                 while (now() - t1 < 0.001) z += 1; (void)z; }
                const double dt = now() - t0 - (mode >= 3 ? 0.001 : 0);
                if (r) sum += dt;
            }
            const double mean = sum/(reps-1);
            printf("   %s %8.3f ms  %7.1f GB/s index\n", nm[mode], mean*1e3,
                   ibytes/mean/1e9);
        }
        free(host_lut); free(host_y);
    }

    /* dispatch floor */
    {
        double best = 1e18;
        for (int r = 0; r < 200; r++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:pStream];
            [en endEncoding];
            const double t0 = now();
            [cb commit]; [cb waitUntilCompleted];
            const double dt = now() - t0; if (dt < best) best = dt;
        }
        printf("\nempty command buffer floor: %.1f us\n", best*1e6);
    }
    }
    return 0;
}
