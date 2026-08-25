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

struct P { uint in; uint ng; uint rowbytes; uint m; uint nv; uint st; uint en; };

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

typedef struct { unsigned in, ng, rowbytes, m, nv, st, en; } P;

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

    /* ---- 1. streaming bandwidth, 2 GB ---- */
    {
        const size_t bytes = 2ull << 30;
        id<MTLBuffer> W = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        memset(W.contents, 1, bytes);
        id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        P p = {0,0,(unsigned)(bytes/16),0,0,0,0};
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
        P p = {K, ng, rowbytes, M, 0,0,0};
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
        P p = {0,0,0,M,nv,ST,EN};
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
