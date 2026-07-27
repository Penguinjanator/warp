/*
 * waste_backend.h — kernel dispatch, modelled on sqlite-vector.
 *
 * sqlite-vector keeps one global table of function pointers, fills it with
 * a baseline CPU implementation that is always compiled in, then lets the
 * best backend detected at runtime overwrite the entries it implements
 * (see src/distance-cpu.c:init_distance_functions there). WASTE uses the
 * same discipline, with two tiers:
 *
 *   tier 1 — in-process SIMD (NEON, AVX2, AVX-512, SVE, RVV): compiled into
 *            the library, selected by runtime CPU detection. No load step.
 *   tier 2 — external accelerators (CUDA, Metal, BLAS, ROCm, Vulkan):
 *            separate shared objects opened with dlopen/LoadLibrary at
 *            runtime. Absent library, or absent hardware = no acceleration,
 *            never a link error and never a failed start.
 *
 * Invariants:
 *   - the universal CPU path is ALWAYS available and always correct; every
 *     other backend is an optimization that must produce the same result;
 *   - a backend may implement any subset of kernels — unimplemented slots
 *     keep the CPU version;
 *   - dispatch is resolved once at init, never per call in a hot loop;
 *   - WASTE_BACKEND=cpu (env) or waste_backend_init(WASTE_BE_FORCE_CPU)
 *     disables acceleration, for bisecting numeric differences.
 *
 * Because WASTE's kernels have heterogeneous signatures (unlike
 * sqlite-vector's distance functions, which all share one), the table is a
 * struct of function pointers rather than a 2-D array. Same idea.
 */

#ifndef WASTE_BACKEND_H
#define WASTE_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- capability bits --------------------------------------------------- */

typedef enum {
    WASTE_CPU_NEON     = 1u << 0,
    WASTE_CPU_DOTPROD  = 1u << 1,   /* ARM SDOT/UDOT                        */
    WASTE_CPU_I8MM     = 1u << 2,   /* ARM SMMLA                            */
    WASTE_CPU_SVE      = 1u << 3,
    WASTE_CPU_SSE2     = 1u << 8,
    WASTE_CPU_AVX2     = 1u << 9,
    WASTE_CPU_FMA      = 1u << 10,
    WASTE_CPU_AVX512F  = 1u << 11,
    WASTE_CPU_AVX512BW = 1u << 12,
    WASTE_CPU_AVX_VNNI = 1u << 13,
    WASTE_CPU_RVV      = 1u << 20,
} waste_cpu_feature;

/* Detected once; safe to call repeatedly. */
uint32_t waste_cpu_features(void);

/* ---- the dispatch table ------------------------------------------------ */

typedef struct {
    /* KDA (see kda.h for semantics) */
    void (*kda_step)(int H, int K, int V,
                     const float *q, const float *k, const float *v,
                     const float *g_log, const float *beta,
                     float *S, float *o, float *scratch);
    void (*short_conv_step)(int C, int KS, const float *w, const float *bias,
                            float *ring, const float *x, float *y);
    void (*rmsnorm_gated)(int C, const float *x, const float *gate,
                          const float *weight, float eps, float *y);

    /* quantized matmul / dequant slots land here as the engine grows:
     *   dot_vq2, dot_vq3, matmul_q4g, dequant_expert, ...
     * A backend implementing only some of them leaves the rest at CPU. */
} waste_kernels;

/* The live table. Read it after waste_backend_init(); never mutate it from
 * outside a backend's registration function. */
extern waste_kernels waste_k;

/* ---- init / introspection ---------------------------------------------- */

typedef enum {
    WASTE_BE_AUTO      = 0,
    WASTE_BE_FORCE_CPU = 1u << 0,  /* baseline only — the numeric reference */
    WASTE_BE_NO_EXTERNAL = 1u << 1, /* skip dlopen of accelerator plugins   */
} waste_backend_flags;

/* Fills the table: CPU baseline first, then the best in-process SIMD
 * backend for this machine, then any external plugin that claims to be
 * faster. Idempotent. Honours the WASTE_BACKEND environment variable
 * ("cpu", "neon", "avx2", "avx512", "cuda", "metal", ...). */
void waste_backend_init(unsigned flags);

/* e.g. "CPU", "NEON+dotprod", "AVX-512", "CUDA (sm_90)". Never NULL. */
const char *waste_backend_name(void);

/* Load an accelerator plugin explicitly (a .so/.dylib/.dll exporting
 * waste_backend_register). Returns 0 on success. Safe to call with a name
 * that does not exist: the engine keeps running on what it already has. */
int waste_backend_load(const char *path_or_name);

/* Every backend module exports this symbol; it overwrites the slots it
 * implements and returns a human-readable name, or NULL to decline (e.g.
 * no compatible GPU present). */
typedef const char *(*waste_backend_register_fn)(waste_kernels *table);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_BACKEND_H */
