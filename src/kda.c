/*
 * kda.c — Kimi Delta Attention kernels (see kda.h for the recurrence).
 *
 * The decode step touches the recurrent state exactly twice:
 *   pass A: decay rows by exp(g) and accumulate u = S'^T k
 *   pass B: rank-1 update S += k d^T and accumulate o = S^T q
 * Both walk S row-major, so the state streams linearly through cache.
 */

#include "kda.h"

#include <math.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define WASTE_SIMD_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define WASTE_SIMD_AVX2 1
#endif

static float l2_rnorm(const float *x, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return 1.0f / sqrtf(s + 1e-12f);
}

void waste_kda_step(int H, int K, int V,
                    const float *q, const float *k, const float *v,
                    const float *g_log, const float *beta,
                    float *S, float *o, float *u)
{
    const float qscale = 1.0f / sqrtf((float)K);

    for (int h = 0; h < H; h++) {
        const float *qh = q + (size_t)h * K;
        const float *kh = k + (size_t)h * K;
        const float *vh = v + (size_t)h * V;
        const float *gh = g_log + (size_t)h * K;
        float *Sh = S + (size_t)h * K * V;
        float *oh = o + (size_t)h * V;
        const float b = beta[h];

        /* q, k are L2-normalized per head; q additionally scaled by K^-0.5 */
        const float qn = l2_rnorm(qh, K) * qscale;
        const float kn = l2_rnorm(kh, K);

        memset(u, 0, (size_t)V * sizeof(float));

        /* pass A: S <- Diag(exp(g)) S, and u = S^T k (post-decay) */
        for (int kk = 0; kk < K; kk++) {
            float *row = Sh + (size_t)kk * V;
            const float d = expf(gh[kk]);
            const float kv = kh[kk] * kn;
#if WASTE_SIMD_NEON
            const float32x4_t vd = vdupq_n_f32(d), vk = vdupq_n_f32(kv);
            int i = 0;
            for (; i + 4 <= V; i += 4) {
                float32x4_t r = vmulq_f32(vld1q_f32(row + i), vd);
                vst1q_f32(row + i, r);
                vst1q_f32(u + i, vfmaq_f32(vld1q_f32(u + i), r, vk));
            }
            for (; i < V; i++) { row[i] *= d; u[i] += row[i] * kv; }
#elif WASTE_SIMD_AVX2
            const __m256 vd = _mm256_set1_ps(d), vk = _mm256_set1_ps(kv);
            int i = 0;
            for (; i + 8 <= V; i += 8) {
                __m256 r = _mm256_mul_ps(_mm256_loadu_ps(row + i), vd);
                _mm256_storeu_ps(row + i, r);
                _mm256_storeu_ps(u + i,
                    _mm256_fmadd_ps(r, vk, _mm256_loadu_ps(u + i)));
            }
            for (; i < V; i++) { row[i] *= d; u[i] += row[i] * kv; }
#else
            for (int i = 0; i < V; i++) { row[i] *= d; u[i] += row[i] * kv; }
#endif
        }

        /* delta: d = beta * (v - u), reused as the rank-1 right factor */
        for (int i = 0; i < V; i++) u[i] = b * (vh[i] - u[i]);

        /* pass B: S += k d^T, and o = S^T q (post-update) */
        memset(oh, 0, (size_t)V * sizeof(float));
        for (int kk = 0; kk < K; kk++) {
            float *row = Sh + (size_t)kk * V;
            const float kv = kh[kk] * kn;
            const float qv = qh[kk] * qn;
#if WASTE_SIMD_NEON
            const float32x4_t vk = vdupq_n_f32(kv), vq = vdupq_n_f32(qv);
            int i = 0;
            for (; i + 4 <= V; i += 4) {
                float32x4_t r = vfmaq_f32(vld1q_f32(row + i), vld1q_f32(u + i), vk);
                vst1q_f32(row + i, r);
                vst1q_f32(oh + i, vfmaq_f32(vld1q_f32(oh + i), r, vq));
            }
            for (; i < V; i++) { row[i] += u[i] * kv; oh[i] += row[i] * qv; }
#elif WASTE_SIMD_AVX2
            const __m256 vk = _mm256_set1_ps(kv), vq = _mm256_set1_ps(qv);
            int i = 0;
            for (; i + 8 <= V; i += 8) {
                __m256 r = _mm256_fmadd_ps(_mm256_loadu_ps(u + i), vk,
                                           _mm256_loadu_ps(row + i));
                _mm256_storeu_ps(row + i, r);
                _mm256_storeu_ps(oh + i,
                    _mm256_fmadd_ps(r, vq, _mm256_loadu_ps(oh + i)));
            }
            for (; i < V; i++) { row[i] += u[i] * kv; oh[i] += row[i] * qv; }
#else
            for (int i = 0; i < V; i++) { row[i] += u[i] * kv; oh[i] += row[i] * qv; }
#endif
        }
    }
}

void waste_kda_forward(int T, int H, int K, int V,
                       const float *q, const float *k, const float *v,
                       const float *g_log, const float *beta,
                       float *S, float *o, float *scratch)
{
    for (int t = 0; t < T; t++) {
        waste_kda_step(H, K, V,
                       q + (size_t)t * H * K, k + (size_t)t * H * K,
                       v + (size_t)t * H * V, g_log + (size_t)t * H * K,
                       beta + (size_t)t * H,
                       S, o + (size_t)t * H * V, scratch);
    }
}

void waste_short_conv_step(int C, int KS, const float *w, const float *bias,
                           float *ring, const float *x, float *y)
{
    /* ring holds the previous KS-1 samples per channel, oldest first */
    const int R = KS - 1;
    for (int c = 0; c < C; c++) {
        const float *wc = w + (size_t)c * KS;
        float *rc = ring + (size_t)c * R;
        float acc = bias ? bias[c] : 0.0f;
        for (int j = 0; j < R; j++) acc += rc[j] * wc[j];
        acc += x[c] * wc[R];
        for (int j = 0; j + 1 < R; j++) rc[j] = rc[j + 1];
        if (R > 0) rc[R - 1] = x[c];
        y[c] = acc / (1.0f + expf(-acc));       /* SiLU */
    }
}

void waste_rmsnorm_gated(int C, const float *x, const float *gate,
                         const float *weight, float eps, float *y)
{
    float s = 0.0f;
    for (int i = 0; i < C; i++) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(s / (float)C + eps);
    for (int i = 0; i < C; i++) {
        const float g = 1.0f / (1.0f + expf(-gate[i]));
        y[i] = x[i] * r * weight[i] * g;
    }
}
