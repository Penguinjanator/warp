/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * crc32.c — CRC-32 (IEEE 802.3: reflected, polynomial 0xEDB88320).
 *
 * Reimplemented rather than linked: the engine has no dependencies, and
 * zlib would be one. It has to be bit-identical to zlib's crc32() all the
 * same, since that is the function that produced the value being checked.
 *
 * Two implementations, and the difference is worth the second one.
 * Measured here over a whole expert record (M-series core, one thread):
 *
 *     byte table        0.60 GB/s
 *     slice-by-8        2.70 GB/s
 *     armv8 crc32d     12.05 GB/s    one chain
 *     armv8 x3         33.03 GB/s    three chains, stitched in GF(2)
 *
 * crc32d retires one per cycle but takes about three to produce its
 * result, so a single dependent chain runs at a third of what the unit can
 * do. Three chains over three slices of the buffer keep it fed; putting
 * the slices back together costs two GF(2) combines per record, which is
 * microseconds against the 0.38 ms a K3 record takes either way.
 *
 * That is the number that matters: this runs once per cache miss, beside a
 * read of the same bytes. 0.38 ms of CRC against several ms of disk for
 * K3's 11.83 MB records is a fraction of the read it protects.
 *
 * The table path is what an aarch64 build without the CRC extension gets
 * (`make WASTE_NATIVE=1` turns it on), and what x86-64 gets always: SSE4.2
 * has a crc32 instruction, but it is Castagnoli — a different polynomial,
 * and no help for this one.
 */

#include "crc32.h"

#include <pthread.h>
#include <string.h>

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#define WASTE_CRC_ARMV8 1
#endif

/* ---- slice-by-8 table ---------------------------------------------------
 * Compiled only where it is the implementation. CI builds x86-64 Linux,
 * so it is exercised on every run rather than left to rot behind an
 * #ifdef nobody takes. */

#ifndef WASTE_CRC_ARMV8
static uint32_t g_tab[8][256];
static pthread_once_t g_tab_once = PTHREAD_ONCE_INIT;

static void crc_tab_build(void)
{
    for (unsigned i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(c & 1)));
        g_tab[0][i] = c;
    }
    for (unsigned i = 0; i < 256; i++)
        for (int s = 1; s < 8; s++)
            g_tab[s][i] = (g_tab[s - 1][i] >> 8) ^ g_tab[0][g_tab[s - 1][i] & 0xFF];
}

static uint32_t crc_slice8(uint32_t c, const uint8_t *p, size_t n)
{
    pthread_once(&g_tab_once, crc_tab_build);
    while (n && ((uintptr_t)p & 7)) { c = (c >> 8) ^ g_tab[0][(c ^ *p++) & 0xFF]; n--; }
    while (n >= 8) {
        uint32_t lo, hi;
        memcpy(&lo, p, 4);
        memcpy(&hi, p + 4, 4);
        lo ^= c;
        c = g_tab[7][lo & 0xFF]         ^ g_tab[6][(lo >> 8) & 0xFF] ^
            g_tab[5][(lo >> 16) & 0xFF] ^ g_tab[4][(lo >> 24) & 0xFF] ^
            g_tab[3][hi & 0xFF]         ^ g_tab[2][(hi >> 8) & 0xFF] ^
            g_tab[1][(hi >> 16) & 0xFF] ^ g_tab[0][(hi >> 24) & 0xFF];
        p += 8;
        n -= 8;
    }
    while (n--) c = (c >> 8) ^ g_tab[0][(c ^ *p++) & 0xFF];
    return c;
}
#endif /* !WASTE_CRC_ARMV8 */

#ifdef WASTE_CRC_ARMV8

/* ---- combining two CRCs ------------------------------------------------
 * Advancing a CRC register over len2 zero bytes is a linear map over
 * GF(2), so it is a 32x32 bit matrix, and the matrix for len2 bytes is
 * reached by repeated squaring — zlib's crc32_combine, which is where
 * this comes from. O(log len2) 32x32 products: a few microseconds, once
 * per slice boundary. */

static uint32_t gf2_mul_vec(const uint32_t *mat, uint32_t vec)
{
    uint32_t sum = 0;
    for (int i = 0; vec; i++, vec >>= 1)
        if (vec & 1) sum ^= mat[i];
    return sum;
}

static void gf2_square(uint32_t *sq, const uint32_t *mat)
{
    for (int i = 0; i < 32; i++) sq[i] = gf2_mul_vec(mat, mat[i]);
}

/* CRC of A||B, given crc(A), crc(B) and |B|. */
static uint32_t crc_combine(uint32_t crc1, uint32_t crc2, size_t len2)
{
    uint32_t even[32], odd[32];
    if (!len2) return crc1;
    odd[0] = 0xEDB88320u;                          /* one bit of shift */
    for (int n = 1; n < 32; n++) odd[n] = 1u << (n - 1);
    gf2_square(even, odd);                         /* two bits  */
    gf2_square(odd, even);                         /* four bits = one byte */
    do {
        gf2_square(even, odd);
        if (len2 & 1) crc1 = gf2_mul_vec(even, crc1);
        len2 >>= 1;
        if (!len2) break;
        gf2_square(odd, even);
        if (len2 & 1) crc1 = gf2_mul_vec(odd, crc1);
        len2 >>= 1;
    } while (len2);
    return crc1 ^ crc2;
}

static uint32_t crc_hw(uint32_t c, const uint8_t *p, size_t n)
{
    while (n >= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        c = __crc32d(c, v);
        p += 8;
        n -= 8;
    }
    while (n--) c = __crc32b(c, *p++);
    return c;
}

/* Below this the two combines cost more than the extra chains save. */
#define CRC_SPLIT_MIN 3072

static uint32_t crc_hw3(const uint8_t *p, size_t n)
{
    if (n < CRC_SPLIT_MIN) return crc_hw(0xFFFFFFFFu, p, n) ^ 0xFFFFFFFFu;

    const size_t blk = (n / 3) & ~(size_t)7;       /* whole 8-byte words */
    uint32_t c0 = 0xFFFFFFFFu, c1 = 0xFFFFFFFFu, c2 = 0xFFFFFFFFu;
    const uint8_t *p0 = p, *p1 = p + blk, *p2 = p + 2 * blk;
    for (size_t k = 0; k < blk; k += 8) {
        uint64_t v0, v1, v2;
        memcpy(&v0, p0 + k, 8);
        memcpy(&v1, p1 + k, 8);
        memcpy(&v2, p2 + k, 8);
        c0 = __crc32d(c0, v0);
        c1 = __crc32d(c1, v1);
        c2 = __crc32d(c2, v2);
    }
    uint32_t c = crc_combine(c0 ^ 0xFFFFFFFFu, c1 ^ 0xFFFFFFFFu, blk);
    c = crc_combine(c, c2 ^ 0xFFFFFFFFu, blk);

    const size_t tail = n - 3 * blk;
    if (tail)
        c = crc_combine(c, crc_hw(0xFFFFFFFFu, p + 3 * blk, tail) ^ 0xFFFFFFFFu,
                        tail);
    return c;
}
#endif /* WASTE_CRC_ARMV8 */

/* ---- entry point -------------------------------------------------------- */

uint32_t waste_crc32(const void *data, size_t n)
{
#ifdef WASTE_CRC_ARMV8
    return crc_hw3((const uint8_t *)data, n);
#else
    return crc_slice8(0xFFFFFFFFu, (const uint8_t *)data, n) ^ 0xFFFFFFFFu;
#endif
}

const char *waste_crc32_impl(void)
{
#ifdef WASTE_CRC_ARMV8
    return "armv8";
#else
    return "slice8";
#endif
}
