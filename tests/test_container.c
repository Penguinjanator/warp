/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_container.c — read a WASTE expert bank with the real C structs.
 *
 * The converter writes records from Python; this proves the bytes it emits
 * are what src/waste_format.h describes — header size, field order, offsets
 * and 4 KiB alignment — and that one pread() per record is enough.
 *
 * It also checks the engine's crc32 against known values of zlib's, which
 * is the function that wrote every crc32 field in every container. The two
 * agreeing is not a detail: the engine now refuses a record whose checksum
 * does not match, so an implementation that were merely self-consistent
 * would reject every container ever built.
 *
 *   cc -O2 -o test_container tests/test_container.c src/crc32.c
 *   ./test_container path/model.waste/experts-L1.bin [n_records]
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/crc32.h"
#include "../src/platform.h"
#include "../src/waste_format.h"

/* Values from zlib: `python3 -c "import zlib; print(zlib.crc32(b'...'))"`.
 * The long ones matter because the ARM path splits a buffer in three and
 * stitches the parts back together in GF(2) above 3072 bytes — a length
 * below that never reaches the code most likely to be wrong. */
static int crc_selftest(void)
{
    static const struct { const char *s; uint32_t want; } vec[] = {
        { "",           0x00000000u },
        { "a",          0xE8B7BE43u },
        { "123456789",  0xCBF43926u },
        { "The quick brown fox jumps over the lazy dog", 0x414FA339u },
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof vec / sizeof *vec; i++) {
        const uint32_t got = waste_crc32(vec[i].s, strlen(vec[i].s));
        if (got != vec[i].want) {
            printf("  crc32(\"%s\") = %08x, zlib says %08x\n",
                   vec[i].s, got, vec[i].want);
            bad++;
        }
    }

    /* 64 KiB of a fixed pattern, checked whole and at four offsets into it,
     * so the split path, its tail and an unaligned start are all covered. */
    static uint8_t buf[65536];
    for (size_t i = 0; i < sizeof buf; i++) buf[i] = (uint8_t)(i * 31 + (i >> 11));
    static const struct { size_t off, len; uint32_t want; } big[] = {
        { 0, 65536, 0x857C6B4Fu }, { 0,  3071, 0xE08F605Eu },
        { 0,  3073, 0x331C420Au }, { 3, 40001, 0x5BBC9F51u },
        { 7, 65529, 0xB1F5AE9Au },
    };
    for (size_t i = 0; i < sizeof big / sizeof *big; i++) {
        const uint32_t got = waste_crc32(buf + big[i].off, big[i].len);
        if (got != big[i].want) {
            printf("  crc32(pattern+%zu, %zu) = %08x, zlib says %08x\n",
                   big[i].off, big[i].len, got, big[i].want);
            bad++;
        }
    }
    printf("crc32 (%s): %s zlib\n", waste_crc32_impl(),
           bad ? "DISAGREES with" : "agrees with");
    return bad;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s experts-LN.bin [n]\n", argv[0]); return 2; }
    const int want = argc > 2 ? atoi(argv[2]) : 4;

    /* Plain open, not the engine's bypass: this checks the bytes, and a
     * bypassed read would impose its own alignment rules on the header
     * read below. fstat is avoided because struct stat carries a 32-bit
     * size on Windows. */
    int fd = open(argv[1], O_RDONLY | WASTE_O_BINARY);
    if (fd < 0) { perror("open"); return 1; }
#ifdef __APPLE__
    fcntl(fd, F_NOCACHE, 1);            /* the engine's real read path */
#endif
    const int64_t fsize = waste_file_size(fd);
    if (fsize < 0) { perror("size"); return 1; }

    printf("file %s (%.1f MB)\n", argv[1], fsize / 1048576.0);
    printf("sizeof(waste_expert_hdr) = %zu\n", sizeof(waste_expert_hdr));

    int64_t off = 0;
    int n = 0, bad = crc_selftest();
    printf("\n");
    while (off < fsize && n < want) {
        waste_expert_hdr h;
        if (waste_pread(fd, &h, sizeof h, off) != (int64_t)sizeof h) { perror("pread"); return 1; }

        if (h.magic != WASTE_MAGIC_EXPERT) {
            printf("record %d at %lld: BAD MAGIC %08x\n", n, (long long)off, h.magic);
            bad++; break;
        }
        if (off % WASTE_ALIGN) { printf("  not 4 KiB aligned!\n"); bad++; }
        if (h.lowrank_id != 0) { printf("  lowrank_id != 0 (v0 violation)\n"); bad++; }
        if (h.fmt != WQ_VQ3R && h.fmt != WQ_VQ2R) { printf("  unexpected fmt %u\n", h.fmt); bad++; }

        /* the whole expert in ONE read — the point of the layout */
        const size_t bytes = (size_t)h.rec_4k_blocks * WASTE_ALIGN;
        void *buf = waste_aligned_alloc(WASTE_ALIGN, bytes);
        if (!buf) { fprintf(stderr, "oom\n"); return 1; }
        if (waste_pread(fd, buf, bytes, off) != (int64_t)bytes) { perror("pread rec"); return 1; }

        printf("record %d: layer %u expert %-3u fmt %u cb %u  %u blocks (%.2f MB)\n",
               n, h.layer, h.expert_id, h.fmt, h.codebook_id, h.rec_4k_blocks,
               bytes / 1048576.0);
        printf("          gate@%u up@%u down@%u corr@%u  crc %08x\n",
               h.gate_off, h.up_off, h.down_off, h.chan_corr_off, h.crc32);

        if (!(h.gate_off < h.up_off && h.up_off < h.down_off &&
              h.down_off < h.chan_corr_off && h.chan_corr_off < bytes)) {
            printf("          OFFSETS OUT OF ORDER/RANGE\n"); bad++;
        }
        waste_aligned_free(buf);        /* _aligned_malloc needs its own free */
        off += (int64_t)bytes;
        n++;
    }
    close(fd);
    printf("\n%d records read, %d problems\n", n, bad);
    return bad ? 1 : 0;
}
