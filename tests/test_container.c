/*
 * test_container.c — read a WASTE expert bank with the real C structs.
 *
 * The converter writes records from Python; this proves the bytes it emits
 * are what src/waste_format.h describes — header size, field order, offsets
 * and 4 KiB alignment — and that one pread() per record is enough.
 *
 *   cc -O2 -o test_container tests/test_container.c
 *   ./test_container path/model.waste/experts-L1.bin [n_records]
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../src/waste_format.h"

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s experts-LN.bin [n]\n", argv[0]); return 2; }
    const int want = argc > 2 ? atoi(argv[2]) : 4;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
#ifdef __APPLE__
    fcntl(fd, F_NOCACHE, 1);            /* the engine's real read path */
#endif
    struct stat st;
    if (fstat(fd, &st)) { perror("fstat"); return 1; }

    printf("file %s (%.1f MB)\n", argv[1], st.st_size / 1048576.0);
    printf("sizeof(waste_expert_hdr) = %zu\n\n", sizeof(waste_expert_hdr));

    off_t off = 0;
    int n = 0, bad = 0;
    while (off < st.st_size && n < want) {
        waste_expert_hdr h;
        if (pread(fd, &h, sizeof h, off) != (ssize_t)sizeof h) { perror("pread"); return 1; }

        if (h.magic != WASTE_MAGIC_EXPERT) {
            printf("record %d at %lld: BAD MAGIC %08x\n", n, (long long)off, h.magic);
            bad++; break;
        }
        if (off % WASTE_ALIGN) { printf("  not 4 KiB aligned!\n"); bad++; }
        if (h.lowrank_id != 0) { printf("  lowrank_id != 0 (v0 violation)\n"); bad++; }
        if (h.fmt != WQ_VQ3R && h.fmt != WQ_VQ2R) { printf("  unexpected fmt %u\n", h.fmt); bad++; }

        /* the whole expert in ONE read — the point of the layout */
        const size_t bytes = (size_t)h.rec_4k_blocks * WASTE_ALIGN;
        void *buf = NULL;
        if (posix_memalign(&buf, WASTE_ALIGN, bytes)) { fprintf(stderr, "oom\n"); return 1; }
        if (pread(fd, buf, bytes, off) != (ssize_t)bytes) { perror("pread rec"); return 1; }

        printf("record %d: layer %u expert %-3u fmt %u cb %u  %u blocks (%.2f MB)\n",
               n, h.layer, h.expert_id, h.fmt, h.codebook_id, h.rec_4k_blocks,
               bytes / 1048576.0);
        printf("          gate@%u up@%u down@%u corr@%u  crc %08x\n",
               h.gate_off, h.up_off, h.down_off, h.chan_corr_off, h.crc32);

        if (!(h.gate_off < h.up_off && h.up_off < h.down_off &&
              h.down_off < h.chan_corr_off && h.chan_corr_off < bytes)) {
            printf("          OFFSETS OUT OF ORDER/RANGE\n"); bad++;
        }
        free(buf);
        off += (off_t)bytes;
        n++;
    }
    close(fd);
    printf("\n%d records read, %d problems\n", n, bad);
    return bad ? 1 : 0;
}
