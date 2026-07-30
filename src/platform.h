/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * platform.h — the calls that are not POSIX everywhere.
 *
 * The engine is POSIX apart from six things, and they are all here rather
 * than spread through the sources as #ifdefs: a positional read, an
 * aligned allocation, the CPU count, a file's size, the physical RAM, and
 * opening a file with the page cache out of the way. Each has a Windows
 * implementation and a one-line POSIX one, so every call site reads the
 * same on all three targets. (Physical RAM is the exception that proves
 * it: macOS answers with sysctlbyname and Linux with sysconf, so that one
 * branch stays in waste.c and only the Windows half lives here.)
 *
 * Two things this fixes that are not "missing functions":
 *
 *   - `long` is 32 bits on Windows (LLP64), so every file offset here is
 *     int64_t. A container is 17 GB at the small end and ~900 GB for K3;
 *     an offset that silently truncates at 2 GB would build fine and read
 *     the wrong expert.
 *   - `_aligned_malloc` needs `_aligned_free`. Passing that pointer to
 *     free() is heap corruption on Windows, not a leak, so the allocation
 *     and its release are a pair and neither is used directly.
 */

#ifndef WASTE_PLATFORM_H
#define WASTE_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32

/* GetActiveProcessorCount and ALL_PROCESSOR_GROUPS are Windows 7. */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <fcntl.h>
#include <io.h>
#include <malloc.h>
#include <string.h>

/* Windows opens in text mode unless told otherwise, which turns 0x0D 0x0A
 * into 0x0A in the middle of weights. Every open of a container file ORs
 * this in; on POSIX it is 0. */
#define WASTE_O_BINARY _O_BINARY

/* One positional read. ReadFile with an OVERLAPPED offset does not touch
 * the shared file pointer, which is what pread() is for: the expert cache
 * reads records by offset and nothing wants a seek in between.
 *
 * Reads are capped at 1 GiB because the count is a DWORD. Callers loop
 * (see pread_all) and a short read is legal, so the cap is invisible. */
static inline int64_t waste_pread(int fd, void *dst, size_t n, int64_t off)
{
    const HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE || off < 0) return -1;
    if (n > (size_t)1 << 30) n = (size_t)1 << 30;

    OVERLAPPED ov;
    memset(&ov, 0, sizeof ov);
    ov.Offset     = (DWORD)((uint64_t)off & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD)((uint64_t)off >> 32);

    DWORD got = 0;
    if (!ReadFile(h, dst, (DWORD)n, &got, &ov)) {
        /* Reading at or past the end is not an error anywhere else. */
        return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    }
    return (int64_t)got;
}

static inline void *waste_aligned_alloc(size_t align, size_t n)
{
    return _aligned_malloc(n, align);
}

static inline void waste_aligned_free(void *p)
{
    _aligned_free(p);            /* NOT free() — see the header comment */
}

/* GetSystemInfo would answer for the current processor group only, so it
 * says 64 on a machine with more. The pool caps at 64 anyway, but the
 * number is also reported to the user. */
static inline int waste_cpu_count(void)
{
    const DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (n > 0) return (int)n;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}

static inline int64_t waste_file_size(int fd)
{
    const HANDLE h = (HANDLE)_get_osfhandle(fd);
    LARGE_INTEGER sz;
    if (h == INVALID_HANDLE_VALUE || !GetFileSizeEx(h, &sz)) return -1;
    return (int64_t)sz.QuadPart;
}

static inline uint64_t waste_physical_ram_bytes(void)
{
    MEMORYSTATUSEX st;
    st.dwLength = sizeof st;
    return GlobalMemoryStatusEx(&st) ? (uint64_t)st.ullTotalPhys : 0;
}

static inline int waste_sync_file(FILE *f)
{
    return fflush(f) || _commit(_fileno(f));
}

static inline int waste_replace_file(const char *src, const char *dst)
{
    return MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING |
                       MOVEFILE_WRITE_THROUGH) ? 0 : -1;
}

/* The Windows half of the page-cache bypass, which is the part of this
 * port that actually matters: the expert-streaming argument is that the
 * hit rate we measure is ours and not the kernel's.
 *
 * FILE_FLAG_NO_BUFFERING is the O_DIRECT of this platform and carries the
 * same contract — offset, length and destination address must all be
 * multiples of the volume's sector size. Records are whole 4 KiB pages and
 * buffers come from waste_aligned_alloc at 16 KiB, so both hold; the
 * caller checks the record size anyway rather than assuming, because a
 * misaligned read fails outright instead of merely running slow.
 *
 * Without the bypass, FILE_FLAG_RANDOM_ACCESS at least stops Windows
 * reading ahead into pages nothing will ask for — the same fallback as
 * posix_fadvise(POSIX_FADV_RANDOM) on Linux.
 *
 * The handle is adopted by the returned descriptor, so close(fd) closes
 * it and the rest of the engine needs no Windows branch. */
static inline int waste_open_stream(const char *path, int bypass)
{
    const DWORD flags = bypass ? FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS
                               : FILE_FLAG_RANDOM_ACCESS;
    const HANDLE h = CreateFileA(path, GENERIC_READ,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                 OPEN_EXISTING, flags, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const int fd = _open_osfhandle((intptr_t)h, _O_RDONLY | _O_BINARY);
    if (fd < 0) { CloseHandle(h); return -1; }
    return fd;
}

#else /* POSIX */

#include <stdlib.h>
#include <unistd.h>

#define WASTE_O_BINARY 0

static inline int64_t waste_pread(int fd, void *dst, size_t n, int64_t off)
{
    return (int64_t)pread(fd, dst, n, (off_t)off);
}

static inline void *waste_aligned_alloc(size_t align, size_t n)
{
    void *p = NULL;
    return posix_memalign(&p, align, n) == 0 ? p : NULL;
}

static inline void waste_aligned_free(void *p) { free(p); }

static inline int waste_cpu_count(void)
{
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 1 ? (int)n : 1;
}

static inline int64_t waste_file_size(int fd)
{
    const off_t n = lseek(fd, 0, SEEK_END);
    return n < 0 ? -1 : (int64_t)n;
}

static inline int waste_sync_file(FILE *f)
{
    return fflush(f) || fsync(fileno(f));
}

static inline int waste_replace_file(const char *src, const char *dst)
{
    return rename(src, dst);
}

#endif /* _WIN32 */

#endif /* WASTE_PLATFORM_H */
