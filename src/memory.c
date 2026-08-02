/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* waste_physical_ram() answers "how big is this machine". On Linux that is
 * sysconf(_SC_PHYS_PAGES), which reads the host's MemTotal even when the
 * process is confined to a fraction of it, so a container gets the wrong
 * number by the whole ratio between the host and its limit. The automatic
 * budget then sizes against RAM it will never be allowed to touch — a
 * 32 GiB cgroup on a 256 GiB host asks for floor + 3x and is killed. That
 * is not the paging cliff of docs/LEARNED.md §16: nothing degrades, the
 * kernel simply kills the process, and no cache policy can soften it.
 *
 * So the ceiling becomes min(physical, cgroup limit) and everything
 * downstream is unchanged. What enters here is capacity only — the same
 * kind of number as physical RAM, fixed for the life of the process.
 * Current pressure (MemAvailable, memory.current) deliberately does not:
 * it moves between the read and the allocation, and a budget resolved once
 * at open cannot track it. Bounding a long-lived allocation by an
 * instantaneous reading makes the same command on the same machine two
 * different runs.
 *
 * memory.high counts as a limit alongside memory.max. It is the reclaim
 * threshold rather than a kill threshold, which makes it the closer
 * analogue of §16: a group allowed to exceed it is a group whose expert
 * cache the kernel takes back, and a hit that costs a page fault is worse
 * than the pread it replaced.
 */

#include "memory.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* There is no #if __linux__ in this file, on purpose. Nothing below is a
 * Linux API — it is fopen and strtoull over paths the caller supplies — so
 * guarding it would only mean the one platform that runs it is the one
 * platform that never compiles it anywhere else. §27 is what that costs.
 * On a host with no cgroups the two paths do not open and the answer is 0,
 * which is the same answer a #else would have returned, for the price of
 * two failed fopen() calls once per waste_open. */

/* A limit file holds a byte count or the literal "max". Unlimited,
 * unreadable and malformed all mean the same thing to a caller looking for
 * a ceiling — nothing constrains us here — so all three answer 0 and there
 * is no tri-state to get wrong at the call site. */
static uint64_t read_limit(const char *dir, const char *leaf)
{
    char path[1600];
    const int n = snprintf(path, sizeof path, "%s/%s", dir, leaf);
    if (n < 0 || (size_t)n >= sizeof path) return 0;

    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char b[64];
    const int got = fgets(b, sizeof b, f) != NULL;
    fclose(f);
    if (!got) return 0;

    char *p = b, *end = NULL;
    while (*p == ' ' || *p == '\t') p++;
    errno = 0;
    const unsigned long long v = strtoull(p, &end, 10);
    if (errno || end == p) return 0;          /* "max", or anything else */
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    return *end ? 0 : (uint64_t)v;
}

static uint64_t smaller_nonzero(uint64_t a, uint64_t b)
{
    if (!a) return b;
    if (!b) return a;
    return a < b ? a : b;
}

static uint64_t limits_at(const char *dir)
{
    return smaller_nonzero(read_limit(dir, "memory.max"),
                           read_limit(dir, "memory.high"));
}

/* The unified hierarchy's line in /proc/self/cgroup, which is "0::" and an
 * absolute path. Rejects "..", because these paths are parameters and a
 * test must not be able to walk out of the root it supplied. */
static int unified_path(const char *path, char *out, size_t cap)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[1024];
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "0::", 3)) continue;
        const char *rel = line + 3;
        const size_t n = strcspn(rel, "\r\n");
        if (!n || rel[0] != '/' || n >= cap) break;
        int bad = 0;
        for (size_t i = 0; i + 1 < n; i++)
            if (rel[i] == '.' && rel[i + 1] == '.' &&
                (i == 0 || rel[i - 1] == '/') &&
                (i + 2 == n || rel[i + 2] == '/')) bad = 1;
        if (bad) break;
        memcpy(out, rel, n);
        out[n] = 0;
        found = 1;
        break;
    }
    fclose(f);
    return found;
}

uint64_t waste_cgroup_limit(const char *self_cgroup_path, const char *cgroup_root)
{
    if (!cgroup_root || !*cgroup_root) return 0;

    char root[1024];
    size_t root_n = strlen(cgroup_root);
    while (root_n > 1 && cgroup_root[root_n - 1] == '/') root_n--;
    if (root_n >= sizeof root) return 0;
    memcpy(root, cgroup_root, root_n);
    root[root_n] = 0;

    char dir[1600];
    char rel[512];
    int walked = 0;
    if (self_cgroup_path && unified_path(self_cgroup_path, rel, sizeof rel)) {
        const int n = snprintf(dir, sizeof dir, "%s%s", root,
                               strcmp(rel, "/") ? rel : "");
        walked = n > 0 && (size_t)n < sizeof dir;
    }
    if (!walked) snprintf(dir, sizeof dir, "%s", root);

    /* memory.max is hierarchical: a leaf saying "max" does not cancel a
     * finite parent, so keep the most restrictive ancestor rather than
     * trusting the leaf.
     *
     * The walk ends *on* the mounted root, and that last read is not a
     * formality — it is what covers `docker run` without a private cgroup
     * namespace, where /proc/self/cgroup reports the host path
     * (0::/system.slice/docker-<id>.scope) while /sys/fs/cgroup is mounted
     * as the container's own root. Every composed directory along the way
     * is then missing, and the limit that actually applies is sitting in
     * the root itself. Stopping one level short would read nothing and
     * report "unconfined" for the one deployment this exists for. An
     * unconfined host is still 0: a real unified root has no memory.max. */
    uint64_t cap = 0;
    for (;;) {
        cap = smaller_nonzero(cap, limits_at(dir));
        const size_t len = strlen(dir);
        if (len <= root_n) break;
        char *slash = strrchr(dir, '/');
        if (!slash || (size_t)(slash - dir) < root_n) break;
        if ((size_t)(slash - dir) == root_n) dir[root_n] = 0;
        else *slash = 0;
    }
    return cap;
}
