/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* The cgroup reader against synthetic files. Every case here is one the
 * host running the suite cannot be put into: a machine is either in a
 * cgroup or it is not, and the two that matter — a leaf saying "max" under
 * a finite parent, and a /proc/self/cgroup path that does not exist under
 * the mounted root — need a hierarchy built to order.
 *
 * Because the reader takes its paths as parameters it is not Linux-only
 * code, so all of this runs on every host the suite does. That is the
 * point: the platform that would have been the only one compiling it is
 * the platform none of us develops on. */

#include "../src/memory.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* mkdir is the one call here that is not the same on Windows, where it
 * takes no mode. remove() rather than unlink() for the same reason: it is
 * in <stdio.h> on every host and needs no shim at all. */
#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define make_dir(p) mkdir((p), 0700)
#endif

#define GiB (1ULL << 30)
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

static char R[512], SELF[640], PARENT[640], CHILD[768];

static int put(const char *dir, const char *leaf, const char *text)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, leaf);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    return fputs(text, f) < 0 || fclose(f) ? -1 : 0;
}

static int rm(const char *dir, const char *leaf)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, leaf);
    return remove(path) && errno != ENOENT ? -1 : 0;
}

static int mk(const char *path)
{
    return make_dir(path) && errno != EEXIST ? -1 : 0;
}

static int self_says(const char *line)
{
    FILE *f = fopen(SELF, "w");
    if (!f) return -1;
    return fputs(line, f) < 0 || fclose(f) ? -1 : 0;
}

static int hierarchy_cases(const char *tmp)
{
    snprintf(R, sizeof R, "%s/cgroup", tmp);
    snprintf(SELF, sizeof SELF, "%s/self.cgroup", tmp);
    snprintf(PARENT, sizeof PARENT, "%s/parent", R);
    snprintf(CHILD, sizeof CHILD, "%s/child", PARENT);
    CHECK(!mk(R) && !mk(PARENT) && !mk(CHILD));

    /* Nothing anywhere is an unconfined host: a real unified root carries
     * no memory.max, so every read fails and nothing is claimed. */
    CHECK(!self_says("0::/parent/child\n"));
    CHECK(waste_cgroup_limit(SELF, R) == 0);

    /* The ordinary case: our own group is limited. */
    CHECK(!put(CHILD, "memory.max", "8589934592\n"));           /* 8 GiB */
    CHECK(waste_cgroup_limit(SELF, R) == 8 * GiB);

    /* "max" is not a limit, and it does not cancel a finite ancestor.
     * memory.max is hierarchical, so the parent still binds. */
    CHECK(!put(CHILD, "memory.max", "max\n"));
    CHECK(!put(PARENT, "memory.max", "17179869184\n"));         /* 16 GiB */
    CHECK(waste_cgroup_limit(SELF, R) == 16 * GiB);

    /* The most restrictive ancestor wins regardless of which one it is. */
    CHECK(!put(CHILD, "memory.max", "4294967296\n"));           /* 4 GiB */
    CHECK(waste_cgroup_limit(SELF, R) == 4 * GiB);

    /* memory.high counts too: a group the kernel reclaims from at 2 GiB
     * cannot hold a 4 GiB expert cache, whatever memory.max permits. */
    CHECK(!put(CHILD, "memory.high", "2147483648\n"));          /* 2 GiB */
    CHECK(waste_cgroup_limit(SELF, R) == 2 * GiB);
    CHECK(!rm(CHILD, "memory.high"));

    /* `docker run` without a private cgroup namespace: /proc/self/cgroup
     * reports the host path while the mount is the container's own root.
     * Nothing along the composed path exists; the limit is in the root.
     * Reading it is the whole point — this is the deployment the change
     * is for, and the version that stopped one level short reported
     * "unconfined" here. */
    CHECK(!rm(CHILD, "memory.max") && !rm(PARENT, "memory.max"));
    CHECK(!put(R, "memory.max", "34359738368\n"));              /* 32 GiB */
    CHECK(!self_says("0::/system.slice/docker-0123456789ab.scope\n"));
    CHECK(waste_cgroup_limit(SELF, R) == 32 * GiB);

    /* Same answer when /proc/self/cgroup cannot be read at all — cgroup v1,
     * or no /proc — because the mounted root is still the truth. */
    CHECK(waste_cgroup_limit("/nonexistent/self.cgroup", R) == 32 * GiB);
    CHECK(waste_cgroup_limit(NULL, R) == 32 * GiB);

    /* Malformed telemetry claims nothing rather than guessing. A limit
     * read wrong is a budget sized wrong, and 0 is the safe reading. */
    CHECK(!put(R, "memory.max", "not-a-number\n"));
    CHECK(waste_cgroup_limit(SELF, R) == 0);
    CHECK(!put(R, "memory.max", "\n"));
    CHECK(waste_cgroup_limit(SELF, R) == 0);
    CHECK(!put(R, "memory.max", "8589934592 trailing\n"));
    CHECK(waste_cgroup_limit(SELF, R) == 0);
    CHECK(!put(R, "memory.max", "34359738368\n"));

    /* The paths are parameters, so a path that walks out of the supplied
     * root is refused rather than followed. */
    CHECK(!self_says("0::/../outside\n"));
    CHECK(waste_cgroup_limit(SELF, R) == 32 * GiB);   /* fell back to root */
    CHECK(!self_says("garbage with no unified line\n"));
    CHECK(waste_cgroup_limit(SELF, R) == 32 * GiB);

    /* A trailing slash on the root is the same root. */
    char slashed[520];
    snprintf(slashed, sizeof slashed, "%s/", R);
    CHECK(!self_says("0::/parent/child\n"));
    CHECK(waste_cgroup_limit(SELF, slashed) == 32 * GiB);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: test_memory TMPDIR\n");
        return 2;
    }

    /* No root, no answer — on every platform. */
    CHECK(waste_cgroup_limit("/proc/self/cgroup", NULL) == 0);
    CHECK(waste_cgroup_limit("/proc/self/cgroup", "") == 0);

    if (hierarchy_cases(argv[1])) return 1;

    /* The production paths, whatever this host is. Either it is in a cgroup
     * with a limit, or nothing opens and the ceiling stays physical RAM;
     * both are correct and the check is that neither crashes nor invents. */
    {
        const uint64_t live = waste_cgroup_limit("/proc/self/cgroup",
                                                 "/sys/fs/cgroup");
        CHECK(live == 0 || live > (1 << 20));
    }

    puts("PASS cgroup limits");
    return 0;
}
