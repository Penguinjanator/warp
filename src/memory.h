/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* How much RAM this process may use, which is not how big the machine is.
 * Private: the public answer is waste_usable_ram() in waste.h. */

#ifndef WASTE_MEMORY_H
#define WASTE_MEMORY_H

#include <stdint.h>

/* The smallest finite cgroup-v2 memory limit applying to this process, or 0
 * when none does — no cgroup, an unlimited one, cgroup v1, or a host that
 * has no such thing. Both paths are parameters so the policy can be checked
 * against synthetic files: the cases worth testing are a leaf that says
 * "max" under a finite parent, and a /proc/self/cgroup path that does not
 * exist under the mounted root, and neither can be arranged on the host
 * running the suite. That also makes this reader platform-independent, so
 * it is compiled and exercised everywhere rather than only where it fires.
 * Production passes "/proc/self/cgroup" and "/sys/fs/cgroup". */
uint64_t waste_cgroup_limit(const char *self_cgroup_path,
                            const char *cgroup_root);

#endif /* WASTE_MEMORY_H */
