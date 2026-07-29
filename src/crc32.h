/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * crc32.h — CRC-32 (IEEE 802.3), the one an expert record header carries.
 *
 * Bit-identical to zlib's crc32(), because that is what writes the value:
 * tools/convert.py calls zlib.crc32 over the record body and the engine's
 * job is to agree with the converter, not merely with itself.
 */

#ifndef WASTE_CRC32_H
#define WASTE_CRC32_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t waste_crc32(const void *data, size_t n);

/* Which implementation this build got — "armv8" or "slice8". Reported by
 * waste_build_info, because the two differ by 12x and the choice is made
 * by compiler flags rather than by anything visible at run time. */
const char *waste_crc32_impl(void);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_CRC32_H */
