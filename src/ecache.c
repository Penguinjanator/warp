/* ecache.c — see ecache.h. */

#include "ecache.h"

#include <stdlib.h>
#include <string.h>

#define EC_SAMPLE 16      /* victims sampled per eviction (Redis-style)     */

static int32_t ec_key(int layer, int expert) { return (layer << 16) | expert; }

static uint32_t ec_hash(int32_t k)
{
    uint32_t x = (uint32_t)k;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

int waste_ecache_init(waste_ecache *c, size_t budget_bytes, size_t rec_bytes,
                      int policy)
{
    memset(c, 0, sizeof *c);
    c->rec_bytes = rec_bytes;
    c->budget_bytes = budget_bytes;
    c->policy = policy;
    c->rng = 0x9e3779b9u;
    if (!rec_bytes || budget_bytes < rec_bytes) return 0;   /* no cache */

    c->n_slots = (int)(budget_bytes / rec_bytes);
    int hs = 1;
    while (hs < c->n_slots * 2) hs <<= 1;
    c->hash_mask = hs - 1;

    c->slot = (waste_eslot *)calloc((size_t)c->n_slots, sizeof *c->slot);
    c->hash = (int32_t *)malloc((size_t)hs * sizeof *c->hash);
    if (!c->slot || !c->hash) { waste_ecache_free(c); return -1; }
    memset(c->hash, 0xff, (size_t)hs * sizeof *c->hash);   /* all -1 */

    for (int i = 0; i < c->n_slots; i++) {
        c->slot[i].key = -1;
        c->slot[i].data = (uint8_t *)malloc(rec_bytes);
        if (!c->slot[i].data) { waste_ecache_free(c); return -1; }
    }
    return 0;
}

void waste_ecache_free(waste_ecache *c)
{
    if (c->slot) {
        for (int i = 0; i < c->n_slots; i++) free(c->slot[i].data);
        free(c->slot);
    }
    free(c->hash);
    c->slot = NULL; c->hash = NULL; c->n_slots = 0;
}

static int ec_lookup(waste_ecache *c, int32_t key)
{
    uint32_t h = ec_hash(key) & (uint32_t)c->hash_mask;
    for (int probe = 0; probe <= c->hash_mask; probe++) {
        const int32_t si = c->hash[h];
        if (si < 0) return -1;
        if (c->slot[si].key == key) return si;
        h = (h + 1) & (uint32_t)c->hash_mask;
    }
    return -1;
}

static void ec_insert(waste_ecache *c, int32_t key, int slot)
{
    uint32_t h = ec_hash(key) & (uint32_t)c->hash_mask;
    while (c->hash[h] >= 0) h = (h + 1) & (uint32_t)c->hash_mask;
    c->hash[h] = slot;
}

/* Rebuilding is simpler and safe: open addressing with deletions needs
 * tombstones, and evictions are rare relative to lookups. */
static void ec_rehash(waste_ecache *c)
{
    memset(c->hash, 0xff, ((size_t)c->hash_mask + 1) * sizeof *c->hash);
    for (int i = 0; i < c->n_slots; i++)
        if (c->slot[i].key >= 0) ec_insert(c, c->slot[i].key, i);
}

static int ec_victim(waste_ecache *c)
{
    /* free slot first */
    for (int i = 0; i < c->n_slots; i++)
        if (c->slot[i].key < 0) return i;

    int best = -1;
    uint32_t best_h = 0;
    uint64_t best_l = 0;
    for (int s = 0; s < EC_SAMPLE; s++) {
        c->rng = c->rng * 1664525u + 1013904223u;
        const int i = (int)(c->rng % (uint32_t)c->n_slots);
        const waste_eslot *sl = &c->slot[i];
        int better;
        if (c->policy == 1)                       /* LRU */
            better = (best < 0) || sl->last < best_l;
        else                                      /* LFRU */
            better = (best < 0) || sl->hits < best_h ||
                     (sl->hits == best_h && sl->last < best_l);
        if (better) { best = i; best_h = sl->hits; best_l = sl->last; }
    }
    return best;
}

const uint8_t *waste_ecache_get(waste_ecache *c, int layer, int expert,
                                waste_fetch_fn fetch, void *user)
{
    const int32_t key = ec_key(layer, expert);
    c->clock++;

    if (c->n_slots > 0) {
        const int si = ec_lookup(c, key);
        if (si >= 0) {
            c->hits++;
            c->slot[si].hits++;
            c->slot[si].last = c->clock;
            return c->slot[si].data;
        }
    }

    c->misses++;
    c->bytes_read += c->rec_bytes;

    if (c->n_slots == 0) return NULL;      /* caller falls back to its own buf */

    const int vi = ec_victim(c);
    const int had = c->slot[vi].key >= 0;
    if (fetch(user, layer, expert, c->slot[vi].data) != 0) {
        c->slot[vi].key = -1;
        if (had) { c->evictions++; ec_rehash(c); }
        return NULL;
    }
    if (had) { c->evictions++; c->slot[vi].key = key; ec_rehash(c); }
    else { c->slot[vi].key = key; ec_insert(c, key, vi); }
    c->slot[vi].hits = 1;
    c->slot[vi].last = c->clock;
    return c->slot[vi].data;
}
