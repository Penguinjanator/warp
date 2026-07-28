/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* tokenizer.c — see tokenizer.h. */

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const uint8_t *p; int len; int rank; } tok_entry;

struct waste_tok {
    uint8_t *blob;          /* decoded token bytes, back to back           */
    tok_entry *by_rank;     /* rank -> bytes                                */
    int n_tokens;
    int32_t *hash;          /* open addressing: -> index into by_rank       */
    int hash_mask;
    int bos, eos;
};

/* ---- base64 ------------------------------------------------------------ */

static int b64val(int c)
{
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static int b64decode(const char *s, int len, uint8_t *out)
{
    /* unsigned: the accumulator shifts past bit 31 on long lines, and signed
     * overflow there is UB the optimizer is entitled to turn into a trap */
    int n = 0, bits = 0;
    uint32_t acc = 0;
    for (int i = 0; i < len; i++) {
        if (s[i] == '=') break;
        const int v = b64val((unsigned char)s[i]);
        if (v < 0) continue;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out[n++] = (uint8_t)((acc >> bits) & 0xff); }
    }
    return n;
}

/* ---- hash over token bytes --------------------------------------------- */

static uint32_t bhash(const uint8_t *p, int len)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static void tok_index(waste_tok *t, int i)
{
    uint32_t h = bhash(t->by_rank[i].p, t->by_rank[i].len) & (uint32_t)t->hash_mask;
    while (t->hash[h] >= 0) h = (h + 1) & (uint32_t)t->hash_mask;
    t->hash[h] = i;
}

/* rank of a byte string, or -1 */
static int tok_rank(const waste_tok *t, const uint8_t *p, int len)
{
    uint32_t h = bhash(p, len) & (uint32_t)t->hash_mask;
    for (int probe = 0; probe <= t->hash_mask; probe++) {
        const int32_t i = t->hash[h];
        if (i < 0) return -1;
        if (t->by_rank[i].len == len && memcmp(t->by_rank[i].p, p, (size_t)len) == 0)
            return t->by_rank[i].rank;
        h = (h + 1) & (uint32_t)t->hash_mask;
    }
    return -1;
}

/* ---- loading ------------------------------------------------------------ */

waste_tok *waste_tok_open(const char *dir)
{
    char path[512];
    FILE *f = NULL;
    const char *names[] = { "tokenizer.model", "tiktoken.model" };
    for (int i = 0; i < 2 && !f; i++) {
        snprintf(path, sizeof path, "%s/%s", dir, names[i]);
        f = fopen(path, "rb");
    }
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *raw = (char *)malloc((size_t)sz + 1);
    if (!raw || fread(raw, 1, (size_t)sz, f) != (size_t)sz) { free(raw); fclose(f); return NULL; }
    raw[sz] = 0;
    fclose(f);

    waste_tok *t = (waste_tok *)calloc(1, sizeof *t);
    t->blob = (uint8_t *)malloc((size_t)sz);          /* decoded is smaller */
    t->by_rank = (tok_entry *)malloc(sizeof(tok_entry) * 300000);
    if (!t->blob || !t->by_rank) { free(raw); waste_tok_free(t); return NULL; }

    size_t used = 0;
    char *line = raw, *end = raw + sz;
    while (line < end) {
        char *nl = memchr(line, '\n', (size_t)(end - line));
        if (!nl) nl = end;
        char *sp = memchr(line, ' ', (size_t)(nl - line));
        if (sp) {
            const int nb = b64decode(line, (int)(sp - line), t->blob + used);
            const int rank = atoi(sp + 1);
            t->by_rank[t->n_tokens].p = t->blob + used;
            t->by_rank[t->n_tokens].len = nb;
            t->by_rank[t->n_tokens].rank = rank;
            t->n_tokens++;
            used += (size_t)nb;
        }
        line = nl + 1;
    }
    free(raw);

    int hs = 1;
    while (hs < t->n_tokens * 2) hs <<= 1;
    t->hash_mask = hs - 1;
    t->hash = (int32_t *)malloc((size_t)hs * sizeof(int32_t));
    if (!t->hash) { waste_tok_free(t); return NULL; }
    memset(t->hash, 0xff, (size_t)hs * sizeof(int32_t));
    for (int i = 0; i < t->n_tokens; i++) tok_index(t, i);

    /* Kimi's reserved block starts right after the base vocabulary:
     * [BOS] = base, [EOS] = base+1, <|im_end|> = base+2. */
    t->bos = t->n_tokens;
    t->eos = t->n_tokens + 2;
    return t;
}

void waste_tok_free(waste_tok *t)
{
    if (!t) return;
    free(t->blob); free(t->by_rank); free(t->hash); free(t);
}

int waste_tok_vocab(const waste_tok *t) { return t->n_tokens; }
int waste_tok_bos(const waste_tok *t) { return t->bos; }
int waste_tok_eos(const waste_tok *t) { return t->eos; }

/* ---- UTF-8 + the character classes the pattern needs -------------------- */

static int utf8_next(const char *s, int len, int *cp)
{
    const unsigned char *u = (const unsigned char *)s;
    if (len <= 0) { *cp = -1; return 0; }
    if (u[0] < 0x80) { *cp = u[0]; return 1; }
    if ((u[0] & 0xe0) == 0xc0 && len >= 2) { *cp = ((u[0] & 0x1f) << 6) | (u[1] & 0x3f); return 2; }
    if ((u[0] & 0xf0) == 0xe0 && len >= 3) {
        *cp = ((u[0] & 0x0f) << 12) | ((u[1] & 0x3f) << 6) | (u[2] & 0x3f); return 3;
    }
    if ((u[0] & 0xf8) == 0xf0 && len >= 4) {
        *cp = ((u[0] & 0x07) << 18) | ((u[1] & 0x3f) << 12) | ((u[2] & 0x3f) << 6) | (u[3] & 0x3f);
        return 4;
    }
    *cp = u[0];
    return 1;
}

static int is_han(int c)
{
    return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
           (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x20000 && c <= 0x2FA1F);
}

/* \p{L} plus \p{M}: ASCII and Latin-1 letters, Latin Extended, Greek,
 * Cyrillic, Hebrew, Arabic, Hangul, Hiragana/Katakana, combining marks.
 * Han is excluded on purpose — the pattern gives it its own branch. */
static int is_letter(int c)
{
    if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    if (is_han(c)) return 0;
    return (c >= 0xC0 && c <= 0x24F) || (c >= 0x300 && c <= 0x36F) ||
           (c >= 0x370 && c <= 0x3FF) || (c >= 0x400 && c <= 0x52F) ||
           (c >= 0x590 && c <= 0x6FF) || (c >= 0x3040 && c <= 0x30FF) ||
           (c >= 0xAC00 && c <= 0xD7AF) || (c >= 0x1E00 && c <= 0x1EFF);
}

__attribute__((unused)) static int is_upper(int c)
{
    if (c < 0x80) return c >= 'A' && c <= 'Z';
    return (c >= 0xC0 && c <= 0xDE) || (c >= 0x391 && c <= 0x3A9) ||
           (c >= 0x410 && c <= 0x42F);
}

static int is_digit(int c) { return c >= '0' && c <= '9'; }
static int is_space(int c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' ||
                                    c == '\f' || c == '\v' || c == 0xA0; }
static int is_nl(int c) { return c == '\r' || c == '\n'; }

/* Advances one pre-token, returning its byte length. Mirrors the branch
 * order of the model's pat_str. */
static int next_piece(const char *s, int len)
{
    int cp, n = utf8_next(s, len, &cp), i;
    if (n == 0) return 0;

    if (is_han(cp)) {                                   /* [\p{Han}]+ */
        i = n;
        while (i < len) { int c2, k = utf8_next(s + i, len - i, &c2);
                          if (!is_han(c2)) break; i += k; }
        return i;
    }

    /* (?i:'s|'t|'re|'ve|'m|'ll|'d) — both letter branches take it */
    #define SUFFIX_AT(i) do { \
        if ((i) < len && s[i] == '\'') { \
            static const char *sfx[] = { "s", "t", "re", "ve", "m", "ll", "d" }; \
            for (size_t q = 0; q < sizeof sfx / sizeof *sfx; q++) { \
                const int sl = (int)strlen(sfx[q]); \
                if ((i) + 1 + sl <= len && \
                    strncasecmp(s + (i) + 1, sfx[q], (size_t)sl) == 0) { \
                    (i) += 1 + sl; break; \
                } \
            } \
        } \
    } while (0)

    /* optional leading non-letter/non-digit, then a letter run */
    if (!is_letter(cp) && !is_digit(cp) && !is_nl(cp)) {
        int c2, k2, j = n;
        k2 = utf8_next(s + j, len - j, &c2);
        if (k2 && is_letter(c2)) {
            i = j;
            while (i < len) { int c3, k = utf8_next(s + i, len - i, &c3);
                              if (!is_letter(c3)) break; i += k; }
            SUFFIX_AT(i);
            return i;
        }
    }
    if (is_letter(cp)) {
        i = n;
        while (i < len) { int c3, k = utf8_next(s + i, len - i, &c3);
                          if (!is_letter(c3)) break; i += k; }
        SUFFIX_AT(i);
        return i;
    }

    if (is_digit(cp)) {                                 /* \p{N}{1,3} */
        i = n;
        int cnt = 1;
        while (i < len && cnt < 3) { int c3, k = utf8_next(s + i, len - i, &c3);
                                     if (!is_digit(c3)) break; i += k; cnt++; }
        return i;
    }

    /* " ?[^\s\p{L}\p{N}]+[\r\n]*" */
    {
        int j = 0;
        if (cp == ' ') {
            int c2, k2 = utf8_next(s + n, len - n, &c2);
            if (k2 && !is_space(c2) && !is_letter(c2) && !is_digit(c2)) j = n;
        }
        int save = j;
        i = j;
        while (i < len) { int c3, k = utf8_next(s + i, len - i, &c3);
                          if (is_space(c3) || is_letter(c3) || is_digit(c3)) break; i += k; }
        if (i > save) {
            while (i < len && is_nl(s[i])) i++;
            return i;
        }
    }

    /* "\s*[\r\n]+" then "\s+(?!\S)" then "\s+" */
    if (is_space(cp)) {
        i = 0;
        int last_nl = -1;
        while (i < len) { int c3, k = utf8_next(s + i, len - i, &c3);
                          if (!is_space(c3)) break;
                          i += k;
                          if (is_nl(c3)) last_nl = i; }
        if (last_nl > 0) return last_nl;
        /* \s+(?!\S): keep all but the last space when more text follows */
        if (i < len && i > 1) return i - 1;
        return i;
    }
    return n;
}

/* ---- byte-pair merge ---------------------------------------------------- */

#define MAXPIECE 256

static int encode_piece(const waste_tok *t, const uint8_t *p, int len,
                        int32_t *out, int cap)
{
    if (len == 0) return 0;
    const int r = tok_rank(t, p, len);
    if (r >= 0) { if (cap < 1) return -1; out[0] = r; return 1; }
    if (len > MAXPIECE) len = MAXPIECE;

    int start[MAXPIECE + 1], rank[MAXPIECE + 1], n = 0;
    for (int i = 0; i < len; i++) { start[n] = i; rank[n] = INT32_MAX; n++; }
    start[n] = len; rank[n] = INT32_MAX;

    for (int i = 0; i + 1 < n; i++) {
        const int rr = tok_rank(t, p + start[i], start[i + 2] - start[i]);
        rank[i] = rr < 0 ? INT32_MAX : rr;
    }

    for (;;) {
        int best = -1, bestr = INT32_MAX;
        for (int i = 0; i + 1 < n; i++)
            if (rank[i] < bestr) { bestr = rank[i]; best = i; }
        if (best < 0) break;

        /* merge best with best+1 */
        for (int i = best + 1; i < n; i++) { start[i] = start[i + 1]; rank[i] = rank[i + 1]; }
        n--;
        for (int i = (best > 0 ? best - 1 : 0); i <= best && i + 1 < n; i++) {
            const int rr = tok_rank(t, p + start[i], start[i + 2] - start[i]);
            rank[i] = rr < 0 ? INT32_MAX : rr;
        }
    }

    if (n > cap) return -1;
    for (int i = 0; i < n; i++) {
        const int rr = tok_rank(t, p + start[i], start[i + 1] - start[i]);
        out[i] = rr < 0 ? 0 : rr;
    }
    return n;
}

int waste_tok_encode(const waste_tok *t, const char *text, int32_t *out, int cap)
{
    int n = 0, pos = 0;
    const int len = (int)strlen(text);
    while (pos < len) {
        const int plen = next_piece(text + pos, len - pos);
        if (plen <= 0) break;
        const int got = encode_piece(t, (const uint8_t *)text + pos, plen,
                                     out + n, cap - n);
        if (got < 0) return -1;
        n += got;
        pos += plen;
    }
    return n;
}

int waste_tok_decode1(const waste_tok *t, int32_t id, char *buf, int cap)
{
    for (int i = 0; i < t->n_tokens; i++)
        if (t->by_rank[i].rank == id) {
            const int l = t->by_rank[i].len < cap ? t->by_rank[i].len : cap;
            memcpy(buf, t->by_rank[i].p, (size_t)l);
            return l;
        }
    return 0;
}

int waste_tok_decode(const waste_tok *t, const int32_t *ids, int n,
                     char *buf, int cap)
{
    int w = 0;
    for (int i = 0; i < n && w < cap; i++)
        w += waste_tok_decode1(t, ids[i], buf + w, cap - w);
    return w;
}
