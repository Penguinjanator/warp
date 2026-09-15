/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * json.h — minimal, allocation-light JSON reader for the manifest.
 *
 * Single pass into a flat token array; values are located by walking that
 * array. Enough for manifest.json (objects, arrays, strings, numbers,
 * true/false/null) and nothing more. Treats input as untrusted: every
 * accessor is bounds-checked and depth is capped.
 *
 * Header-only, C11, no dependencies.
 */

#ifndef WASTE_JSON_H
#define WASTE_JSON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef enum { JS_OBJ, JS_ARR, JS_STR, JS_NUM, JS_BOOL, JS_NULL } js_type;

typedef struct {
    js_type type;
    int start, end;      /* byte range in the source                        */
    int size;            /* members (obj) / elements (arr)                  */
    int next;            /* index just past this value's subtree            */
} js_tok;

typedef struct {
    const char *src;
    js_tok *tok;
    int n, cap;
} js_doc;

#define JS_MAX_DEPTH 64

static int js__parse(js_doc *d, int pos, int depth);

static int js__push(js_doc *d, js_type t, int start)
{
    if (d->n == d->cap) {
        int c = d->cap ? d->cap * 2 : 256;
        js_tok *p = (js_tok *)realloc(d->tok, (size_t)c * sizeof *p);
        if (!p) return -1;
        d->tok = p; d->cap = c;
    }
    d->tok[d->n].type = t;
    d->tok[d->n].start = start;
    d->tok[d->n].end = start;
    d->tok[d->n].size = 0;
    d->tok[d->n].next = -1;
    return d->n++;
}

static int js__ws(const char *s, int p)
{
    while (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r') p++;
    return p;
}

static int js__str(js_doc *d, int pos)
{
    const char *s = d->src;
    int t = js__push(d, JS_STR, pos + 1);
    if (t < 0) return -1;
    int p = pos + 1;
    while (s[p] && s[p] != '"') {
        if (s[p] == '\\' && s[p + 1]) p++;
        p++;
    }
    if (s[p] != '"') return -1;
    d->tok[t].end = p;
    return p + 1;
}

static int js__parse(js_doc *d, int pos, int depth)
{
    if (depth > JS_MAX_DEPTH) return -1;
    const char *s = d->src;
    pos = js__ws(s, pos);
    char c = s[pos];

    if (c == '"') return js__str(d, pos);

    if (c == '{' || c == '[') {
        const int is_obj = (c == '{');
        int t = js__push(d, is_obj ? JS_OBJ : JS_ARR, pos);
        if (t < 0) return -1;
        int p = js__ws(s, pos + 1);
        if (s[p] == (is_obj ? '}' : ']')) { d->tok[t].end = p; d->tok[t].next = d->n; return p + 1; }
        for (;;) {
            if (is_obj) {
                p = js__ws(s, p);
                if (s[p] != '"') return -1;
                p = js__str(d, p);
                if (p < 0) return -1;
                p = js__ws(s, p);
                if (s[p] != ':') return -1;
                p++;
            }
            p = js__parse(d, p, depth + 1);
            if (p < 0) return -1;
            d->tok[t].size++;
            p = js__ws(s, p);
            if (s[p] == ',') { p++; continue; }
            if (s[p] == (is_obj ? '}' : ']')) { d->tok[t].end = p; p++; break; }
            return -1;
        }
        d->tok[t].next = d->n;
        return p;
    }

    /* number / true / false / null */
    int t = js__push(d, c == 't' || c == 'f' ? JS_BOOL : (c == 'n' ? JS_NULL : JS_NUM), pos);
    if (t < 0) return -1;
    int p = pos;
    while (s[p] && !strchr(",]} \t\n\r", s[p])) p++;
    d->tok[t].end = p;
    d->tok[t].next = d->n;
    return p;
}

/* Parse a NUL-terminated buffer. Returns 0 on success; caller frees with
 * js_free(). `src` must outlive the doc. */
static inline int js_parse(js_doc *d, const char *src)
{
    memset(d, 0, sizeof *d);
    d->src = src;
    return js__parse(d, 0, 0) < 0 ? -1 : 0;
}

static inline void js_free(js_doc *d) { free(d->tok); d->tok = NULL; d->n = d->cap = 0; }

/* Index just past the subtree rooted at t. */
static inline int js_skip(const js_doc *d, int t)
{
    if (t < 0 || t >= d->n) return d->n;
    return d->tok[t].next >= 0 ? d->tok[t].next : t + 1;
}

static inline int js_streq(const js_doc *d, int t, const char *k)
{
    if (t < 0 || t >= d->n || d->tok[t].type != JS_STR) return 0;
    const int len = d->tok[t].end - d->tok[t].start;
    return (int)strlen(k) == len && strncmp(d->src + d->tok[t].start, k, (size_t)len) == 0;
}

/* Member `key` of object `obj`, or -1. */
static inline int js_get(const js_doc *d, int obj, const char *key)
{
    if (obj < 0 || obj >= d->n || d->tok[obj].type != JS_OBJ) return -1;
    int p = obj + 1;
    for (int i = 0; i < d->tok[obj].size; i++) {
        const int k = p, v = p + 1;
        if (js_streq(d, k, key)) return v;
        p = js_skip(d, v);
    }
    return -1;
}

/* Members of an object or elements of an array, and 0 for anything else.
 *
 * js_get and js_at return -1 when a key is missing or a token is not the
 * container the caller assumed. Reading d->tok[-1].size then indexes
 * before the allocation — a heap read a malformed manifest can reach, and
 * did: three call sites had it before a fuzzer found them. Use this
 * instead of touching d->tok directly. */
static inline int js_size(const js_doc *d, int tok)
{
    if (tok < 0 || tok >= d->n) return 0;
    if (d->tok[tok].type != JS_OBJ && d->tok[tok].type != JS_ARR) return 0;
    return d->tok[tok].size;
}

/* Element `i` of array `arr`, or -1. */
static inline int js_at(const js_doc *d, int arr, int i)
{
    /* i < 0 used to fall through the loop and return the first element,
     * so a caller computing an index and getting it wrong read a value
     * instead of the -1 that says "not there". */
    if (arr < 0 || arr >= d->n || d->tok[arr].type != JS_ARR ||
        i < 0 || i >= d->tok[arr].size) return -1;
    int p = arr + 1;
    for (int j = 0; j < i; j++) p = js_skip(d, p);
    return p;
}

/* The token's type, or -1 for no such token. The accessors below fold
 * "absent" and "present but not that type" into the same default; a caller
 * that has to tell those apart needs this. */
static inline int js_typeof(const js_doc *d, int t)
{
    return (t < 0 || t >= d->n) ? -1 : (int)d->tok[t].type;
}

static inline double js_num(const js_doc *d, int t, double dflt)
{
    if (t < 0 || t >= d->n || d->tok[t].type != JS_NUM) return dflt;
    char buf[64];
    int len = d->tok[t].end - d->tok[t].start;
    if (len <= 0 || len >= (int)sizeof buf) return dflt;
    memcpy(buf, d->src + d->tok[t].start, (size_t)len);
    buf[len] = 0;
    return atof(buf);
}

/* int64_t rather than long: the manifest's `off` and `scale_off` are byte
 * offsets into a trunk that is 57 GB on K3, and `long` is 32 bits on
 * Windows. Every other caller wants a small number and casts to int. */
static inline int64_t js_int(const js_doc *d, int t, int64_t dflt)
{
    double v = js_num(d, t, (double)dflt);
    return (int64_t)v;
}

/* Reads a value, not a presence: `false` is false, and anything that is not
 * a JSON boolean — including a missing key — is dflt. */
static inline int js_bool(const js_doc *d, int t, int dflt)
{
    if (t < 0 || t >= d->n || d->tok[t].type != JS_BOOL) return dflt;
    return d->src[d->tok[t].start] == 't';
}

/* Copies at most cap-1 bytes; always NUL-terminates. */
/* One \uXXXX, or -1. Reads exactly four hex digits and no further. */
static inline int js_hex4(const char *p, const char *end)
{
    if (end - p < 4) return -1;
    int v = 0;
    for (int i = 0; i < 4; i++) {
        const unsigned char c = (unsigned char)p[i];
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return -1;
        v = v * 16 + d;
    }
    return v;
}

/* A JSON string's bytes, UNESCAPED into `buf`, NUL-terminated and
 * truncated to `cap`. `p`..`end` is the span BETWEEN the quotes.
 *
 * It used to be a memcpy, which is right for ASCII and silent for
 * everything else: Python's json.dump escapes non-ASCII by default, so
 * DeepSeek-V4.1's control tokens reached the engine as the seven literal
 * characters "\uff5c" and every one of them tokenized as prose. Every
 * container before it had ASCII-only markup — <|open|>, <|endoftext|> —
 * which is why a JSON reader that did not decode JSON went four releases
 * without being noticed.
 *
 * Returns the byte length written. Exposed because specials.json is read
 * by a scanner of its own in tokenizer.c, and one of the two decoding and
 * the other not is how this came to be wrong in two places at once. */
static inline size_t js_unescape(const char *p, const char *end,
                                 char *buf, size_t cap)
{
    size_t o = 0;
    if (cap == 0) return 0;
    while (p < end && o + 1 < cap) {
        if (*p != '\\') { buf[o++] = *p++; continue; }
        if (++p >= end) break;                    /* trailing backslash */
        switch (*p) {
        case 'b': buf[o++] = '\b'; p++; break;
        case 'f': buf[o++] = '\f'; p++; break;
        case 'n': buf[o++] = '\n'; p++; break;
        case 'r': buf[o++] = '\r'; p++; break;
        case 't': buf[o++] = '\t'; p++; break;
        case 'u': {
            int cp = js_hex4(p + 1, end);
            if (cp < 0) { buf[o++] = *p++; break; }   /* not an escape */
            p += 5;
            if (cp >= 0xD800 && cp <= 0xDBFF && end - p >= 6 &&
                p[0] == '\\' && p[1] == 'u') {
                const int lo = js_hex4(p + 2, end);
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p += 6;
                }
            }
            if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;   /* lone half */
            /* Written only when the whole sequence fits, so a truncated
             * buffer ends on a character boundary rather than mid-glyph. */
            const size_t need = cp < 0x80 ? 1 : cp < 0x800 ? 2
                              : cp < 0x10000 ? 3 : 4;
            if (o + need + 1 > cap) { p = end; break; }
            if (cp < 0x80) {
                buf[o++] = (char)cp;
            } else if (cp < 0x800) {
                buf[o++] = (char)(0xC0 | (cp >> 6));
                buf[o++] = (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                buf[o++] = (char)(0xE0 | (cp >> 12));
                buf[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                buf[o++] = (char)(0x80 | (cp & 0x3F));
            } else {
                buf[o++] = (char)(0xF0 | (cp >> 18));
                buf[o++] = (char)(0x80 | ((cp >> 12) & 0x3F));
                buf[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                buf[o++] = (char)(0x80 | (cp & 0x3F));
            }
            break;
        }
        default: buf[o++] = *p++; break;           /* \" \\ \/ and the rest */
        }
    }
    buf[o] = 0;
    return o;
}

/* Keys are still compared raw (js_streq): one written with an escape does
 * not match, which is a lookup that fails rather than one that succeeds
 * wrongly. */
static inline const char *js_str(const js_doc *d, int t, char *buf, size_t cap)
{
    buf[0] = 0;
    if (t < 0 || t >= d->n || d->tok[t].type != JS_STR || cap == 0) return buf;
    js_unescape(d->src + d->tok[t].start, d->src + d->tok[t].end, buf, cap);
    return buf;
}

#endif /* WASTE_JSON_H */
