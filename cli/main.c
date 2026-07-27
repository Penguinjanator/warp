/*
 * cli/main.c — the WASTE command line.
 *
 * A client of src/waste.h with no private access: if the CLI can do it, an
 * embedding host can do it too. That rule is the point of the split.
 *
 *   waste run    MODEL "prompt"   generate once
 *   waste chat   MODEL            REPL, state kept across turns
 *   waste bench  MODEL            throughput and cache behaviour
 *   waste plan   MODEL [--budget] memory floor and what a budget buys
 *   waste info   MODEL            container and build details
 *   waste version
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/waste.h"

#define MAXTOK 8192

static uint64_t parse_size(const char *s)
{
    char *end;
    double v = strtod(s, &end);
    while (*end == ' ') end++;
    switch (*end) {
    case 'g': case 'G': return (uint64_t)(v * (1ULL << 30));
    case 'm': case 'M': return (uint64_t)(v * (1ULL << 20));
    case 'k': case 'K': return (uint64_t)(v * (1ULL << 10));
    default: return (uint64_t)v;
    }
}

static void human(uint64_t b, char *out, size_t cap)
{
    const double g = (double)b / (1ULL << 30);
    if (g >= 1.0) snprintf(out, cap, "%.2f GB", g);
    else snprintf(out, cap, "%.0f MB", (double)b / (1ULL << 20));
}

/* ---- shared option parsing --------------------------------------------- */

typedef struct {
    uint64_t budget;
    uint32_t ctx, max_tokens;
    float temperature, top_p;
    int top_k, threads, quiet;
    uint64_t seed;
} opts;

static void opts_init(opts *o)
{
    memset(o, 0, sizeof *o);
    o->ctx = 4096;
    o->max_tokens = 128;
    o->top_p = 1.0f;
}

static int parse_opts(int argc, char **argv, int from, opts *o)
{
    for (int i = from; i < argc; i++) {
        if (!strcmp(argv[i], "--budget") && i + 1 < argc) o->budget = parse_size(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) o->ctx = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) o->max_tokens = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) o->temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) o->top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) o->top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) o->seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) o->threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q") || !strcmp(argv[i], "--quiet")) o->quiet = 1;
        else if (argv[i][0] == '-') { fprintf(stderr, "unknown option %s\n", argv[i]); return -1; }
    }
    return 0;
}

static waste_status open_model(const char *path, const opts *o, waste_ctx **ctx)
{
    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.ram_budget_bytes = o->budget;
    cfg.ctx_tokens = o->ctx;
    cfg.n_threads = o->threads;
    return waste_open(path, &cfg, ctx);
}

static int fail(const char *what, waste_status s)
{
    fprintf(stderr, "%s: %s\n", what, waste_strerror(s));
    if (s == WASTE_E_RAM_BUDGET)
        fprintf(stderr, "  (run `waste plan MODEL` to see the floor)\n");
    return 1;
}

/* ---- token callback ----------------------------------------------------- */

typedef struct { int quiet; uint32_t n; uint64_t hit, miss; double ms; } sink;

static int on_token(const waste_token_info *i, const char *piece, void *user)
{
    sink *s = (sink *)user;
    if (!s->quiet) { fputs(piece, stdout); fflush(stdout); }
    s->n++;
    s->hit += i->experts_hit;
    s->miss += i->experts_missed;
    s->ms += i->ms_total;
    return 0;
}

/* ---- commands ----------------------------------------------------------- */

static int cmd_plan(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: waste plan MODEL [--budget N] [--ctx N]\n"); return 2; }
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 3, &o)) return 2;

    waste_memplan p;
    const waste_status st = waste_plan_memory(argv[2], o.ctx, &p);
    if (st != WASTE_OK) return fail("plan", st);

    char b[5][32];
    human(p.trunk_bytes, b[0], 32); human(p.state_bytes, b[1], 32);
    human(p.scratch_bytes, b[2], 32); human(p.min_expert_cache, b[3], 32);
    human(p.floor_bytes, b[4], 32);
    printf("memory plan for %s (ctx %u)\n\n", argv[2], o.ctx);
    printf("  resident trunk        %12s\n", b[0]);
    printf("  KDA state + KV cache  %12s\n", b[1]);
    printf("  scratch               %12s\n", b[2]);
    printf("  minimum expert cache  %12s\n", b[3]);
    printf("  ---------------------------------\n");
    printf("  FLOOR                 %12s\n", b[4]);
    human(p.recommended_bytes, b[0], 32);
    printf("  recommended           %12s   (floor + 3x a token's working set;\n"
           "                                      below that the cache keeps\n"
           "                                      nothing alive between tokens)\n", b[0]);
    if (o.budget) {
        char bb[32];
        human(o.budget, bb, 32);
        if (o.budget < p.floor_bytes) printf("\n  budget %s is BELOW the floor — open would fail\n", bb);
        else {
            char cb[32];
            human(o.budget - p.floor_bytes + p.min_expert_cache, cb, 32);
            printf("\n  budget %s -> expert cache %s\n", bb, cb);
        }
    }
    return 0;
}

static int cmd_info(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: waste info MODEL\n"); return 2; }
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 3, &o)) return 2;
    waste_ctx *c;
    const waste_status st = open_model(argv[2], &o, &c);
    if (st != WASTE_OK) return fail("open", st);

    waste_model_info mi;
    waste_model_get_info(c, &mi);
    waste_memplan used;
    waste_memory_used(c, &used);
    char cb[32];
    human(used.min_expert_cache, cb, 32);
    printf("%s\n\n", waste_build_info());
    printf("  arch          %s\n", mi.arch);
    printf("  layers        %u\n", mi.n_layers);
    printf("  experts       %u (top-%u)\n", mi.n_experts, mi.top_k);
    printf("  hidden        %u\n", mi.hidden);
    printf("  routed params %.1f B total, %.1f B active/token\n",
           mi.params_total / 1e9, mi.params_active / 1e9);
    printf("  quantization  %s\n", mi.quant_summary);
    printf("  expert cache  %s\n", cb);
    waste_close(c);
    return 0;
}

static int run_prompt(waste_ctx *c, const opts *o, const char *prompt, int show_stats)
{
    int32_t ids[MAXTOK];
    size_t n = 0;
    waste_status st = waste_tokenize(c, prompt, 0, ids, MAXTOK, &n);
    if (st != WASTE_OK) return fail("tokenize", st);

    waste_gen_params p;
    waste_gen_params_init(&p);
    p.max_tokens = o->max_tokens;
    p.temperature = o->temperature;
    p.top_p = o->top_p;
    p.top_k = o->top_k;
    p.seed = o->seed;

    sink s = { o->quiet, 0, 0, 0, 0.0 };
    if (!o->quiet) fputs(prompt, stdout);
    st = waste_generate(c, ids, n, &p, on_token, &s);
    printf("\n");
    if (st != WASTE_OK && st != WASTE_E_CANCELLED) return fail("generate", st);

    if (show_stats && s.n) {
        const double sec = s.ms / 1000.0;
        fprintf(stderr, "\n[%u tokens, %.2f s, %.2f tok/s | experts %llu hit / "
                        "%llu miss = %.0f%%]\n",
                s.n, sec, s.n / sec,
                (unsigned long long)s.hit, (unsigned long long)s.miss,
                100.0 * (double)s.hit / (double)(s.hit + s.miss ? s.hit + s.miss : 1));
    }
    return 0;
}

static int cmd_run(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: waste run MODEL \"prompt\" [options]\n"); return 2; }
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 4, &o)) return 2;
    waste_ctx *c;
    const waste_status st = open_model(argv[2], &o, &c);
    if (st != WASTE_OK) return fail("open", st);
    const int r = run_prompt(c, &o, argv[3], !o.quiet);
    waste_close(c);
    return r;
}

static int cmd_chat(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: waste chat MODEL [options]\n"); return 2; }
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 3, &o)) return 2;
    waste_ctx *c;
    const waste_status st = open_model(argv[2], &o, &c);
    if (st != WASTE_OK) return fail("open", st);

    printf("%s — /reset clears state, /stats prints counters, Ctrl-D exits\n",
           waste_build_info());
    char line[8192];
    while (1) {
        fputs("\n> ", stdout);
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        size_t l = strlen(line);
        while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
        if (!l) continue;
        if (!strcmp(line, "/reset")) { waste_state_reset(c); printf("(state cleared)\n"); continue; }
        if (!strcmp(line, "/stats")) {
            waste_stats s;
            waste_get_stats(c, &s);
            printf("%llu tokens, %.2f tok/s, experts %llu hit / %llu miss (%.0f%%), %.2f GB read\n",
                   (unsigned long long)s.tokens_generated,
                   s.sec_total > 0 ? s.tokens_generated / s.sec_total : 0.0,
                   (unsigned long long)s.experts_hit, (unsigned long long)s.experts_missed,
                   100.0 * (double)s.experts_hit /
                       (double)(s.experts_hit + s.experts_missed ? s.experts_hit + s.experts_missed : 1),
                   (double)s.bytes_read / (1ULL << 30));
            continue;
        }
        printf("\n");
        run_prompt(c, &o, line, 0);
    }
    waste_close(c);
    return 0;
}

static int cmd_bench(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: waste bench MODEL [-n N] [--budget N]\n"); return 2; }
    opts o; opts_init(&o);
    o.max_tokens = 64;
    if (parse_opts(argc, argv, 3, &o)) return 2;
    waste_ctx *c;
    const waste_status st = open_model(argv[2], &o, &c);
    if (st != WASTE_OK) return fail("open", st);

    waste_memplan used;
    waste_memory_used(c, &used);
    char cb[32];
    human(used.min_expert_cache, cb, 32);
    printf("%s\n  expert cache %s, %u tokens\n\n", waste_build_info(), cb, o.max_tokens);

    opts q = o;
    q.quiet = 1;
    run_prompt(c, &q, "Write a C function that parses a JSON array of integers "
                      "and explain its edge cases.", 0);

    waste_stats s;
    waste_get_stats(c, &s);
    const double tps = s.sec_total > 0 ? s.tokens_generated / s.sec_total : 0;
    const uint64_t acc = s.experts_hit + s.experts_missed;
    printf("  %.2f tok/s (%.0f ms/token)\n", tps, 1000.0 / (tps > 0 ? tps : 1));
    printf("  experts   %llu hit / %llu miss = %.1f%% hit\n",
           (unsigned long long)s.experts_hit, (unsigned long long)s.experts_missed,
           100.0 * (double)s.experts_hit / (double)(acc ? acc : 1));
    printf("  disk      %.2f GB total, %.3f GB/token\n",
           (double)s.bytes_read / (1ULL << 30),
           (double)s.bytes_read / (1ULL << 30) /
               (s.tokens_generated ? s.tokens_generated : 1));
    waste_close(c);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        printf("waste %s — run large MoE models from a streaming container\n\n"
               "  waste run    MODEL \"prompt\"   generate\n"
               "  waste chat   MODEL            interactive, state kept\n"
               "  waste bench  MODEL            throughput and cache stats\n"
               "  waste plan   MODEL            memory floor and budget\n"
               "  waste info   MODEL            container details\n"
               "  waste version\n\n"
               "options: --budget 8G  --ctx N  -n N  --temp F  --top-p F\n"
               "         --top-k N  --seed N  --threads N  -q\n",
               waste_version());
        return argc < 2 ? 2 : 0;
    }
    if (!strcmp(argv[1], "version")) {
        printf("%s\n", waste_build_info());
        return 0;
    }
    if (!strcmp(argv[1], "run"))   return cmd_run(argc, argv);
    if (!strcmp(argv[1], "chat"))  return cmd_chat(argc, argv);
    if (!strcmp(argv[1], "bench")) return cmd_bench(argc, argv);
    if (!strcmp(argv[1], "plan"))  return cmd_plan(argc, argv);
    if (!strcmp(argv[1], "info"))  return cmd_info(argc, argv);
    fprintf(stderr, "unknown command '%s' (try --help)\n", argv[1]);
    return 2;
}
