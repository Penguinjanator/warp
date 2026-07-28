#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# tests/run.sh — every check we have, in one place, exiting non-zero on the
# first real failure.
#
# Written after losing time twice to checks that silently did not run: once
# to objects compiled against a stale header, once to a stale test binary.
# So this rebuilds first, states what it is about to do, and never treats a
# missing prerequisite as a pass — it says SKIP, loudly.
#
#   tests/run.sh [model.waste]
#
# Env: WASTE_REF_MODEL  container to use for the end-to-end checks
#      WASTE_REF_SRC    source weights, for the container round-trip
#      WASTE_ORACLE     logits dumped by tools/kimi_ref.py for THE SAME
#                       token ids this script uses (see IDS below) — a dump
#                       from a different prompt will look like an engine bug

set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-${WASTE_REF_MODEL:-/Users/marco/models/kimi-linear.waste}}"
SRC="${WASTE_REF_SRC:-/Volumes/WasteDisk/kimi-linear}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Without a reference container, build a synthetic one: a few megabytes of
# deterministic noise in the real format. It cannot check the engine against
# the oracle — those logits belong to actual Kimi-Linear weights — but every
# check that compares the engine against itself works on it, which is what
# lets CI and a fresh clone run the engine at all instead of skipping it.
SYNTHETIC=0
if [ ! -d "$MODEL" ]; then
    if python3 tools/make_test_container.py "$TMP/tiny.waste" >/dev/null 2>&1; then
        MODEL="$TMP/tiny.waste"
        SYNTHETIC=1
    fi
fi

pass=0; fail=0; skip=0
ok()   { printf "  \033[32mPASS\033[0m  %s\n" "$1"; pass=$((pass+1)); }
no()   { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; fail=$((fail+1)); }
sk()   { printf "  \033[33mSKIP\033[0m  %s — %s\n" "$1" "$2"; skip=$((skip+1)); }
head_() { printf "\n\033[1m%s\033[0m\n" "$1"; }

head_ "build"
if make -s test >/dev/null 2>&1 && make -s >/dev/null 2>&1; then
    ok "make && make test"
else
    no "build failed"
    make test 2>&1 | grep -E "error" | head -5
    exit 1
fi

# ---------------------------------------------------------------- unit ----
head_ "kernels vs the reference implementations"

if command -v uv >/dev/null 2>&1; then
    ./test_k3parts "$TMP/k3parts.bin" >/dev/null 2>&1
    if uv run --quiet --with torch --no-project python tools/k3parts_ref.py \
           "$TMP/k3parts.bin" 2>/dev/null | grep -q "^PASS"; then
        ok "K3 components (SiTU, both decay gates, AttnRes)"
    else
        no "K3 components"
    fi

    if KDA_T=24 KDA_H=4 KDA_K=32 KDA_V=32 uv run --quiet --with torch \
           --with fla-core --with einops --no-project python tools/kda_ref.py \
           2>/dev/null | grep -q "^PASS"; then
        ok "KDA kernel vs fla's naive_recurrent_kda"
    else
        no "KDA kernel"
    fi
else
    sk "kernel checks" "uv not installed"
fi

# ------------------------------------------------------------ container ----
head_ "container"

if [ -d "$MODEL" ]; then
    if ./test_container "$MODEL"/experts-L*.bin 2 2>/dev/null | grep -q "0 problems"; then
        ok "expert records read through the C structs (magic, offsets, 4 KiB)"
    else
        no "expert record layout"
    fi

    if [ "$SYNTHETIC" = 1 ]; then
        sk "container round-trip" "synthetic container has no source weights"
    elif [ -d "$SRC" ] && command -v uv >/dev/null 2>&1; then
        if uv run --quiet --with torch --no-project python tools/verify_container.py \
               --container "$MODEL" --src "$SRC" --experts 1 2>/dev/null \
               | grep -q "^PASS"; then
            ok "dequantized weights match the source"
        else
            no "container round-trip"
        fi
    else
        sk "container round-trip" "source weights not at $SRC"
    fi
else
    sk "container checks" "no container at $MODEL"
fi

# ----------------------------------------------------------------- e2e ----
head_ "engine"

if [ -d "$MODEL" ]; then
    # The oracle fixture pins these to Kimi-Linear's vocabulary; a synthetic
    # container has 256 entries, and an id past the table is an out-of-range
    # read rather than a different answer.
    if [ "$SYNTHETIC" = 1 ]; then
        IDS=3,7,11,5,9,13,2,17,4,8,19,23,6,29,12,31
    else
        IDS=1008,10484,318,15383,387,11,316,276,10484,318,19509,387,31082,13,646,10484
    fi

    ./test_forward "$MODEL" "$IDS" "$TMP/seq.bin" 0 >/dev/null 2>&1
    WASTE_CHUNK=1 ./test_forward "$MODEL" "$IDS" "$TMP/chunk.bin" 0 >/dev/null 2>&1
    if python3 - "$TMP/seq.bin" "$TMP/chunk.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
d = max(abs(x - y) for x, y in zip(a, b))
sys.exit(0 if d < 1e-3 and a.index(max(a)) == b.index(max(b)) else 1)
PY
    then ok "chunked prefill == token-at-a-time"
    else no "chunked prefill diverges"
    fi

    WASTE_Q8=0 ./test_forward "$MODEL" "$IDS" "$TMP/f32.bin" 0 >/dev/null 2>&1
    if python3 - "$TMP/seq.bin" "$TMP/f32.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
    then ok "int8 trunk storage == f32 weights"
    else no "int8 storage changes results"
    fi

    WASTE_BACKEND=cpu ./test_forward "$MODEL" "$IDS" "$TMP/cpu.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/seq.bin" "$TMP/cpu.bin"; then
        ok "SIMD backend bit-identical to the CPU baseline"
    else
        # a difference here is allowed to be tiny, but must be tiny
        if python3 - "$TMP/seq.bin" "$TMP/cpu.bin" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
        then ok "SIMD backend matches the CPU baseline (within fp noise)"
        else no "SIMD backend diverges from the CPU baseline"
        fi
    fi

    WASTE_CACHE_MB=512 ./test_forward "$MODEL" "$IDS" "$TMP/cache.bin" 0 >/dev/null 2>&1
    if cmp -s "$TMP/seq.bin" "$TMP/cache.bin"; then
        ok "expert cache is bit-identical to no cache"
    else
        no "expert cache changes results"
    fi

    ORACLE="${WASTE_ORACLE:-tests/fixtures/oracle_kimilinear_16tok.bin}"
    if [ "$SYNTHETIC" = 1 ]; then
        sk "engine vs the PyTorch oracle" "synthetic container has no reference logits"
    elif [ -f "$ORACLE" ]; then
        if python3 - "$TMP/seq.bin" "$ORACLE" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if max(abs(x - y) for x, y in zip(a, b)) < 1e-3 else 1)
PY
        then ok "engine matches the PyTorch oracle"
        else no "engine diverges from the oracle"
        fi
    else
        sk "oracle diff" "no fixture; regenerate with tools/kimi_ref.py --dump"
    fi

    if ./test_state "$MODEL" 2>/dev/null | grep -q "^STATE OK"; then
        ok "saved session resumes identically"
    else
        no "session state does not round-trip"
    fi

    # learned hotlist: a second run should start warmer than the first
    if [ "$SYNTHETIC" = 1 ]; then
        sk "learned hotlist" "synthetic container carries no tokenizer"
    else
    rm -f "$MODEL/usage.waste"
    cold=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G --learn 2>&1 \
           | grep -oE "[0-9]+ miss" | grep -oE "[0-9]+" || echo 0)
    warm=$(./waste run "$MODEL" "The capital of France is" -n 12 --budget 5G 2>&1 \
           | grep -oE "[0-9]+ miss" | grep -oE "[0-9]+" || echo 0)
    rm -f "$MODEL/usage.waste"
    if [ "${warm:-0}" -gt 0 ] && [ "${warm:-0}" -lt "${cold:-0}" ]; then
        ok "learned hotlist warms the cache ($cold -> $warm misses)"
    else
        no "hotlist did not reduce misses ($cold -> $warm)"
    fi
    fi
else
    sk "engine checks" "no container at $MODEL"
fi

# --------------------------------------------------------------- budget ----
head_ "RAM budget"

# The default budget is the one path check_budget.sh cannot reach, because
# it always passes --budget. With no flag the engine chooses, and that
# choice is all that stands between a model whose recommendation exceeds
# the machine — K3 asks for 80.63 GB — and a swap storm. So assert the
# rule, not a number, and it holds on any host: the ceiling is the
# recommendation capped at 88% of physical RAM, or the floor when even
# that does not fit, less at most one expert record of slot rounding.
default_budget() {
    python3 - "$1" <<'PY'
import json, os, subprocess, sys

def j(*a):
    r = subprocess.run(["./waste", *a, "--json"], capture_output=True, text=True)
    return json.loads(r.stdout)

plan, info = j("plan", sys.argv[1]), j("info", sys.argv[1])
phys = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
cap = phys - phys // 8
# what the engine actually holds: the plan's mandatory parts plus the
# cache it really allocated, which is what `info` reports
held = plan["floor_bytes"] - plan["min_expert_cache"] + info["expert_cache_bytes"]
want = (plan["floor_bytes"] if plan["floor_bytes"] > cap
        else min(plan["recommended_bytes"], cap))
rec = plan["min_expert_cache"] // (2 * info["top_k"]) if info["top_k"] else 0
G = 1 << 30
print(f"{held/G:.2f} GB held, ceiling {want/G:.2f} GB, machine {phys/G:.2f} GB")
sys.exit(0 if want - rec - 1 <= held <= want else 1)
PY
}

# params_total is the number that ends up in a model card, and it is
# derived rather than stored: three matrices per expert, each as wide as
# the expert's input. In a latent MoE that width is the latent, not the
# hidden — K3 reported 5.44 T instead of 2.72 T for exactly one wrong
# field. Mirroring the formula here would only prove this script and the
# engine agree, so the count is also weighed against the bytes on disk:
# at 3 bits per weight the experts have to fit their bank, give or take
# one fp16 scale per output row and the record's 4 KiB alignment.
params_rule() {
    python3 - "$1" <<'PY'
import json, subprocess, sys

d = sys.argv[1]
man = json.load(open(f"{d}/manifest.json"))
r = subprocess.run(["./waste", "info", d, "--json"], capture_output=True, text=True)
info = json.loads(r.stdout)
c, lay = man["config"], man["layers"]

width = c.get("routed_expert_hidden_size") or c["hidden_size"]
inter = c["moe_intermediate_size"]
per_expert = 3 * width * inter
moe_layers = c["num_hidden_layers"] - c.get("first_k_dense_replace", 0)
total = per_expert * c["num_experts"] * moe_layers
active = per_expert * info["top_k"] * moe_layers

bits = man["expert_quant"]["bits_per_weight"]
rec = lay[next(iter(lay))]
on_disk = rec["bytes"] // rec["experts"]
lo = per_expert * bits // 8
hi = lo + 2 * (2 * inter + width) + 4096          # scales, then alignment

T = 1e12
print(f"{total/T:.2f} T total, {active/1e9:.1f} B active, "
      f"{on_disk/(1<<20):.2f} MiB/expert on disk")
sys.exit(0 if (moe_layers == len(lay)
               and info["params_total"] == total
               and info["params_active"] == active
               and lo <= on_disk <= hi) else 1)
PY
}

if [ -d "$MODEL" ]; then
    if ./waste plan "$MODEL" >/dev/null 2>&1; then ok "waste plan"; else no "waste plan"; fi
    # capture first: `set -o pipefail` would otherwise propagate the
    # deliberate non-zero exit of the command under test
    refusal=$(./waste run "$MODEL" x --budget 1M 2>&1 || true)
    if printf '%s' "$refusal" | grep -q "below the model's floor"; then
        ok "a budget under the floor is refused, not swapped into"
    else
        no "under-floor budget not refused"
    fi

    if out=$(default_budget "$MODEL" 2>/dev/null); then
        ok "no --budget picks a ceiling the machine can hold ($out)"
    else
        no "default budget off the rule (${out:-no output})"
    fi
    # A container from a future layout must be refused, not read against the
    # old rules — the field was written from the first converter and read by
    # nobody until it was wired up.
    FV=$(mktemp -d)
    python3 - "$MODEL/manifest.json" "$FV/manifest.json" <<'PYFV'
import json, sys
m = json.load(open(sys.argv[1])); m["format_version"] = 999
json.dump(m, open(sys.argv[2], "w"))
PYFV
    out=$(./waste info "$FV" 2>&1); rc=$?
    python3 - "$MODEL/manifest.json" "$FV/manifest.json" <<'PYFV'
import json, sys
m = json.load(open(sys.argv[1])); m.pop("format_version", None)
json.dump(m, open(sys.argv[2], "w"))
PYFV
    out2=$(./waste info "$FV" 2>&1); rc2=$?
    rm -rf "$FV"
    if [ "$rc" -ne 0 ] && [ "$rc2" -ne 0 ] &&
       printf '%s' "$out"  | grep -q "format version mismatch" &&
       printf '%s' "$out2" | grep -q "format version missing"; then
        ok "a container from another format version is refused"
    else
        no "format_version not enforced (rc=$rc rc2=$rc2)"
    fi

    if [ -n "${WASTE_SANITIZED:-}" ]; then
        sk "peak RSS inside the budget" "sanitizer shadow memory makes RSS meaningless"
    elif [ "$SYNTHETIC" = 1 ]; then
        sk "peak RSS inside the budget" "needs a tokenizer to drive the CLI"
    elif tests/check_budget.sh "$MODEL" 2>/dev/null | grep -q "^BUDGET OK"; then
        ok "peak RSS stays inside the configured budget"
    else
        no "peak RSS exceeded the budget"
    fi
else
    sk "budget checks" "no container at $MODEL"
fi

# The small model cannot catch budget accounting that is wrong in
# proportion to the model: K3 overran by 2-3 GB on scratch that Kimi-Linear
# sizes in single megabytes. Run the same check against K3 when it is here.
BIG="${BIG_MODEL:-/Users/marco/models/k3.waste}"
if [ -f "$BIG/manifest.json" ]; then
    if [ -n "${WASTE_SANITIZED:-}" ]; then
        sk "K3 budget check" "sanitizer shadow memory makes RSS meaningless"
    elif tests/check_budget.sh "$BIG" 32 long 2>/dev/null | grep -q "^BUDGET OK"; then
        ok "peak RSS stays inside the budget on K3 too"
    else
        no "peak RSS exceeded the budget on K3"
    fi

    # K3 is the only model here whose recommendation can exceed the
    # machine, so it is the one that makes the cap bite at all: on a 64 GB
    # host the default lands on 56.00 GB rather than the 80.63 GB asked
    # for. On a host large enough to hold the recommendation this still
    # passes — it checks the rule, not the clamp.
    if out=$(default_budget "$BIG" 2>/dev/null); then
        ok "no --budget is capped to the machine on K3 ($out)"
    else
        no "default budget not capped on K3 (${out:-no output})"
    fi
else
    sk "K3 budget check" "no container at $BIG"
fi

# ----------------------------------------------------------- parameters ----
head_ "parameter counts"

if [ -d "$MODEL" ]; then
    if out=$(params_rule "$MODEL" 2>/dev/null); then
        ok "params_total is what the container holds ($out)"
    else
        no "params_total off the rule (${out:-no output})"
    fi
else
    sk "parameter counts" "no container at $MODEL"
fi

# The only latent MoE here, so the only model on which the expert width and
# the hidden differ at all: without it the check passes either way.
if [ -f "$BIG/manifest.json" ]; then
    if out=$(params_rule "$BIG" 2>/dev/null); then
        ok "params_total counts K3's experts at the latent ($out)"
    else
        no "params_total off the rule on K3 (${out:-no output})"
    fi
else
    sk "K3 parameter counts" "no container at $BIG"
fi

# ------------------------------------------------------------ tokenizer ----
head_ "tokenizer"

if [ "$SYNTHETIC" = 1 ]; then
    sk "tokenizer diff" "synthetic container carries no tokenizer"
elif [ -d "$MODEL" ] && command -v uv >/dev/null 2>&1 && [ -d "$SRC" ]; then
    if uv run --quiet --with tiktoken --no-project python tools/tokdiff.py \
           "$MODEL" "$SRC" 2>/dev/null | tail -1 | grep -q "identical"; then
        ok "C tokenizer matches Python tiktoken"
    else
        no "tokenizer differs from tiktoken"
    fi
else
    sk "tokenizer diff" "needs uv, a container and source weights"
fi

printf "\n\033[1m%d passed, %d failed, %d skipped\033[0m\n" "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ]
