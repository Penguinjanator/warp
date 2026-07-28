#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Batch-1 routing trace from any HF MoE model, for routing_stats.py simulate.

Registers a hook on every layer's router gate, generates N tokens greedy,
and writes trace.jsonl in the `simulate` format:
  {"tok": n, "layers": [[e0..e7], ...]}   (decode steps only, batch 1)

Usage: trace_hf.py [out.jsonl] [model_id] [n_tokens]
  e.g. uv run --with torch --with transformers python tools/trace_hf.py \
       trace.jsonl allenai/OLMoE-1B-7B-0924 300

For Kimi K3 this runs on a rented high-RAM box (the model must fit);
everything else in the WASTE pipeline consumes only the trace file.
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[2] if len(sys.argv) > 2 else "allenai/OLMoE-1B-7B-0924"
N_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 300
PROMPT = (
    "Write a C function that parses a JSON array of integers without using "
    "any external library, then explain its memory usage and edge cases, "
    "and finally propose unit tests for it."
)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "mps" else torch.float32
print(f"device={dev} dtype={dtype}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=dtype, low_cpu_mem_usage=True
).to(dev).eval()

cfg = model.config
top_k = cfg.num_experts_per_tok
print(f"layers={cfg.num_hidden_layers} experts={cfg.num_experts} top_k={top_k}",
      flush=True)

step_layers = []   # current decode step: one topk-list per layer
trace = []


def mk_hook(layer_idx):
    def hook(module, inp, out):
        # newer transformers: gate returns (logits, topk_weights, topk_index);
        # older: raw router logits [tokens, n_experts]
        if isinstance(out, tuple):
            idx = out[2].reshape(-1, top_k)
        else:
            idx = torch.topk(out.float(), top_k).indices.reshape(-1, top_k)
        # last position = routing of the newest token (works both with KV
        # cache, where idx has 1 row, and with full re-forward, where it
        # has seq_len rows)
        step_layers.append((layer_idx, sorted(idx[-1].tolist())))
    return hook


for li, layer in enumerate(model.model.layers):
    layer.mlp.gate.register_forward_hook(mk_hook(li))

ids = tok(PROMPT, return_tensors="pt").input_ids.to(dev)
print(f"prompt tokens: {ids.shape[1]}; generating {N_NEW}...", flush=True)

torch.manual_seed(0)
with torch.no_grad():
    cur = ids
    for i in range(N_NEW):
        out = model(cur)  # full re-forward: O(n^2) but cache-agnostic
        step_layers.sort()
        if i > 0 and len(step_layers) == cfg.num_hidden_layers:
            trace.append({"tok": i, "layers": [l for _, l in step_layers]})
        step_layers.clear()
        # sample (temp 0.7): greedy on base models degenerates into loops,
        # which would inflate expert-reuse stats and bias the feasibility gate
        probs = torch.softmax(out.logits[0, -1].float() / 0.7, dim=-1)
        nxt = torch.multinomial(probs, 1).reshape(1, 1).to(cur.device)
        cur = torch.cat([cur, nxt], dim=1)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_NEW}", flush=True)

with open(sys.argv[1] if len(sys.argv) > 1 else "trace_olmoe.jsonl", "w") as f:
    for t in trace:
        f.write(json.dumps(t) + "\n")
print(f"wrote {len(trace)} tokens of trace", flush=True)
print(tok.decode(cur[0, -80:]), flush=True)
