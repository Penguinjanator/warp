#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_convert_ds41.py — what convert.py has to decide about
DeepSeek-V4.1-Flash before it writes a byte, and cannot decide by looking at
the shapes.

Every check here is for something that is silent when wrong.

The tensor names are the loudest of them only because the engine looks its
tensors up by fixed strings: a name this converter does not recognise comes
back unchanged, is written under the checkpoint's own spelling, and the load
then refuses a container that holds every weight — after hours. So the whole
name set is pinned, both directions.

The rest are quieter. `bos/eos/pad_token_id` are stated on the wrapper and
the engine reads the text config, so an unlifted eos stops on nothing. The
MoE keys are spelled `n_routed_experts` here and `num_experts` there, and
the engine reading zero experts refuses with no diagnostic. And the fp4
experts are `I8` tensors under their own `.weight` name, so a reader that
knows only K3's `_packed` suffix finds them, returns the raw nibble pairs as
floats, and every shape still checks out.

No torch and no source weights: this is convert.py's own decision code.

  python3 tests/test_convert_ds41.py
"""
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The release's own shapes at test scale, with every key the engine reads.
DS41_TEXT = {
    "model_type": "deepseek_v41_text",
    "num_hidden_layers": 6,
    "hidden_size": 128,
    "head_dim": 32,
    "qk_rope_head_dim": 8,
    "q_lora_rank": 48,
    "o_groups": 2,
    "o_lora_rank": 16,
    "sliding_window": 4,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.5,
    "moe_intermediate_size": 64,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "swiglu_limit": 10.0,
    "compress_ratios": [0, 0, 2, 2, 1, 1],
    "kv_source_layer_ids": [2, 4],
    "index_source_layer_ids": [2, 4],
    "compress_rope_theta": 160000.0,
    "index_n_heads": 2,
    "index_head_dim": 16,
    "index_topk": 4,
    "engram_layer_ids": [1, 3],
    "engram_max_ngram_size": 3,
    "engram_n_heads": 2,
    "engram_head_dim": 64,
}
DS41_OUTER = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "model_type": "deepseek_v41",
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "image_token_id": 129264,
}

# Every tensor pattern the release ships, one instance each, and what the
# engine looks it up by. Read against docs/DS41.md and against
# validate_text_tensors in src/model.c — the two lists are one list.
RENAMES = [
    ("embed.weight",                 "model.embed_tokens.weight"),
    ("norm.weight",                  "model.norm.weight"),
    ("head.weight",                  "lm_head.weight"),
    ("image_start",                  "model.image_start"),
    ("image_end",                    "model.image_end"),
    ("image_newline",                "model.image_newline"),
    ("layers.0.attn_norm.weight",    "model.layers.0.input_layernorm.weight"),
    ("layers.0.ffn_norm.weight",
     "model.layers.0.post_attention_layernorm.weight"),
    ("layers.0.hc_attn_fn",          "model.layers.0.hc_attn_fn"),
    ("layers.0.hc_attn_base",        "model.layers.0.hc_attn_base"),
    ("layers.0.hc_attn_scale",       "model.layers.0.hc_attn_scale"),
    ("layers.0.hc_ffn_fn",           "model.layers.0.hc_ffn_fn"),
    ("layers.0.hc_ffn_base",         "model.layers.0.hc_ffn_base"),
    ("layers.0.hc_ffn_scale",        "model.layers.0.hc_ffn_scale"),
    ("layers.0.attn.attn_sink",      "model.layers.0.self_attn.attn_sink"),
    ("layers.0.attn.wq_a.weight",    "model.layers.0.self_attn.q_a_proj.weight"),
    ("layers.0.attn.q_norm.weight",
     "model.layers.0.self_attn.q_a_layernorm.weight"),
    ("layers.0.attn.wq_b.weight",    "model.layers.0.self_attn.q_b_proj.weight"),
    ("layers.0.attn.wkv.weight",     "model.layers.0.self_attn.kv_proj.weight"),
    ("layers.0.attn.kv_norm.weight",
     "model.layers.0.self_attn.kv_layernorm.weight"),
    ("layers.0.attn.wo_a.weight",    "model.layers.0.self_attn.o_a_proj.weight"),
    ("layers.0.attn.wo_b.weight",    "model.layers.0.self_attn.o_b_proj.weight"),
    ("layers.2.attn.compressor.wkv.weight",
     "model.layers.2.self_attn.compress.kv_proj.weight"),
    ("layers.2.attn.compressor.wgate.weight",
     "model.layers.2.self_attn.compress.gate_proj.weight"),
    ("layers.2.attn.compressor.norm.weight",
     "model.layers.2.self_attn.compress.norm.weight"),
    ("layers.2.attn.indexer.wq_b.weight",
     "model.layers.2.self_attn.indexer.q_b_proj.weight"),
    ("layers.2.attn.indexer.wk.weight",
     "model.layers.2.self_attn.indexer.k_proj.weight"),
    ("layers.2.attn.indexer.k_norm.weight",
     "model.layers.2.self_attn.indexer.k_layernorm.weight"),
    ("layers.2.attn.indexer.weights_proj.weight",
     "model.layers.2.self_attn.indexer.weights_proj.weight"),
    ("layers.0.ffn.gate.weight",
     "model.layers.0.block_sparse_moe.gate.weight"),
    ("layers.0.ffn.gate.bias",
     "model.layers.0.block_sparse_moe.gate.e_score_correction_bias"),
    ("layers.0.ffn.gate.bias_vl",
     "model.layers.0.block_sparse_moe.gate.e_score_correction_bias_vl"),
    ("layers.0.ffn.shared_experts.w1.weight",
     "model.layers.0.block_sparse_moe.shared_experts.gate_proj.weight"),
    ("layers.0.ffn.shared_experts.w3.weight",
     "model.layers.0.block_sparse_moe.shared_experts.up_proj.weight"),
    ("layers.0.ffn.shared_experts.w2.weight",
     "model.layers.0.block_sparse_moe.shared_experts.down_proj.weight"),
    ("layers.1.engram.wkv.weight",   "model.layers.1.engram.wkv.weight"),
    ("layers.1.engram.q_weight",     "model.layers.1.engram.q_weight"),
    ("layers.1.engram.k_weight",     "model.layers.1.engram.k_weight"),
    ("vision.blocks.3.attn.wo.bias", "vision_tower.blocks.3.attn.wo.bias"),
    ("aligner.w1.weight",            "mm_projector.w1.weight"),
]


def install_stubs():
    torch = types.ModuleType("torch")
    torch.device = lambda s: s
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch
    mx = types.ModuleType("mxfp4")
    mx.ST = object
    mx.unblock_scale = lambda q, scale, block: q
    mx.e8m0_scale = lambda raw: raw
    sys.modules["mxfp4"] = mx
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import convert
    return convert


CONV = install_stubs()
fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        fails.append(name)
        print(f"  FAIL {name}{(': ' + detail) if detail else ''}")


def nested(outer, text):
    return {**text, "_outer": dict(outer)}


def main():
    ds = nested(DS41_OUTER, DS41_TEXT)
    kimi = nested({"architectures": ["KimiLinearForCausalLM"]},
                  {"model_type": "kimi_linear", "num_hidden_layers": 4})
    glm = nested({"architectures": ["Glm5NextForConditionalGeneration"]},
                 {"model_type": "glm5_next_text", "num_hidden_layers": 4})

    check("DeepSeek-V4.1 is recognised by the name it gives itself",
          CONV.is_ds41(ds))
    check("a Kimi container is not mistaken for one", not CONV.is_ds41(kimi))
    check("a GLM container is not mistaken for one", not CONV.is_ds41(glm))
    check("and it is not mistaken for a GLM", not CONV.is_glm(ds))

    # ---- names, both directions -----------------------------------------
    wrong = [(src, CONV.ds41_rename(src), want)
             for src, want in RENAMES if CONV.ds41_rename(src) != want]
    check(f"all {len(RENAMES)} tensor kinds rename to what the engine looks up",
          not wrong, "; ".join(f"{s} -> {g} (want {w})" for s, g, w in wrong[:3]))
    # The half that actually protects the conversion: an unrecognised name
    # comes back unchanged, and ds41_check_names is what notices.
    stray = CONV.ds41_check_names([s for s, _ in RENAMES], CONV.ds41_rename)
    check("nothing in the known set is reported as stray", not stray,
          str(stray[:3]))
    invented = CONV.ds41_check_names(
        ["layers.0.attn.w_something_new.weight", "brand_new_tensor"],
        CONV.ds41_rename)
    check("a tensor this release does not have IS reported",
          len(invented) == 2, str(invented))
    check("the MTP draft head is dropped, not renamed",
          CONV.ds41_drop_trunk(6)("mtp.6.attn.wq_a.weight") and
          not CONV.ds41_drop_trunk(6)("layers.5.attn.wq_a.weight"))
    check("the vision tower is NOT dropped — it is carried and left on disk",
          not CONV.ds41_drop_trunk(6)("vision.blocks.0.attn.wo.weight"))

    # ---- the prefixes ----------------------------------------------------
    # DS41 puts nothing before `layers.N`, where K3 puts two components and
    # GLM puts the same two the other way round. Pinned together, because
    # the three are one function and adding the third is how the other two
    # get broken.
    K3 = {"architectures": ["KimiK3ForConditionalGeneration"],
          "text_config": {"model_type": "kimi_linear",
                          "architectures": ["KimiLinearForCausalLM"]}}
    GLM_OUT = {"architectures": ["Glm5NextForConditionalGeneration"],
               "model_type": "glm5_next",
               "text_config": {"model_type": "glm5_next_text"}}
    DS_OUT = {**DS41_OUTER, "text_config": DS41_TEXT}
    KL = {"architectures": ["KimiLinearForCausalLM"],
          "model_type": "kimi_linear"}
    want = {
        "Kimi-Linear": (KL, "", "model."),
        "Kimi K3": (K3, "language_model.", "language_model.model."),
        "GLM-5.3-Flash": (GLM_OUT, "", "model.language_model."),
        "DeepSeek-V4.1": (DS_OUT, "", ""),
    }
    for who, (c, pfx, src) in want.items():
        got = CONV.source_prefixes(c)[:2]
        check(f"{who}: prefixes {pfx!r} {src!r}", got == (pfx, src), str(got))

    # ---- config ----------------------------------------------------------
    out = CONV.normalise_cfg(ds)
    check("the MoE keys the engine reads are present",
          out.get("num_experts") == 8 and
          out.get("num_experts_per_token") == 2 and
          out.get("num_shared_experts") == 1 and
          out.get("moe_renormalize") is True)
    check("bos/eos/pad are lifted off the wrapper onto the text config",
          out.get("eos_token_id") == 1 and out.get("bos_token_id") == 0 and
          out.get("pad_token_id") == 2)
    check("the CSA2 keys pass through untouched",
          out.get("head_dim") == 32 and out.get("o_groups") == 2 and
          out.get("o_lora_rank") == 16 and out.get("sliding_window") == 4 and
          out.get("compress_ratios") == [0, 0, 2, 2, 1, 1] and
          out.get("kv_source_layer_ids") == [2, 4] and
          out.get("compress_rope_theta") == 160000.0)
    check("the Engram keys pass through untouched",
          out.get("engram_layer_ids") == [1, 3] and
          out.get("engram_max_ngram_size") == 3 and
          out.get("engram_n_heads") == 2 and
          out.get("engram_head_dim") == 64)
    check("scoring_func survives — sigmoid would be a different router",
          out.get("scoring_func") == "sqrtsoftplus")
    check("normalise_cfg does not mutate its argument",
          "eos_token_id" not in DS41_TEXT and "num_experts" not in DS41_TEXT)

    # ---- the MoE layout probe -------------------------------------------
    # `ffn` is a third segment name, and the probe reads it off the names on
    # disk rather than off a flag. A checkpoint whose experts it cannot find
    # converts zero of them.
    class FakeST:
        def __init__(self, names):
            self.names = set(names)

        def have(self, n):
            return n in self.names
    st = FakeST(["layers.0.ffn.experts.0.w1.weight"])
    check("the ffn segment is found by probing, not by a flag",
          CONV.moe_layout(st, "", 0)[:2] == ("ds41", "ffn"))
    check("and w1/w3/w2 map onto gate/up/down",
          CONV.moe_layout(st, "", 0)[2] ==
          (("gate", "w1"), ("up", "w3"), ("down", "w2")))

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
