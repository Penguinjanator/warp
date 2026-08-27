#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_convert_glm.py — the three places GLM-5.3-Flash states something the
engine also states, differently.

All three are silent when wrong. `linear_attn_config.kda_layers` is 0-based
on GLM and 1-based in a WASTE manifest, so a straight copy puts KDA on the
wrong layers: every tensor is still found, every shape still checks out,
and the model answers noise. `eos_token_id` is a list of three, and the
engine's config holds one id, so a list read as an integer stops on nothing.
And the text model is nested the other way round from K3 —
`model.language_model.layers.N` against `language_model.model.layers.N` —
which no prefix can reconcile, so every tensor reads as absent and the load
refuses a container that holds every weight.

None of them can be caught by a forward-pass diff against an oracle that
reads the same manifest — the oracle would be wrong the same way. They have
to be checked here, against what the release says.

No torch and no source weights: this is convert.py's own decision code.

  python3 tests/test_convert_glm.py
"""
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A GLM-5.3-Flash config at test scale, with the release's *shapes*: the
# layer mix alternating three linear-attention layers and one sparse
# attention, kda_layers written 0-based as the release writes it, and the
# three-id eos list.
GLM_TEXT = {
    "model_type": "glm5_next_text",
    "num_hidden_layers": 8,
    "hidden_size": 64,
    "n_routed_experts": 16,
    "num_experts_per_tok": 4,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "eos_token_id": [154820, 154827, 154829],
    "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"] +
                   ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
    "indexer_types": ["full"] * 8,
    "hc_mult": 4,
    "swiglu_limit": 10.0,
    "index_topk": 2048,
    "index_kpool": 4,
    "linear_attn_config": {
        "num_heads": 8, "head_dim": 16, "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        # 0-based, exactly as the release ships it
        "kda_layers": [0, 1, 2, 4, 5, 6],
        "full_attn_layers": [3, 7],
    },
}
GLM = {"architectures": ["Glm5NextForConditionalGeneration"],
       "model_type": "glm5_next"}

KIMI_TEXT = {
    "model_type": "kimi_linear",
    "num_hidden_layers": 4,
    "eos_token_id": 2,
    # 1-based already, and no layer_types to derive from
    "linear_attn_config": {"num_heads": 4, "head_dim": 32,
                           "kda_layers": [1, 2, 4],
                           "full_attn_layers": [3]},
}


def install_stubs():
    torch = types.ModuleType("torch")
    torch.device = lambda s: s
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch
    mx = types.ModuleType("mxfp4")
    mx.ST = object
    mx.unblock_scale = lambda q, scale, block: q
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
    """What main() builds: the inner text config with the wrapper kept."""
    return {**text, "_outer": {k: v for k, v in outer.items()}}


def main():
    glm = nested(GLM, GLM_TEXT)
    kimi = nested({"architectures": ["KimiLinearForCausalLM"]}, KIMI_TEXT)

    check("GLM is recognised by the name it gives itself", CONV.is_glm(glm))
    check("a Kimi container is not mistaken for one", not CONV.is_glm(kimi))

    out = CONV.normalise_cfg(glm)
    kda = out["linear_attn_config"]["kda_layers"]
    check("kda_layers comes out 1-based", kda == [1, 2, 3, 5, 6, 7],
          f"got {kda}")
    check("it is not the release's 0-based list copied through",
          kda != GLM_TEXT["linear_attn_config"]["kda_layers"])
    check("full_attn_layers agrees with it",
          out["linear_attn_config"]["full_attn_layers"] == [4, 8])
    # Every layer is one or the other, and none is both: the property the
    # off-by-one breaks without changing any count.
    n = GLM_TEXT["num_hidden_layers"]
    both = set(kda) & set(out["linear_attn_config"]["full_attn_layers"])
    cover = set(kda) | set(out["linear_attn_config"]["full_attn_layers"])
    check("the two lists partition the layers 1..n",
          not both and cover == set(range(1, n + 1)))

    check("eos_token_id is one id, the first", out["eos_token_id"] == 154820)
    check("the MoE keys the engine reads are present",
          out.get("num_experts") == 16 and
          out.get("num_experts_per_token") == 4 and
          out.get("num_shared_experts") == 1 and
          out.get("moe_renormalize") is True)
    check("the GLM-only keys pass through untouched",
          out.get("hc_mult") == 4 and out.get("swiglu_limit") == 10.0 and
          out.get("index_topk") == 2048 and out.get("index_kpool") == 4)
    check("normalise_cfg does not mutate its argument",
          GLM_TEXT["linear_attn_config"]["kda_layers"] == [0, 1, 2, 4, 5, 6]
          and GLM_TEXT["eos_token_id"] == [154820, 154827, 154829])

    kout = CONV.normalise_cfg(kimi)
    check("a Kimi config is left exactly as it was",
          kout["linear_attn_config"]["kda_layers"] == [1, 2, 4] and
          kout["eos_token_id"] == 2)

    # ---- the wrapper, which GLM nests the other way round ---------------
    #
    # K3   language_model.model.layers.N.…   language_model.lm_head.weight
    # GLM  model.language_model.layers.N.…   lm_head.weight
    #
    # The engine looks up {tensor_prefix}model.layers.N, so GLM's spelling
    # is not a prefix away from it — copied through, every tensor reads as
    # absent and the load refuses a container that holds every weight.
    check("the wrapper is unnested, so the container is prefix-less",
          CONV.glm_rename("model.language_model.layers.7.self_attn.o_proj.weight")
          == "model.layers.7.self_attn.o_proj.weight"
          and CONV.glm_rename("model.language_model.embed_tokens.weight")
          == "model.embed_tokens.weight"
          and CONV.glm_rename("model.language_model.norm.weight")
          == "model.norm.weight")
    check("lm_head is already where a prefix-less container wants it",
          CONV.glm_rename("lm_head.weight") == "lm_head.weight")
    check("a name that is not under the wrapper is left alone",
          CONV.glm_rename("model.visual.blocks.0.attn.qkv.weight")
          == "model.visual.blocks.0.attn.qkv.weight")

    # ---- the trunk tensors GLM ships that nothing here reads ------------
    drop = CONV.glm_drop_trunk("model.language_model.", 45)
    keep = [
        "model.language_model.layers.44.self_attn.o_proj.weight",
        "model.language_model.layers.0.hc_attn_fn",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    gone = [
        # the MTP layer sits at index num_hidden_layers
        "model.language_model.layers.45.eh_proj.weight",
        "model.language_model.layers.45.shared_head.norm.weight",
        # a tower this engine does not have a reader for
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.merger.proj.weight",
    ]
    check("the layers of the forward pass are kept",
          all(not drop(n) for n in keep),
          str([n for n in keep if drop(n)]))
    check("the MTP layer and the vision tower are dropped",
          all(drop(n) for n in gone),
          str([n for n in gone if not drop(n)]))
    # A 10-layer model must not drop layer 4 because "45" contains "4".
    check("the layer index is matched whole",
          not CONV.glm_drop_trunk("model.", 10)("model.layers.4.mlp.up_proj.weight")
          and CONV.glm_drop_trunk("model.", 10)("model.layers.10.eh_proj.weight"))

    print(f"\n{len(fails)} failed" if fails else "\nall ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
