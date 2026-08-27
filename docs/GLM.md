# GLM-5.3-Flash on WASTE

`zai-org/GLM-5.3-Flash` — 313 B parameters, 328 GB of fp8 as published,
`Glm5NextForConditionalGeneration`. It is close enough to Kimi K3 that most
of this engine already ran it: the same KDA recurrence, the same MLA with
`kv_b_proj` absorbed, the same sigmoid / top-k router over per-layer expert
banks, the same expert-per-record container. Three things are new, and this
document is mostly about those and about the three places the release states
something the engine already states, differently.

## The shape

| | |
|---|---:|
| layers | 45 (34 KDA, 11 MLA — one in four) |
| hidden | 4096 |
| routed experts | 288, top-8, 2048 wide |
| shared experts | 1 |
| dense layers | the first 3 |
| KDA | 64 heads x 128, conv 4, `gate_lower_bound` -5.0 |
| MLA | 64 heads, `q_lora` 1536, `kv_lora` 512, qk_nope 256, v 256 |
| rope | none at all: `qk_rope_head_dim` 0 and `mla_use_nope` |
| context | 1,048,576 |
| vocab | 154,880 |

The layer mix, the KDA parameterization and the router are Kimi K3's, field
for field. What follows is what is not.

### mHC — Manifold-Constrained Hyper-Connections

The residual stream is not one vector but `hc_mult` = 4 parallel ones.
Before each sublayer a learned mapping reads all four at once — a
`24 x 16384` projection of the flattened, RMS-normalized streams — and
produces three things:

    pre  [4]     collapse weights: the one vector the sublayer runs on
    post [4]     where the sublayer's output lands, per stream
    comb [4][4]  how the streams mix into each other

`comb` is then projected onto the doubly-stochastic manifold by
Sinkhorn-Knopp, twenty alternating row/column normalizations, which is the
constrained half of the name and what keeps the four streams' norms from
diverging over 45 layers. After the sublayer,

    x[i] <- post[i] * y + sum_k comb[k][i] * x[k]

and at the end of the stack the four are collapsed by an unweighted mean.

Two sites per layer, ninety in all. The arithmetic is negligible — a 4x4
Sinkhorn is nothing — but the projection that feeds it is 393216 weights
per site, 35 B in total, and it lives in the resident trunk.

This is the same *kind* of mechanism as K3's Attention Residuals: both let a
layer see more than the one before it. They are mutually exclusive in
practice, and `src/model.c` treats them as three alternatives — plain
residual, AttnRes, mHC — at the same two points in the layer loop.

### The clamped SwiGLU

`swiglu_limit` is 10.0: the gate half is clamped above, the up half on both
sides, **before** the SiLU-and-multiply. It fires on real activations, so
dropping it produces a model that looks right and drifts.

The family now has three activations — plain, K3's SiTU, and this — at
seven call sites in `model.c`, which is exactly the shape in which a fourth
gets forgotten at one of them. They are one function,
`waste_act_pair_range`.

### DeepSeek Sparse Attention, k-pool flavour

An MLA layer does not attend over the whole context. A small indexer, with
projections of its own, scores *pools* of `index_kpool` = 4 adjacent cached
tokens and keeps the best `index_topk / index_kpool` = 512 of them; the
query attends over those pools' tokens plus the tail of the context that has
not filled a pool yet.

Two things make it cheap to keep resident:

- A pool's compressed key is one 128-wide vector per four tokens — 32x less
  than the raw keys the scores would need — and it is computed once, when
  the pool's last token arrives. The per-step state is a rolling buffer of
  the four (key, gate) pairs of the pool being filled, plus an append-only
  array of finished pool keys. At 128 B per token per full-attention layer
  it is a fourteenth of what the latents already cost.

- **Below 2048 tokens of context this is exactly dense attention.** With no
  more complete pools than the query is allowed to keep, the selection is
  every visible token; `dsa_select` says so by returning -1 and the head
  loop takes its ordinary path. That is not an approximation for short
  prompts — it is what the arithmetic reduces to, and it is why the
  selection cost only appears where the saving does.

The one place this is unspecified: a pool whose every head scored negative
lands at exactly 0 after the relu, and so do its neighbours. Which of those
the top-k keeps is ordered by index upstream and by heap order here. They
contribute the same nothing either way.

## The three silent differences

All three are converter-side, all three are invisible to a forward-pass
diff against an oracle that reads the same manifest — it would be wrong the
same way — and all three are covered by `tests/test_convert_glm.py`.

**The text model is nested the other way round from K3.**

    K3    language_model.model.layers.N.…    language_model.lm_head.weight
    GLM   model.language_model.layers.N.…    lm_head.weight

Same two components, opposite order, and `lm_head` inside the wrapper on
one and outside it on the other. The engine looks up
`{tensor_prefix}model.layers.N.…`, so K3's spelling is a prefix away from
it and GLM's is not: nothing is found, every tensor reads as absent, and
the load refuses a container that in fact holds every weight. GLM's
container is therefore prefix-less and the wrapper is unnested at
conversion (`glm_rename`); there is nothing left for a prefix to
disambiguate, since the vision tower is not carried.

This one was caught before the first conversion rather than after it, by
checking every name and shape the engine will demand against the
checkpoint's own index — 1246 tensors, 0 missing, 0 mismatched — which is
an hour of conversion cheaper than finding out at load.

**`linear_attn_config.kda_layers` is 0-based on GLM and 1-based in a WASTE
manifest.** Copied through, it puts KDA on the wrong layers and MLA on the
rest. Every tensor is still found, every shape still checks out, and the
model answers noise. `convert.py` rebuilds it from `layer_types`, which is
unambiguous, rather than shifting it by one and hoping.

**`eos_token_id` is a list of three** — end-of-text plus two turn markers.
The engine's config holds one id; a list read as an integer stops on
nothing. The first is taken, because the turn markers belong to the chat
format rather than to the model config.

## The tokenizer

GLM ships `tokenizer.json` (the `tokenizers` library's own JSON) where both
Kimi releases ship a tiktoken rank file. Same byte-level BPE, different
container, so `tools/hf_tokenizer.py` re-encodes it: undo GPT-2's
bytes-to-unicode escape, emit `base64(bytes) rank`, and `src/tokenizer.c`
reads it exactly as it reads Kimi's. It refuses rather than approximates
when the merge list is not ordered by the id of what it produces — that
ordering is what makes merge-by-rank and merge-by-list-position the same
encoder.

One difference is real and was found by checking rather than by reading.
**Kimi's pre-tokenization pattern gives Han its own `[\p{Han}]+` branch and
GLM's does not**, so on GLM a Han run and the Latin run it touches are one
pre-token rather than two. Sixteen tokens in this vocabulary span that
boundary, and they are the frequent kind:

| | Kimi's pattern | GLM's, and the release |
|---|---|---|
| `A股` | `32 98963` | `111321` |
| `维生素C` | `103261 34` | `121569` |
| `C罗` | `34 99209` | `126152` |
| `QQ音乐` | `47724 99908` | `126724` |

So the container states which, in `tokenizer_han_split`, and the engine
passes it to `waste_tok_set_han_split`. The default is the Kimi pattern: a
default that has to be set to keep working is a default that gets missed.
`tools/tokdiff.py` picks its reference from what the source ships and
compares 21 strings, the last two of which cross that boundary.

## Converting

```bash
uv run --with torch python tools/convert.py \
    --src /path/to/GLM-5.3-Flash --out glm.waste
```

Nothing GLM-specific on the command line: the fp8 reader, the DeepSeek MoE
tensor naming and the nested `text_config` were all already there for K2 and
K3. The conversion drops two families of tensor the container has no reader
for — the MTP layer at index 45 (`num_nextn_predict_layers`, a
speculative-decoding head) and the vision tower — because the trunk is
resident for the life of the process and a tensor nothing reads is RAM taken
from the expert cache.

The mHC projections are kept at 8 bits rather than the trunk's usual 4. Its
24 outputs decide how the residual streams mix for a whole layer, so an
error there is not one weight's worth; the two widths are 34 MB and 8 MB.

### What it comes to

Arithmetic from `config.json`, not a measurement — nothing here has been
converted yet:

| | |
|---|---:|
| trunk, resident | ~4.8 GiB |
| experts, on disk at VQ3R | ~106.5 GiB |
| one token's working set | ~3.2 GB (8 experts x 42 layers x 9.5 MB) |

Which is a very different proposition from K3's 27.3 GiB trunk and 982 GiB
of experts. On a 64 GB machine the trunk leaves room to cache something like
40% of the whole expert set, where K3 gets a token's working set and a half.

## Not implemented

- **The vision tower.** GLM's is not K3's — 24 blocks at 1024 wide, a
  downsample conv, a gated merger, 2D rope — and `src/vision.c` reads K3's.
  `convert.py` therefore writes no `vision.json`, so the container is
  text-only and `waste_image_add` refuses images by name rather than running
  the wrong tower on them.
- **Chunked prefill.** `waste_model_prefill` routes a GLM container through
  the per-token path. The chunked path carries one residual per token and
  one dense attention per layer, and mHC's parallel streams and the
  indexer's per-token pool bookkeeping are neither; a chunk that quietly ran
  without them would differ from the same prompt decoded one token at a
  time, which is the exact failure `WASTE_CHUNK` exists to be checked
  against.
- **Cross-layer top-k sharing.** `indexer_types` is all `"full"` on this
  release. A container converted from a release that shares a selection
  across layers is refused rather than produced.
- **MTP.** The extra prediction layer is dropped, as above.
- **`chat.json`.** The release ships a Jinja template, which `convert.py`
  copies, but the CLI reads the declarative form in `examples/`. Until one
  is transcribed the CLI says so and falls back to raw continuation.

## What is checked

`tests/run.sh` builds its own GLM-shaped container — a few megabytes, seed
0, byte-reproducible — because none of the checks that run on a Kimi reach
any of the three new things, and all three fail quietly. mHC produces
weight-shaped logits from any mixing matrix; a clamp that never fires looks
like a clamp that works; and an indexer that selects every pool is
indistinguishable from one that selects the right ones until the context
outgrows `index_topk`.

- **against the oracle.** `tools/kimi_ref.py` grew mHC, the clamped SwiGLU
  and the indexer, so one reference covers the family. Sixteen tokens with
  `index_topk` 8 over pools of 4: four complete pools of which the indexer
  may keep two, so the sparse branch is really reached. At twelve tokens or
  fewer every pool is kept and the check would pass with the selection
  deleted. Measured max abs 5.7e-6 on the last token's logits, against a
  1e-3 threshold — the same order as the Kimi baseline.
- **that the indexer selects.** The same weights with `index_topk` raised
  above the prompt length, which is one number in the config and nothing
  else. The two must differ; measured 0.56 max abs.
- **session state.** The four streams and the indexer's pool history are
  both session state, and both are written to and read from a saved state.
- **the chunked fallback.** Bit-identical to decoding the same prompt.
- **the converter's config handling**, `tests/test_convert_glm.py`, for the
  two silent differences above.
- **the tokenizer**, `tools/tokdiff.py` against the `tokenizers` library on
  the real `tokenizer.json`: 21 of 21 identical, including the Han/Latin
  boundary.

What is *not* checked is the model itself. No GLM container has been
converted here, so every number above that is not from `config.json` comes
from a synthetic container at 1/32 scale. The shapes are the ones the engine
branches on; the weights are noise.
