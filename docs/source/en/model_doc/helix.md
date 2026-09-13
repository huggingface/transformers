<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-09-13.*

# HELIX

HELIX (**H**ierarchical **E**pisodic **L**inear **I**nde**X**) is a decoder-only architecture built to keep
what self-attention is good at while dropping the quadratic bill. Every block braids three sequence mixers
that read the same residual stream and are summed back into it:

| strand | reads | training cost | what it holds at decode time |
| --- | --- | --- | --- |
| **L — local** | the last `local_blocks * block_size` tokens, exactly, with a different window per head | `O(N · local_span)` | a `sliding_window`-token ring buffer of keys and values |
| **R — recurrent** | everything, compressed into a matrix-valued delta-rule state | `O(N · d_k · d_v)` | one `d_k × d_v` matrix and two short convolution states per head |
| **I — index** | `index_topk` memory blocks chosen anywhere in the past by a beam descent over a landmark tree | `O(N · index_topk · block_size)` attention, `O(N log N)` routing | the whole key/value history plus an append-only landmark tree — of which a step reads `O(index_topk · block_size + branching · log N)` |

Only the layers that run strand I keep the full history; the rest cap their key/value cache at the window,
so with the default `index_layer_stride=3` two layers in three carry an O(1) cache.

Strands L and I share one set of `q`/`k`/`v` projections and differ only in *which* keys they read; their
outputs are mixed by a per-token, per-head softmax gate. Strand R has its own projections and its own
output projection.

## Why three strands

Each of the three exists to cover a specific failure of the other two.

**Attention is exact but quadratic.** Constraining it to a short window makes it linear, and a window is
also the right *prior* — most of what a token needs is nearby. HELIX gives each head group a different
window on a geometric ladder (`num_window_scales`), so a single layer sees several scales at once instead
of committing to one.

**A fixed-size recurrent state is `O(1)` but forgets.** HELIX's recurrence is a *gated delta rule*, not a
decaying sum: writing `v_t` first subtracts whatever the state already associates with `k_t`, so
re-binding a key does not pile up interference. It trains in the chunkwise-parallel (UT transform) form —
one batch of matmuls over the whole sequence, no token-by-token scan — and decodes from a state whose size
does not depend on how much text came before.

**But no fixed state can recall exactly.** This is not an engineering gap, it is a counting argument: a
state of `S` bits cannot distinguish more than `2^S` distinct histories, so exact recall over an
arbitrarily long context needs storage that grows with the context. HELIX takes that seriously instead of
trying to compress its way around it, and separates *storage* from *bandwidth*:

* storage grows as `O(N)`, is append-only, is never rewritten, and can sit in cold memory;
* the amount of it a single token touches is `O(index_topk · block_size + branching · log N)` — independent
  of `N` up to the log.

That is the actual claim HELIX makes about generation: **`O(1)` hot state and `O(log N)` probes**, not
`O(1)` total memory, which is impossible for any model that can quote a name it saw once.

## How the index stays causal and parallel

The past is cut into memory blocks of `block_size` tokens. Each closed block gets a **landmark**: a learned
pooled summary combining a plain mean with an attention pool, because a mean alone erases the rare token a
later query will be hunting for. Landmarks are then pooled again, `index_branching` at a time, into a tree —
using one weight-shared pooler at every internal level, which both saves parameters and biases the tree
towards summarizing the same way at every scale.

Two rules keep the whole thing trainable in parallel and free of leakage:

1. **A node is eligible for query block `J` only if the entire span it summarizes ends at or before `J`.**
   A summary is never consulted by a query it partly describes, so a landmark can be computed once for the
   whole sequence and reused by every query.
2. **Routing for block `J` is driven by the hidden state at the end of block `J - 1`.** That state is
   strictly in the past for every token of block `J`, so one gather serves `block_size` queries — which is
   what makes the gathered attention fit in `O(N · index_topk · block_size)` memory instead of blowing up
   per query. Anything the routing is therefore "stale" about is exactly what strand L covers exactly.

The descent keeps `index_beam_width` nodes per level. At every level the candidate set is the children of
the current beam **plus** that level's *frontier* — the eligible nodes whose parent is not itself eligible.
There are at most `branching - 1` frontier nodes per level, and the frontiers across all levels tile
`[0, J)` exactly, so nothing reachable is cut off by a beam that descended elsewhere.

Top-k is discrete, so the scores are fed back in as an additive bias on the attention logits of the blocks
that were selected. The bias is the sum of the scores along the descent path, which is what lets gradient
reach every level of the tree and not just the leaves.

## Inductive biases

HELIX is deliberately less agnostic than a transformer, on the theory that the right priors buy sample
efficiency:

- short depthwise **causal convolutions** on the recurrent q/k/v stream (locality);
- **multi-scale windows** across head groups (scale, rather than one hand-picked span);
- the **delta rule** itself, which is an explicit key→value binding prior rather than a generic mixer;
- **surprise gating** (`use_surprise_gating`): write strength is modulated by how novel a key is against a
  short causal pool of the keys before it, so state capacity goes to unpredictable content. It is a
  depthwise convolution, so it stays parallel over the sequence;
- **log-bucketed relative block distance** biasing the landmark descent — a saturating table still means
  something at ranges where a rotary phase has wrapped far past anything seen in training. (The attention
  *after* the descent reads the rotated keys shared with strand L, so token order inside a retrieved block
  is preserved; the bias adds the coarse block-level distance the landmarks themselves do not carry);
- **weight sharing** across the tree's internal levels.

These are design arguments, not measured results. See "What is and isn't verified" below.

## Usage

```python
import torch
from transformers import HelixConfig, HelixForCausalLM

config = HelixConfig(
    vocab_size=32000,
    hidden_size=1024,
    num_hidden_layers=12,
    num_attention_heads=16,
    num_key_value_heads=4,
    block_size=64,      # memory-block granularity
    local_blocks=4,     # strand L sees 4 previous blocks plus its own
    index_topk=8,       # strand I retrieves 8 blocks, shared by every query in the block
    index_layer_stride=3,  # ...on every third layer
)
model = HelixForCausalLM(config)

input_ids = torch.randint(0, config.vocab_size, (1, 4096))
print(model(input_ids, use_cache=False).logits.shape)
print(model.generate(input_ids[:, :64], max_new_tokens=16, do_sample=False).shape)
```

`layer_types` gives per-layer control: `"helix"` runs all three strands and keeps a full key/value history,
`"helix_local"` runs L + R only and caps its key/value cache at the local window. With the default
`index_layer_stride=3`, two out of every three layers keep an `O(1)` key/value cache.

## Practical notes

- **Padding.** Right padding is *exactly* invariant — trailing tokens cannot reach earlier positions, and
  the tests assert bit-identical logits. Left padding shifts the memory-block grid, since blocks are
  anchored to absolute cache positions; results stay causal and correct but are not bit-identical to an
  unpadded run. This is the same caveat chunked-attention models carry. Prefer right padding for training.
- **Masks.** HELIX consumes the raw 2D padding mask and builds its own block-structured masks; passing a
  prepared 4D mask raises.
- **Attention outputs.** The three strands normalize separately, so there is no single attention matrix to
  return and `output_attentions` is not supported.
- **Kernels.** This is a reference implementation in plain PyTorch. The recurrence picks up
  `flash-linear-attention` and `causal-conv1d` when they are installed; the block-gathered attention has no
  fused kernel yet, so absolute throughput is well below what the asymptotics allow.

## Known limitations of this design

These are properties of the architecture, not bugs, and they are the things to weigh before using it.

- **One retrieval serves a whole query block.** That is what makes strand I affordable, but it means
  `index_topk` has to cover the diversity of everything a `block_size`-token span asks for. Natural text
  suits this — consecutive tokens usually want related context — but a workload where adjacent tokens each
  need a *different* distant fact needs a larger `index_topk` or a smaller `block_size`.
- **Routing is one block stale.** Block `J` is routed from the end of block `J - 1`, which is what keeps it
  causal and shared. Whatever the routing could not anticipate within that block is covered exactly by
  strand L, so the staleness never leaves a hole — but it does mean the index cannot react to the token
  currently being generated.
- **Landmarks are pooled from rotated keys.** Strands L and I share one key projection, and RoPE is applied
  before the cache, so a landmark summarizes keys that already carry their absolute rotary phase. Routing
  is therefore not purely content-addressed; the block-distance bias exists partly to compensate. Pooling
  pre-rotation would be cleaner but needs a second key cache, which would give back the memory this design
  is buying.
- **The recurrent strand still decays over padding.** Padded positions are zeroed on the way in, but the
  state's decay gate still advances, the same as in other delta-rule and SSM models.

## What is and isn't verified

Verified in `tests/models/helix/test_modeling_helix.py`, on small random-weight models:

- strict causality (editing token `t` leaves every earlier logit bitwise unchanged);
- token-by-token decoding reproduces the one-shot forward, as does unaligned chunked prefill;
- right-padding invariance is exact;
- the index reaches blocks outside the local window, and every eligible query block selects something;
- each query attends over the same number of keys regardless of context length — the direct evidence for
  linear compute;
- the decode cache has the shape the architecture claims: fixed-size recurrent and convolution states, and
  a windowed key/value cache on `"helix_local"` layers.

Measured in [`examples/pytorch/helix/`](../../../examples/pytorch/helix), on 4 CPU cores with small models:

* `scaling.py` — the number of query/key products per token is flat in the context length for HELIX
  (fitted `N^1.00`) and grows for full attention (`N^2.00`); by 32k tokens HELIX forms 32x fewer and, even
  as an unfused reference implementation against a fused SDPA kernel, runs 1.6x faster.
* `mqar.py` — on multi-query associative recall with the bindings written ~200 tokens before they are
  queried, the *same model with the index strand removed* sits at 6.0% against a 6.2% chance floor, and
  putting the index back takes it to 30.8%. That is the index strand, and it is ~21 standard errors wide.
  Full attention scores 32.0% on the same budget, which is a statistical tie with HELIX rather than a win
  either way (z = 0.72). Single seed and well short of saturation, so it establishes that the mechanism
  works, not that HELIX matches full attention in general.

**Not verified:** anything about language-modeling quality at scale. There are no trained HELIX checkpoints.
The claims above are about complexity, causality and recall mechanics — properties you can check on an
untrained model or on a synthetic probe — not about perplexity, and the inductive-bias arguments in
particular are untested hypotheses. Treat this as an architecture proposal with a working, tested reference
implementation.

## HelixConfig

[[autodoc]] HelixConfig

## HelixModel

[[autodoc]] HelixModel
    - forward

## HelixForCausalLM

[[autodoc]] HelixForCausalLM
    - forward
