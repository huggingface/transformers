# HELIX

**H**ierarchical **E**pisodic **L**inear **I**nde**X** — a sequence architecture that keeps what attention
is good at while dropping the quadratic bill.

```bash
pip install helix-lm
```

```python
import torch
from helix_lm import HelixConfig, HelixForCausalLM

model = HelixForCausalLM(HelixConfig(vocab_size=32000, hidden_size=1024, num_hidden_layers=12))
logits = model(torch.randint(0, 32000, (1, 8192))).logits
text = model.generate(torch.randint(0, 32000, (1, 64)), max_new_tokens=32, temperature=0.8)
```

```bash
helix info      # what it is and what it costs
helix demo      # build a model, generate, verify the decode state
helix bench     # cost against context length
helix recall    # train with and without the index strand, and compare
```

---

## The idea

A transformer compares every token to every other token. Ten thousand tokens is fifty million
comparisons; double the input and you quadruple the work. Linear-time alternatives fix that by squeezing
the past into a fixed-size state — and then they cannot quote you a name they saw once, because a state of
`S` bits cannot tell apart more than `2^S` histories. That is a counting argument, not an engineering gap.

HELIX does not try to compress its way around it. It **separates memory capacity from memory bandwidth**.
Every block braids three mixers over the same residual stream:

| strand | reads | training cost | held at decode |
| --- | --- | --- | --- |
| **L** local | the last `local_blocks × block_size` tokens, exactly, with a different window per head | `O(N · span)` | a window-sized ring buffer |
| **R** recurrent | everything, compressed into a matrix-valued delta-rule state | `O(N · d_k · d_v)` | one matrix + two conv states per head |
| **I** index | `index_topk` blocks chosen anywhere in the past by a beam descent over a landmark tree | `O(N · topk · block_size)` + `O(N log N)` routing | the key/value history and a landmark tree, of which a step reads `O(topk · block_size + branching · log N)` |

Storage is `O(N)`, append-only, never rewritten, and can live in cold memory. What a single token *touches*
is bounded. So the claim is **`O(1)` hot state and `O(log N)` probes** — not `O(1)` total memory, which is
impossible for anything that can quote an exact name from a million tokens back.

## How the index works

The past is cut into blocks of `block_size` tokens. Each closed block gets a **landmark** — a learned pooled
summary that mixes a plain mean with an attention pool, because a mean alone erases the rare token a later
query will be hunting for. Landmarks are pooled again, `index_branching` at a time, into a tree, using one
weight-shared pooler at every internal level.

Two rules keep it causal *and* parallel:

1. **A node is eligible for query block `J` only if the whole span it summarizes ends at or before `J`.** A
   summary is never consulted by a query it partly describes, so a landmark is computed once and reused by
   every query.
2. **Routing for block `J` comes from the hidden state at the end of block `J − 1`.** Strictly past for
   every token of block `J`, so one gather serves `block_size` queries. Anything routing could not
   anticipate inside that block is covered exactly by strand L.

The descent keeps `index_beam_width` nodes per level. Candidates at each level are the beam's children
**plus** that level's *frontier* — eligible nodes whose parent is not eligible. At most `branching − 1` per
level, and the frontiers across all levels tile `[0, J)` exactly, so nothing reachable is cut off by a beam
that descended elsewhere. Selection scores are fed back as additive attention logits, accumulated along the
descent path, which is what lets gradient reach every level of the tree rather than just the leaves.

## What has been measured

On small models, 4 CPU cores. Reproduce with `helix bench` and `helix recall`.

**Cost against context length.** Attention pairs are exact — the query/key products the forward pass
actually forms, counted by instrumenting the attention call.

| tokens | HELIX pairs/token | full attention pairs/token | HELIX | full attention |
| ---: | ---: | ---: | ---: | ---: |
| 4 096 | 8 192 | 32 776 | 0.4 s | 0.2 s |
| 8 192 | 8 192 | 65 544 | 0.9 s | 0.6 s |
| 16 384 | 8 192 | 131 080 | 1.7 s | 1.9 s |
| 32 768 | 8 192 | 262 152 | 4.3 s | 6.7 s |

HELIX fits `N^1.00` in attention pairs; full attention fits `N^2.00`. By 32k tokens HELIX forms **32×
fewer** and runs **1.6× faster** — as unfused reference PyTorch against a fused SDPA kernel.

**Recall from far back.** Multi-query associative recall: bindings written near the start, queried ~200
tokens later, well outside the local window.

| model | recall | 95% CI |
| --- | ---: | :---: |
| HELIX (L+R+I) | **30.8%** | [28.5%, 33.1%] |
| HELIX no index (L+R) | 6.0% | [4.8%, 7.2%] |
| full attention | 32.0% | [29.7%, 34.3%] |
| *chance* | *6.2%* | |

Read the two gaps separately. **HELIX against its own ablation is the result**: strip the index strand and
the same model, same width, same depth, sits at chance — its window cannot see the bindings and its
recurrent state cannot hold them. That gap is ~21 standard errors. **HELIX against full attention is a
tie**, not a win or a loss: 1.2 points with a 1.7-point standard error, z = 0.72.

## Correctness

`pytest` covers the properties you can check without training anything:

- **strictly causal** — editing token `t` leaves every earlier logit bitwise unchanged;
- **decode == prefill** — token-by-token generation from the fixed-size state reproduces the one-shot
  forward, as does unaligned chunked prefill;
- **right padding is exactly invariant**;
- **constant work per token** — each query reads the same number of keys at any context length;
- **the cache is what the architecture claims** — fixed-size recurrent and convolution states everywhere,
  a window-capped key/value cache on non-indexing layers;
- **tiling is bitwise inert** — `attention_tile_blocks` caps peak activation memory and changes nothing;
- **every parameter gets gradient**, landmark poolers included.

## What this is not

**There are no trained checkpoints.** Nobody has trained HELIX on real text. Everything above is about
complexity, causality and recall mechanics — properties provable on an untrained model or a synthetic
probe — not about perplexity or language quality. The inductive-bias arguments (multi-scale windows, short
convolutions, surprise-gated writes) are untested hypotheses.

Treat this as an architecture proposal with a working, tested reference implementation, not a better model.
Finding out whether it is actually better needs real training runs on real hardware.

Known design limits, stated plainly:

- One retrieval serves a whole query block, so `index_topk` must cover the diversity a `block_size` span
  asks for.
- Routing is one block stale — the index cannot react to the token currently being generated.
- Landmarks are pooled from keys that already carry their rotary phase, so routing is not purely
  content-addressed.
- Left padding shifts the block grid, so it is not bit-identical to an unpadded run. Right padding is.
- No fused kernel for the gathered attention yet; the scaling is right, the constant factor is not.

## Configuration

Every knob is on `HelixConfig`, documented in its docstring. The ones that matter most:

```python
HelixConfig(
    block_size=64,          # memory-block granularity
    local_blocks=4,         # strand L sees 4 previous blocks plus its own
    num_window_scales=4,    # head groups get geometrically different windows
    index_topk=8,           # blocks strand I retrieves per query block
    index_layer_stride=3,   # ...on every third layer; the rest keep an O(1) KV cache
    index_branching=8,      # landmark tree fan-out
    attention_tile_blocks=64,  # caps peak activation memory; no effect on the result
)
```

## License

Apache-2.0. Portions of the delta-rule and rotary helpers derive from
[HuggingFace Transformers](https://github.com/huggingface/transformers) (Apache-2.0); see `NOTICE`.

**Made by Nathan.**
