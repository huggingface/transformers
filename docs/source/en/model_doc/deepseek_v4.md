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
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-05-02.*

# DeepSeek-V4

[DeepSeek-V4](https://huggingface.co/deepseek-ai) is the next-generation MoE language model from DeepSeek
([paper](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/DeepSeek_V4.pdf)). The architecture replaces
DeepSeek-V3's Multi-head Latent Attention (MLA) with a hybrid local + long-range design, swaps residual connections
for Manifold-Constrained Hyper-Connections (mHC), and bootstraps the first few MoE layers with a static
token-id → expert-id hash table.

This implementation covers `DeepSeek-V4-Flash`, `DeepSeek-V4-Pro`, and their `-Base` pretrained siblings. All four
share the same architecture; they differ only in width / depth / expert count and weights.

## Architecture (paper §2)

### Hybrid attention (§2.3)

Each decoder block is one of three attention types, dispatched by `config.layer_types[i]`:

* **Sliding-window full attention** (`"sliding_attention"`): only the local window of `sliding_window` tokens, no
  long-range branch. Matches V3's "Full Attention" style for the bootstrap layers.
* **Compressed Sparse Attention** (`"compressed_sparse_attention"`, **CSA** — paper §2.3.1): a low-compression
  pool (`compress_rate_csa`, default `m=4`) with overlapping windows, plus a **Lightning Indexer** (eqs. 13–17)
  that scores queries against the pool and gathers the top `index_topk` blocks per query before they reach core
  attention.
* **Heavily Compressed Attention** (`"heavily_compressed_attention"`, **HCA** — paper §2.3.2): a high-compression
  pool (`compress_rate_hca`, default `m'=128`) with non-overlapping windows. No indexer — every pooled entry
  contributes to attention.

All three types share the same backbone:

* **Shared K=V Multi-Query Attention**: `num_key_value_heads = 1`; `kv_proj` produces a single KV head and the same
  tensor is read as both key and value.
* **Partial RoPE** (interleaved-pair, paper §2.3.3 "Partial Rotary Positional Embedding") on the trailing
  `qk_rope_head_dim = head_dim * partial_rotary_factor` channels of each head. The same rotation is applied with
  position `-i` to the attention output's rope slice (eq. 26) so the contribution of each KV entry stays a function
  of the *relative* distance to the query.
* **Per-head learnable attention sink** (eq. 27).
* **Grouped low-rank output projection** (§2.3.1 "Grouped Output Projection"): `o_groups` head-groups → `o_lora_rank`
  per group → `hidden_size`, computed by [`DeepseekV4GroupedLinear`] (`o_a_proj`) followed by `o_b_proj`. Cuts the
  per-token cost of the wide attention output without losing expressivity.
* **Shared sliding-window K=V branch** of size `sliding_window` ("Additional Branch of Sliding Window Attention",
  §2.3.1) preserves local fine-grained dependencies; the long-range compressor's output is concatenated with this
  branch's KVs before core attention.

### Manifold-Constrained Hyper-Connections (§2.2)

Residual connections are replaced by mHC (Xie et al., 2026): `hc_mult` parallel residual streams kept in shape
`[B, S, hc_mult, D]` throughout each block. Two [`DeepseekV4HyperConnection`] modules — `attn_hc` and `ffn_hc` — mix
streams in and out around the attention / MLP sublayers via a `(pre, post, comb)` triplet. The `comb` matrix is a
doubly-stochastic projection produced by `hc_sinkhorn_iters` Sinkhorn–Knopp iterations on the manifold, making
signal propagation non-expansive across deep stacks. A final [`DeepseekV4HyperHead`] collapses the `hc_mult`
streams down to a single sequence before the model norm.

### MoE schedule (§2.1)

Routing is configured per layer by `config.mlp_layer_types`, with values from `{"hash_moe", "moe"}`:

* `"hash_moe"`: expert indices come from a frozen `tid2eid[input_ids]` lookup populated from the V4 checkpoint.
  The learned gate `weight` still produces the per-expert scores that weight the selected experts; only
  *which-experts* is static. Used for the first few bootstrap layers (default 3, override via legacy
  `num_hash_layers`).
* `"moe"`: standard top-k routed MoE. The expert affinity uses **Sqrt(Softplus(·))** instead of V3's Sigmoid
  ("we change the activation function that computes the affinity scores from Sigmoid(·) into Sqrt(Softplus(·))",
  paper §2.1), and V3's `n_group` / `topk_group` constraint is dropped. The auxiliary-loss-free strategy
  (DeepSeek's `noaux_tc`) is preserved via the `e_score_correction_bias` buffer that biases the top-k argmax
  without flowing gradients.

Routed experts use a **clamped SwiGLU** (`gate.clamp(max=swiglu_limit)`, `up.clamp(min=-swiglu_limit, max=swiglu_limit)`,
then `act_fn(gate) * up`) on top of the standard Mixtral `[num_experts, 2 * moe_intermediate_size, hidden_size]`
expert weight layout. A single shared expert (a plain SwiGLU MLP at `moe_intermediate_size` width) runs in parallel
on every token.

### Attention mask layout

Each `DeepseekV4Attention` layer extends the standard sliding-window-causal mask along the key axis with a
`block_bias` returned by its compressor, then feeds the concatenated mask to `eager_attention_forward`. The
sliding-section (left, `[S, S]`) is the same for every layer type; the compressor-section (right) differs by
layer type and is the actual "novel" piece introduced by V4.

The diagrams below were produced with a tiny config (`sliding_window=8`, CSA `m=4`, HCA `m'=8`, `index_topk=2`)
on a 16-token input so the full per-layer-type mask fits on screen. Green = the query/key diagonal in the
sliding section, dark = a visible standard KV position, light = masked, amber = a compressor / indexer slot
the query is allowed to attend to. Columns past the dashed line are appended by the compressor via
`cat([sliding_causal_mask, block_bias], dim=-1)`.

**Sliding-only layer (`"sliding_attention"`).** No compressor, no right-padding — the mask is the plain
sliding-window-causal mask of shape `[S, S]` (window = 8). For `i ≥ window` the lower-left triangle is cut
off, recovering the local-only attention pattern.

<img alt="DeepSeek-V4 sliding attention mask" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deepseek_v4/deepseek_v4_mask_layer0_sliding_attention.svg" />

**CSA layer (`"compressed_sparse_attention"`).** The compressor flattens its per-query gathered output to
`[B, 1, S·k, D]` and right-pads the mask by `S·k` columns. For query `t`, only the `k` slots at columns
`[S + t·k, S + (t+1)·k)` carry the indexer's picks; all other compressor columns are `-inf`. Queries before
the first window has closed (`t < m − 1`) get nothing — the indexer's `-1` sentinel propagates straight to
the mask. As `t` grows, more compressed entries are ready and the indexer can fill all `k` slots.

<img alt="DeepSeek-V4 CSA attention mask" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deepseek_v4/deepseek_v4_mask_layer1_compressed_sparse_attention.svg" />

**HCA layer (`"heavily_compressed_attention"`).** No indexer — every cached compressed entry is potentially
visible. Right-padded by `T_total = entry_count["compressor"]` columns. Query `t` may only see entry `w` once
its source window has closed, i.e. `w < (t + 1) // m`. With `m=8` here, entries 0 (covers positions `0..7`)
and 1 (covers `8..15`) only become visible at `t ≥ 7` and `t ≥ 15` respectively.

<img alt="DeepSeek-V4 HCA attention mask" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deepseek_v4/deepseek_v4_mask_layer2_heavily_compressed_attention.svg" />

These diagrams are reproducible end-to-end via:

```bash
python docs/source/en/imgs/deepseek_v4/visualize_attention_masks.py \
    --svg docs/source/en/imgs/deepseek_v4
```

The script runs a forward pass on this tiny config, wraps each attention layer to capture the exact
post-`cat([attention_mask, block_bias])` mask, remaps CSA's `[S, S·k]` flat-slot mask back to a
`[S, T_entries]` entry-visibility view (so each `C_w` column is a compressed *entry*, not a gather slot),
and writes the three SVGs above. It also prints an ANSI grid to stdout for quick terminal inspection and
dumps the indexer's per-query top-k picks so warm-up sentinels and pick choices are auditable.

### Cache layers

Each non-sliding attention block needs to thread compressor / indexer state across forward calls. V4 ships two
cache layer types that auto-register with `LAYER_TYPE_CACHE_MAPPING`:

* `DeepseekV4HCACache`: sliding-window K=V + HCA compressor buffer / pool / count (no overlap, no indexer).
* `DeepseekV4CSACache`: sliding-window K=V + CSA compressor (with overlap state) + parallel indexer
  buffer / pool / count / overlap at `index_head_dim`.

`DynamicCache(config=…)` builds the right cache layer per `config.layer_types[i]`.

## DeepseekV4Config

[[autodoc]] DeepseekV4Config

## DeepseekV4Model

[[autodoc]] DeepseekV4Model
    - forward

## DeepseekV4ForCausalLM

[[autodoc]] DeepseekV4ForCausalLM
    - forward
