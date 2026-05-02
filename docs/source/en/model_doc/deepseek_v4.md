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
