<!--Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-05.*

# Qwen4-Exp

Qwen4-Exp builds on Qwen3.5's hybrid text and multimodal architecture with three key innovations: GatedResidual (GR), Per-Layer Embedding (PLE), and Qwen Sparse Attention (QSA).

GR is a Qwen-developed residual architecture that combines Hyper-Connection with GatedNorm. It mixes multiple residual streams with fine-grained elementwise gating before each attention and Mixture-of-Experts (MoE) block, then controls how much of the block output is injected back into each stream. PLE augments selected decoder layers with layer-specific lexical features derived from hashed token n-grams and a dilated depthwise convolution.

QSA uses multi-head query representations to score compressed key blocks, selects relevant contiguous token blocks, and keeps the incomplete trailing block uncompressed. This block-level design reduces selection overhead and improves memory locality over long sequences. Together, Gated DeltaNet and QSA make Qwen4-Exp the first hybrid architecture to combine linear attention with sparse attention, substantially improving inference efficiency for long-context workloads.

## Usage tips

- `ple_layer_ids` contains one-indexed decoder layer numbers. PLE needs the original token ids even when the language
  model receives `inputs_embeds`; pass them as `ple_input_ids` in that case.
- PLE maintains both n-gram context and dilated-convolution state during cached generation. [`DynamicCache`] is the
  recommended default. [`StaticCache`] is also supported. Offloaded caches are not supported when PLE or QSA is
  enabled.
- `split_ngram_parts` controls the sharded checkpoint layout of each large PLE embedding table. Transformers
  concatenates those shards along the vocabulary dimension into one runtime embedding weight, while `save_pretrained`
  restores the configured sharded layout.
- `hc_count` is the number of residual streams and `hc_lowrank` controls the rank of the learned input mixer.
- Setting the QSA indexer fields enables sparse token selection on full-attention layers. SDPA and eager attention are
  supported, other requested attention backends fall back to eager, and automatic generation compilation is disabled
  because selection is data-dependent.
- `tp_plan="auto"` supports the inherited attention, MoE, and GatedDeltaNet rules together with QSA, hyper-connections,
  and vocabulary-row sharding of PLE tables. Custom configurations must keep the sharded attention heads, QSA and
  GatedDeltaNet projection dimensions, hyper-connection stream width, expert dimensions, and padded PLE vocabulary
  divisible by the TP size. The replicated hyper-connection low-rank output does not add a divisibility constraint.
- FSDP2 shards token embeddings and decoder layers while keeping the final hyper-connection mixer gathered. TP and FSDP
  cannot currently be combined, and a pipeline-parallel plan is not provided.
- Use [`Qwen4ExpForCausalLM`] with [`Qwen4ExpTextConfig`] for text-only generation. Use
  [`Qwen4ExpForConditionalGeneration`] with [`Qwen4ExpConfig`] for multimodal inputs.

## Qwen4ExpConfig

[[autodoc]] Qwen4ExpConfig

## Qwen4ExpTextConfig

[[autodoc]] Qwen4ExpTextConfig

## Qwen4ExpVisionConfig

[[autodoc]] Qwen4ExpVisionConfig

## Qwen4ExpVisionModel

[[autodoc]] Qwen4ExpVisionModel
    - forward

## Qwen4ExpTextModel

[[autodoc]] Qwen4ExpTextModel
    - forward

## Qwen4ExpModel

[[autodoc]] Qwen4ExpModel
    - forward

## Qwen4ExpForCausalLM

[[autodoc]] Qwen4ExpForCausalLM
    - forward

## Qwen4ExpForConditionalGeneration

[[autodoc]] Qwen4ExpForConditionalGeneration
    - forward

## Qwen4ExpForSequenceClassification

[[autodoc]] Qwen4ExpForSequenceClassification
    - forward

## Qwen4ExpTextForSequenceClassification

[[autodoc]] Qwen4ExpTextForSequenceClassification
    - forward

## Qwen4ExpForTokenClassification

[[autodoc]] Qwen4ExpForTokenClassification
    - forward
