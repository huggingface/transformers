<!--Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-26.*

# Qwen4-Exp

Qwen4-Exp builds on Qwen3.5's hybrid text and multimodal architecture with three key components: GatedResidual (GR), Qwen Sparse Attention (QSA), and Per-Layer Embedding (PLE).

GR is a Qwen-developed residual architecture that combines Hyper-Connection with GatedNorm. It mixes multiple residual streams with fine-grained elementwise gating before each attention and Mixture-of-Experts (MoE) block, then controls how much of the block output is injected back into each stream.

QSA uses multiple query heads to score compressed key blocks, selects the most relevant contiguous token blocks, and keeps the incomplete trailing block uncompressed. This block-level selection reduces indexing overhead and improves memory locality for long sequences. Combined with Gated DeltaNet, QSA makes Qwen4-Exp the first hybrid architecture to integrate linear and sparse attention, substantially improving inference efficiency for long-context workloads.

PLE enriches selected decoder layers with layer-specific lexical features derived from hashed token n-grams and a dilated depthwise convolution.

## Usage tips

- `ple_layer_ids` uses one-based decoder layer indices. When PLE is enabled and the model receives `inputs_embeds`, pass the original token ids through `ple_input_ids`. If `input_ids` are provided, the model uses them for PLE automatically.
- During cached generation, PLE maintains both n-gram context and dilated-convolution state. [`DynamicCache`] is the recommended default, and [`StaticCache`] is also supported. Cache cropping is not supported when PLE or QSA is enabled. With cache offloading, GatedDeltaNet, PLE, and QSA indexer states remain on device while attention key/value states are offloaded.
- `split_ngram_parts` controls the logical checkpoint shards for each large PLE n-gram embedding table. Transformers concatenates these shards along the vocabulary dimension into one runtime weight. The default `save_pretrained(save_original_format=True)` writes the configured original sharded layout.
- `hc_count` sets the number of residual streams, and `hc_lowrank` sets the rank of the learned GR input mixer.
- Providing the complete set of QSA indexer fields enables sparse token selection on full-attention layers. Eager and SDPA attention are supported; other requested backends fall back to eager. Automatic generation compilation is disabled because token selection is data-dependent.
- `tp_plan="auto"` supports attention, MoE, GatedDeltaNet, QSA, GR, and vocabulary-row sharding of PLE tables. Custom configurations must keep sharded attention heads, QSA and GatedDeltaNet projection dimensions, GR stream width, expert dimensions, and the padded PLE vocabulary divisible by the TP size. The replicated low-rank GR output adds no divisibility constraint.
- FSDP2 shards token embeddings and decoder layers while keeping the final GR mixer gathered. TP and FSDP cannot currently be combined, and no pipeline-parallel plan is provided.
- Use [`Qwen4ExpForCausalLM`] with [`Qwen4ExpTextConfig`] for text-only generation. Use [`Qwen4ExpForConditionalGeneration`] with [`Qwen4ExpConfig`] for multimodal inputs.

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
