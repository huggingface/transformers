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

Qwen4-Exp extends the Qwen3.5 hybrid text and multimodal architecture with learned hyper-connections and positional
lexical embeddings (PLE). Each decoder block mixes several residual streams before attention and the sparse MoE
block, then learns how strongly to inject each block output back into those streams. Selected layers also add PLE
features built from hashed token n-grams and a dilated depthwise convolution.

The text backbone retains Qwen3.5's mixture of Gated DeltaNet linear-attention layers and gated full-attention layers.
Every decoder layer uses routed experts together with a shared expert; Qwen4-Exp does not provide dense decoder blocks or
a dense fallback. The multimodal model reuses the Qwen3.5 vision encoder and multimodal rotary position encoding.

## Usage tips

- `ple_layer_ids` contains one-indexed decoder layer numbers. PLE needs the original token ids even when the language
  model receives `inputs_embeds`; pass them as `ple_input_ids` in that case.
- PLE maintains both n-gram context and dilated-convolution state during cached generation. Pass a
  [`DynamicCache`] initialized with the text config when constructing a cache explicitly.
- `split_ngram_parts` controls the original checkpoint layout of each large PLE embedding table. Transformers
  concatenates those shards along the vocabulary dimension into one runtime embedding weight, and splits it back when
  saving in the original format.
- `hc_count` is the number of residual streams and `hc_lowrank` controls the rank of the learned input mixer.
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
