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

[Qwen4-Exp](https://huggingface.co/Qwen/Qwen4-Exp) extends the Qwen3.5 hybrid text and multimodal architecture with learned hyper-connections and positional
lexical embeddings (PLE). Each decoder block mixes several residual streams before attention and the sparse MoE
block, then learns how strongly to inject each block output back into those streams. Selected layers also add PLE
features built from hashed token n-grams and a dilated depthwise convolution.

The text backbone retains Qwen3.5's mixture of Gated DeltaNet linear-attention layers and gated full-attention layers.
Every decoder layer uses routed experts together with a shared expert; Qwen4-Exp does not provide dense decoder blocks or
a dense fallback. The multimodal model reuses the Qwen3.5 vision encoder and multimodal rotary position encoding.

The official checkpoint and its model card are available under the [Qwen organization](https://huggingface.co/Qwen).

## Quickstart

```py
from transformers import AutoProcessor, Qwen4ExpForConditionalGeneration


model_id = "Qwen/Qwen4-Exp"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen4ExpForConditionalGeneration.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": [{"type": "text", "text": "Explain positional lexical embeddings."}]}]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

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
  and vocabulary-row sharding of PLE tables. The plan is tested with two ranks. Custom configurations must keep the
  sharded attention heads, QSA and GatedDeltaNet projection dimensions, hyper-connection stream width, expert dimensions,
  and padded PLE vocabulary divisible by the TP size. The replicated hyper-connection low-rank output does not add a
  divisibility constraint.
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
