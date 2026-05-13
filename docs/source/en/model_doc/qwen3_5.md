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
*This model was released on 2026-01-01 and added to Hugging Face Transformers on 2026-02-09.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Qwen3.5

[Qwen3.5](https://qwen.ai/blog?id=qwen3.5) is Qwen's natively multimodal foundation model family, trained from scratch on interleaved text, image, and video tokens. It uses a 3:1 hybrid attention stack — three Gated DeltaNet (linear attention) layers for every one Gated Attention (full attention) layer — so long context and vision tokens can be served without paying full quadratic cost on every block.

This page covers the dense Qwen3.5 and Qwen3.6 variants (Qwen/Qwen3.5-9B, Qwen/Qwen3.5-27B, Qwen/Qwen3.6-27B). Qwen3.6 checkpoints share the same architecture and `model_type` as Qwen3.5 and are loaded with the same classes. For the sparse mixture-of-experts variants see [Qwen3.5 MoE](./qwen3_5_moe). The text backbone reuses Qwen3-Next's linear-attention decoder with a three-component multimodal RoPE; the vision tower reuses the Qwen3-VL encoder.

You can find all the official Qwen3.5 checkpoints under the [Qwen](https://huggingface.co/Qwen) organization.

## Quickstart

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen3.5-9B",
    device_map="auto",
)
print(pipe("The capital of France is", max_new_tokens=20)[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
model = Qwen3_5ForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B",
    device_map="auto",
)

inputs = tokenizer("Hey, are you conscious? Can you talk to me?", return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips and notes

- Layers are hybrid: [`Qwen3_5TextConfig`]'s `layer_types` is a per-layer list of `"linear_attention"` or `"full_attention"` that encodes the 3:1 Gated DeltaNet / Gated Attention stack. The DeltaNet path (`Qwen3NextGatedDeltaNet`) needs the optional `causal_conv1d` (from [Dao-AILab](https://github.com/Dao-AILab/causal-conv1d)) and `fla` packages for its fast kernels — without them, the model silently falls back to slower and more memory hungry PyTorch ops.
- Multimodal RoPE splits the head dimension into three components (temporal, height, width) via `mrope_section` on the text config. If you replace the rotary module, preserve this split or position encodings for image and video tokens will be misaligned.
- Use [`Qwen3_5ForCausalLM`] for text-only generation with [`Qwen3_5TextConfig`]; use [`Qwen3_5ForConditionalGeneration`] with the full [`Qwen3_5Config`] and a processor ([`~AutoProcessor.from_pretrained`]) to feed interleaved image/video + text via [`~ProcessorMixin.apply_chat_template`].

## Qwen3_5Config

[[autodoc]] Qwen3_5Config

## Qwen3_5TextConfig

[[autodoc]] Qwen3_5TextConfig

## Qwen3_5VisionConfig

[[autodoc]] Qwen3_5VisionConfig

## Qwen3_5Tokenizer

[[autodoc]] Qwen3_5Tokenizer

## Qwen3_5VisionModel

[[autodoc]] Qwen3_5VisionModel
    - forward

## Qwen3_5TextModel

[[autodoc]] Qwen3_5TextModel
    - forward

## Qwen3_5Model

[[autodoc]] Qwen3_5Model
    - forward

## Qwen3_5ForCausalLM

[[autodoc]] Qwen3_5ForCausalLM
    - forward

## Qwen3_5ForConditionalGeneration

[[autodoc]] Qwen3_5ForConditionalGeneration
    - forward
<<<<<<< HEAD

## Qwen3_5ForSequenceClassification

[[autodoc]] Qwen3_5ForSequenceClassification
    - forward

## Qwen3_5TextForSequenceClassification

[[autodoc]] Qwen3_5TextForSequenceClassification
    - forward

## Qwen3_5ForTokenClassification

[[autodoc]] Qwen3_5ForTokenClassification
    - forward

## Qwen3_5Tokenizer

[[autodoc]] Qwen3_5Tokenizer
=======
>>>>>>> 52b4732861 (docs)
