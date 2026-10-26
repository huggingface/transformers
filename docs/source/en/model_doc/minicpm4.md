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
*This model was published in HF papers on 2025-06-09 and contributed to Hugging Face Transformers on 2026-10-21.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# MiniCPM4

[MiniCPM4](https://huggingface.co/papers/2506.07900) is a family of decoder-only language models from OpenBMB designed for efficient on-device inference. It uses grouped-query attention, SwiGLU feed-forward layers, RMS normalization, and LongRoPE. MiniCPM4 also scales its embeddings, residual branches, and language-model logits to keep training stable across model sizes.

MiniCPM4 uses the `minicpm4` model type and the `MiniCPM4` class prefix in Transformers. For supported checkpoints saved with the legacy `minicpm` model type, [`MiniCPMConfig`] redirects loading to [`MiniCPM4Config`]. Checkpoints without a `model_type`, such as MiniCPM4-0.5B, can be loaded directly with [`MiniCPM4ForCausalLM`] while their Hub configuration metadata is being updated.

The original code can be found at [OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM).

This model was contributed by [Sushmanth Reddy](https://github.com/sushmanthreddy).

The following checkpoints use the same native [`MiniCPM4ForCausalLM`] implementation:

- [`openbmb/MiniCPM4-0.5B`](https://huggingface.co/openbmb/MiniCPM4-0.5B)
- [`openbmb/MiniCPM4-8B`](https://huggingface.co/openbmb/MiniCPM4-8B)
- [`openbmb/MiniCPM4.1-8B`](https://huggingface.co/openbmb/MiniCPM4.1-8B), an updated 8B checkpoint with a 65,536-token context window and hybrid reasoning support

> [!NOTE]
> This implementation supports dense attention. The optional InfLLM-v2 sparse inference mode described by OpenBMB is not supported.

The example below loads MiniCPM4 without custom model code.

```python
from transformers import AutoTokenizer, MiniCPM4ForCausalLM


checkpoint = "openbmb/MiniCPM4-0.5B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = MiniCPM4ForCausalLM.from_pretrained(checkpoint, dtype="auto", device_map="auto")

inputs = tokenizer("Large language models are", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## MiniCPM4Config

[[autodoc]] MiniCPM4Config

## MiniCPMConfig

[[autodoc]] MiniCPMConfig

## MiniCPM4Model

[[autodoc]] MiniCPM4Model
    - forward

## MiniCPM4ForCausalLM

[[autodoc]] MiniCPM4ForCausalLM
    - forward

## MiniCPM4ForSequenceClassification

[[autodoc]] MiniCPM4ForSequenceClassification
    - forward

## MiniCPM4PreTrainedModel

[[autodoc]] MiniCPM4PreTrainedModel
