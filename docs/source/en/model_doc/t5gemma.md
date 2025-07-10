
<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# T5Gemma

T5Gemma (aka encoder-decoder Gemma) was proposed in a [research paper](https://arxiv.org/abs/2504.06225) by Google. It is a family of encoder-decoder large langauge models, developed by adapting pretrained decoder-only models into encoder-decoder. T5Gemma includes pretrained and instruction-tuned variants. The architecture is based on transformer encoder-decoder design following T5, with improvements from Gemma 2: GQA, RoPE, GeGLU activation, RMSNorm, and interleaved local/global attention.

T5Gemma has two groups of model sizes: 1) [Gemma 2](https://ai.google.dev/gemma/docs/core/model_card_2) sizes (2B-2B, 9B-2B, and 9B-9B), which are based on the offical Gemma 2 models (2B and 9B); and 2) [T5](https://arxiv.org/abs/1910.10683) sizes (Small, Base, Large, and XL), where are pretrained under the Gemma 2 framework following T5 configuration. In addition, we also provide a model at ML size (medium large, ~2B in total), which is in-between T5 Large and T5 XL.

The pretrained varaints are trained with two objectives: prefix language modeling with knowledge distillation (PrefixLM) and UL2, separately. We release both variants for each model size. The instruction-turned varaints was post-trained with supervised fine-tuning and reinforcement learning.

The example below demonstrates how to chat with the model with [`Pipeline`] or the [`AutoModel`] class, and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">


```python
import torch
from transformers import pipeline

pipe = pipeline(
    task="text2text-generation",
    model="google/t5gemma-placeholder",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

pipe("Question: Why is the sky blue?\nAnswer:", max_new_tokens=50)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-placeholder")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5gemma-placeholder",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

input_text = "Question: Why is the sky blue?\nAnswer:"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

</hfoption>
<hfoption id="transformers CLI">

```
echo -e "Question: Why is the sky blue? Answer:" | transformers run --task text2text-generation --model google/t5gemma-placeholder --device 0
```

## T5GemmaConfig

[[autodoc]] T5GemmaConfig

## T5GemmaModuleConfig

[[autodoc]] T5GemmaModuleConfig

## T5GemmaModel

[[autodoc]] T5GemmaModel
    - forward

## T5GemmaEncoderModel

[[autodoc]] T5GemmaEncoderModel
    - forward

## T5GemmaForConditionalGeneration

[[autodoc]] T5GemmaForConditionalGeneration
    - forward

## T5GemmaForSequenceClassification

[[autodoc]] T5GemmaForSequenceClassification
    - forward

## T5GemmaForTokenClassification

[[autodoc]] T5GemmaForTokenClassification
    - forward
