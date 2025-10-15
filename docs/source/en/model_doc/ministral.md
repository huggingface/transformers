<!--Copyright 2024 Mistral AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-09-11.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Ministral

[Ministral](https://mistral.ai/news/ministraux) models are optimized for edge and on-device computing. Both support up to 128k context length (32k on vLLM), with the 8B model using an interleaved sliding-window attention mechanism for faster, memory-efficient inference. They outperform peers like Llama and Gemma across benchmarks, even surpassing Mistral 7B despite smaller size. Designed for privacy-first, low-latency applications such as offline assistants, robotics, and local analytics, the models are available via API, with weights for research and quantization support for deployment

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mistralai/Ministral-8B-Instruct-2410", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
model = AutoModelForCausalLM.from_pretrained("mistralai/Ministral-8B-Instruct-2410", dtype="auto",)

messages = [
    {"role": "user", "content": "How do plants create energy?"},
    {"role": "assistant", "content": "Plants create energy through a process known as photosynthesis. Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts."},
    {"role": "user", "content": "How much light should a plant receive?"}
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])
```

</hfoption>
</hfoptions>

## MinistralConfig

[[autodoc]] MinistralConfig

## MinistralModel

[[autodoc]] MinistralModel
    - forward

## MinistralForCausalLM

[[autodoc]] MinistralForCausalLM
    - forward

## MinistralForSequenceClassification

[[autodoc]] MinistralForSequenceClassification
    - forward

## MinistralForTokenClassification

[[autodoc]] MinistralForTokenClassification
    - forward

## MinistralForQuestionAnswering

[[autodoc]] MinistralForQuestionAnswering
- forward

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mistralai/Ministral-8B-Instruct-2410", dtype="auto")
pipeline("The future of artificial intelligence is")
```