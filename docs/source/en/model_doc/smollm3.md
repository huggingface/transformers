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
*This model was released on 2025-07-08 and added to Hugging Face Transformers on 2025-06-25.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SmolLM3

[SmolLM3](https://huggingface.co/blog/smollm3) is a transformer-decoder model based on the Llama architecture, optimized for efficiency and long-context performance. It replaces standard multi-head attention with 4-group Grouped Query Attention (GQA), reducing KV cache size without performance loss, and implements NoPE by removing rotary position embeddings from every fourth layer to enhance long-context capabilities. Training incorporates intra-document masking to prevent cross-document attention and removes weight decay from embedding layers for greater stability. The model was trained on 100B tokens from FineWeb-Edu using a global batch size of 2.36M tokens, sequence length of 4096, AdamW optimizer, and a WSD scheduler over 384 H100 GPUs for 24 days.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="HuggingFaceTB/SmolLM3-3B", dtype="auto")

messages = [
    {"role": "system", "content": "You are a plant biologist."},
    {"role": "user", "content": "How do plants create energy?"},
]
pipeline(messages, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do plants create energy?"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt")
generated_ids = model.generate(
    model_inputs.input_ids,
    cache_implementation="static",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

</hfoption>
</hfoptions>

## SmolLM3Config

[[autodoc]] SmolLM3Config

## SmolLM3Model

[[autodoc]] SmolLM3Model
    - forward

## SmolLM3ForCausalLM

[[autodoc]] SmolLM3ForCausalLM
    - forward

## SmolLM3ForSequenceClassification

[[autodoc]] SmolLM3ForSequenceClassification
    - forward

## SmolLM3ForTokenClassification

[[autodoc]] SmolLM3ForTokenClassification
    - forward

## SmolLM3ForQuestionAnswering

[[autodoc]] SmolLM3ForQuestionAnswering
    - forward
