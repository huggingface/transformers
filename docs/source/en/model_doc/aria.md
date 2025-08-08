<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Aria

[Aria](https://huggingface.co/papers/2410.05993) is a multimodal mixture-of-experts (MoE) model. The goal of this model is to open-source a training recipe for creating a multimodal native model from scratch. Aria has 3.9B and 3.5B activated parameters per visual and text token respectively. Text is handled by a MoE decoder and visual inputs are handled by a lightweight visual encoder. It is trained in 4 stages, language pretraining, multimodal pretraining, multimodal long-context pretraining, and multimodal post-training.

You can find all the original Aria checkpoints under the [Aria](https://huggingface.co/rhymes-ai?search_models=aria) organization.

> [!TIP]
> Click on the Aria models in the right sidebar for more examples of how to apply Aria to different multimodal tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    "image-to-text",
    model="rhymes-ai/Aria",
    device=0,
    dtype=torch.bfloat16
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="What is shown in this image?"
)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "rhymes-ai/Aria",
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained("rhymes-ai/Aria")

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
ipnuts = inputs.to(model.device, torch.bfloat16)

output = model.generate(
    **inputs,
    max_new_tokens=15,
    stop_strings=["<|im_end|>"],
    tokenizer=processor.tokenizer,
    do_sample=True,
    temperature=0.9,
)
output_ids = output[0][inputs["input_ids"].shape[1]:]
response = processor.decode(output_ids, skip_special_tokens=True)
print(response)
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.
	
The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4 and the [rhymes-ai/Aria-sequential_mlp](https://huggingface.co/rhymes-ai/Aria-sequential_mlp) checkpoint. This checkpoint replaces grouped GEMM with `torch.nn.Linear` layers for easier quantization.

```py
# pip install torchao
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoProcessor

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = AutoModelForCausalLM.from_pretrained(
    "rhymes-ai/Aria-sequential_mlp",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(
    "rhymes-ai/Aria-sequential_mlp",
)

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device, torch.bfloat16)

output = model.generate(
    **inputs,
    max_new_tokens=15,
    stop_strings=["<|im_end|>"],
    tokenizer=processor.tokenizer,
    do_sample=True,
    temperature=0.9,
)
output_ids = output[0][inputs["input_ids"].shape[1]:]
response = processor.decode(output_ids, skip_special_tokens=True)
print(response)
```


## AriaImageProcessor

[[autodoc]] AriaImageProcessor

## AriaProcessor

[[autodoc]] AriaProcessor

## AriaTextConfig

[[autodoc]] AriaTextConfig

## AriaConfig

[[autodoc]] AriaConfig

## AriaTextModel

[[autodoc]] AriaTextModel

## AriaModel

[[autodoc]] AriaModel

## AriaTextForCausalLM

[[autodoc]] AriaTextForCausalLM

## AriaForConditionalGeneration

[[autodoc]] AriaForConditionalGeneration
    - forward
