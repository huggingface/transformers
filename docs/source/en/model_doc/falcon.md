<!--Copyright 2023 The HuggingFace Team. All rights reserved.

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

# Falcon

[Falcon](https://huggingface.co/papers/2311.16867) is a family of large language models, available in 7B, 40B, and 180B parameters, as pretrained and instruction tuned variants. This model focuses on scaling pretraining over three categories, performance, data, and hardware. Falcon uses multigroup attention to significantly reduce inference memory requirements and rotary positional embeddings (RoPE). These models are pretrained on [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), a high-quality and deduplicated 5T token dataset.

You can find all the original Falcon checkpoints under the [Falcon](https://huggingface.co/collections/tiiuae/falcon-64fb432660017eeec9837b5a) collection.

> [!TIP]
> Click on the Falcon models in the right sidebar for more examples of how to apply Falcon to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="tiiuae/falcon-7b-instruct",
    dtype=torch.bfloat16,
    device=0
)
pipeline(
    "Write a short poem about coding",
    max_length=100,
    do_sample=True,
    temperature=0.7
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

input_ids = tokenizer("Write a short poem about coding", return_tensors="pt").to("cuda")

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
# pip install -U flash-attn --no-build-isolation
transformers chat tiiuae/falcon-7b-instruct --dtype auto --attn_implementation flash_attention_2 --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
)

inputs = tokenizer("In quantum physics, entanglement means", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Notes

- If you're upgrading from an older custom code checkpoint, remember to convert it to the official Transformers format for better stability and performance using the conversion script located in the [Falcon model directory](https://github.com/huggingface/transformers/tree/main/src/transformers/models/falcon).

   ```bash
   python convert_custom_code_checkpoint.py --checkpoint_dir my_model
   ```

## FalconConfig

[[autodoc]] FalconConfig
    - all

## FalconModel

[[autodoc]] FalconModel
    - forward

## FalconForCausalLM

[[autodoc]] FalconForCausalLM
    - forward

## FalconForSequenceClassification

[[autodoc]] FalconForSequenceClassification
    - forward

## FalconForTokenClassification

[[autodoc]] FalconForTokenClassification
    - forward

## FalconForQuestionAnswering

[[autodoc]] FalconForQuestionAnswering
    - forward
