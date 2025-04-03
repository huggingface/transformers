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

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# FalconMamba

[FalconMamba](https://huggingface.co/papers/2410.05355) is a family of large language models based on the State Space Model (SSM) architecture, available in 7B parameter size as pretrained and instruction-tuned variants. This model implements a pure Mamba design that focuses on computational efficiency while maintaining strong performance. FalconMamba uses linear-time selective state space models and rotary positional embeddings (RoPE). The models are pretrained on a diverse 5.8 trillion token dataset including [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), technical content, code, and mathematical data.

You can find the official FalconMamba checkpoints in the [TII UAE collection](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a).

> [!TIP]
> Click on the FalconMamba models in the right sidebar for more examples of how to apply FalconMamba to different language tasks.

The examples below demonstrate how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    "text-generation", 
    model="tiiuae/falcon-mamba-7b-instruct",
    torch_dtype=torch.bfloat16,
    device=0
)
pipeline(
    "Explain the difference between transformers and SSMs",
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

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-mamba-7b-instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

input_ids = tokenizer("Explain the difference between transformers and SSMs", return_tensors="pt").to("cuda")

output = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli chat --model_name_or_path tiiuae/falcon-mamba-7b-instruct --torch_dtype auto --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 4-bits.

```python
import torch
from transformers import AutoTokenizer, FalconMambaForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
model = FalconMambaForCausalLM.from_pretrained(
    "tiiuae/falcon-mamba-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
)

inputs = tokenizer("Explain the concept of state space models in simple terms", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Notes

- FalconMamba is based on the Mamba architecture. The same [tips and best practices](./mamba) for Mamba models are relevant here.
- The architecture is compatible with `torch.compile` for faster generation via `model = torch.compile(model)`.

## FalconMambaConfig

[[autodoc]] FalconMambaConfig

## FalconMambaModel

[[autodoc]] FalconMambaModel
    - forward

## FalconMambaLMHeadModel

[[autodoc]] FalconMambaForCausalLM
    - forward
