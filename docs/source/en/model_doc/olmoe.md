<!--

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

# OLMoE

[OLMoE](https://huggingface.co/papers/2409.02060) is a sparse Mixture-of-Experts (MoE) language model with 7B parameters but only 1B parameters are used per input token. It has similar inference costs as dense models but trains ~3x faster. OLMoE uses fine-grained routing with 64 small experts in each layer and uses a dropless token-based routing algorithm.

You can find all the original OLMoE checkpoints under the [OLMoE](https://huggingface.co/collections/allenai/olmoe-november-2024-66cf678c047657a30c8cd3da) collection.

> [!TIP]
> This model was contributed by [Muennighoff](https://hf.co/Muennighoff).
>
> Click on the OLMoE models in the right sidebar for more examples of how to apply OLMoE to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="allenai/OLMoE-1B-7B-0125",
    dtype=torch.float16,
    device=0,
)

result = pipe("Dionysus is the god of")
print(result)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", attn_implementation="sdpa", dtype="auto", device_map="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

inputs = tokenizer("Bitcoin is", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
output = model.generate(**inputs, max_length=64)
print(tokenizer.decode(output[0]))
```

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.
The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", attn_implementation="sdpa", dtype="auto", device_map="auto", quantization_config=quantization_config).to(device)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

inputs = tokenizer("Bitcoin is", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
output = model.generate(**inputs, max_length=64)
print(tokenizer.decode(output[0]))
```

## OlmoeConfig

[[autodoc]] OlmoeConfig

## OlmoeModel

[[autodoc]] OlmoeModel
    - forward

## OlmoeForCausalLM

[[autodoc]] OlmoeForCausalLM
    - forward
