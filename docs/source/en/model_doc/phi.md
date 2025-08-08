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
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Phi

[Phi](https://huggingface.co/papers/2306.11644) is a 1.3B parameter transformer model optimized for Python code generation. It focuses on "textbook-quality" training data of code examples, exercises and synthetic Python problems rather than scaling the model size or compute.

You can find all the original Phi checkpoints under the [Phi-1](https://huggingface.co/collections/microsoft/phi-1-6626e29134744e94e222d572) collection.

> [!TIP]
> Click on the Phi models in the right sidebar for more examples of how to apply Phi to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="microsoft/phi-1.5", device=0, dtype=torch.bfloat16)
pipeline("pipeline('''def print_prime(n): """ Print all primes between 1 and n"""''')")

```

</hfoption>

<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")

input_ids = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "'''def print_prime(n): """ Print all primes between 1 and n"""'''" | transformers run --task text-classification --model microsoft/phi-1.5 --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) to only quantize the weights to 4-bits.

```py
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1", dtype=torch.float16, device_map="auto", attn_implementation="sdpa", quantization_config=bnb_config)

input_ids = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- If you're using Transformers < 4.37.0.dev, set `trust_remote_code=True` in [`~AutoModel.from_pretrained`]. Otherwise, make sure you update Transformers to the latest stable version.

    ```py
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-1",
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa")

    input_ids = tokenizer('''def print_prime(n):
       """
       Print all primes between 1 and n
       """''', return_tensors="pt").to("cuda")

    output = model.generate(**input_ids, cache_implementation="static")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    ```

## PhiConfig

[[autodoc]] PhiConfig

## PhiModel

[[autodoc]] PhiModel
    - forward

## PhiForCausalLM

[[autodoc]] PhiForCausalLM
    - forward
    - generate

## PhiForSequenceClassification

[[autodoc]] PhiForSequenceClassification
    - forward

## PhiForTokenClassification

[[autodoc]] PhiForTokenClassification
    - forward
