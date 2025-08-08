<!--Copyright 2021 The HuggingFace Team. All rights reserved.

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
    <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    <img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=flax&logoColor=white">
  </div>
</div>

# ByT5

[ByT5](https://huggingface.co/papers/2105.13626) is tokenizer-free version of the [T5](./t5) model designed to works directly on raw UTF-8 bytes. This means it can process any language, more robust to noise like typos, and simpler to use because it doesn't require a preprocessing pipeline.

You can find all the original ByT5 checkpoints under the [Google](https://huggingface.co/google?search_models=byt5) organization.

> [!TIP]
> Refer to the [T5](./t5) docs for more examples of how to apply ByT5 to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text2text-generation",
    model="google/byt5-small",
    dtype=torch.float16,
    device=0
)
pipeline("translate English to French: The weather is nice today")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/byt5-small"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/byt5-small",
    dtype=torch.float16,
    device_map="auto"
)

input_ids = tokenizer("summarize: Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy.", return_tensors="pt").to("cuda")

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers-cli">

```bash
echo -e "translate English to French: Life is beautiful." | transformers-cli run --task text2text-generation --model google/byt5-small --device 0
```

</hfoption>
</hfoptions>

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends. 

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```python
# pip install torchao
import torch
from transformers import TorchAoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/byt5-xl",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("google/byt5-xl")
input_ids = tokenizer("translate English to French: The weather is nice today.", return_tensors="pt").to("cuda")

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- It is recommended to use the tokenizer for batched inference and training.
- The example below shows how to use the model without a tokenizer.

    ```python
    import torch
    from transformers import AutoModelForSeq2SeqLM
    
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    
    num_special_tokens = 3
    
    input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens
    labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens
    loss = model(input_ids, labels=labels).loss
    loss.item()
    ```

- ByT5 uses the top byte values (258, 257, etc.) for masking instead of sentinel tokens like `{extra_id_0}`.

    ```python
    # Example: character-level denoising with mask tokens
    input_ids = tokenizer("The dog chases a ball in the park.").input_ids
    masked_input = torch.tensor([input_ids[:8] + [258] + input_ids[14:21] + [257] + input_ids[28:]])
    output = model.generate(masked_input, max_length=100)
    ```

## ByT5Tokenizer

[[autodoc]] ByT5Tokenizer
