<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->

# VaultGemma

## Overview

[VaultGemma](https://google.com) is a text-only decoder model based on the
[Gemma 2](https://huggingface.co/docs/transformers/en/model_doc/gemma2)
[family](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) of models, trained from
scratch with sequence-level differential privacy (DP). VaultGemma model is available as a pretrained model with 1B
parameters that uses a 1024 token sequence length.

VaultGemma was trained on the same training mixture as the Gemma 2 models, consisting of a number of documents of
varying lengths. Additionally, it is trained using
[DP stochastic gradient descent (DP-SGD)](https://arxiv.org/abs/1607.00133) and provides a
(ε ≤ 2.0, δ ≤ 1.1e-10)-sequence-level DP guarantee, where a sequence consists of 1024 consecutive tokens extracted from
heterogeneous data sources. Specifically, the privacy unit of the guarantee is for the sequences after sampling and
packing of the mixture.

> [!TIP]
> Click on the VaultGemma models in the right sidebar for more examples of how to apply VaultGemma to different language tasks.

The example below demonstrates how to chat with the model with [`Pipeline`], the [`AutoModel`] class, or from the
command line.

<hfoptions id="usage">
<hfoption id="Pipeline">


```python
import torch
from transformers import pipeline

pipe = pipeline(
    "text2text-generation",
    model="google/vaultgemma-1b",
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Tell me an unknown interesting biology fact about the brain."},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

pipe(prompt, max_new_tokens=32)
```

</hfoption>
<hfoption id="AutoModel">

```python
# pip install accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/vaultgemma-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="auto")

messages = [
    {"role": "user", "content": "Tell me an unknown interesting biology fact about the brain."},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True
).to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
<hfoption id="transformers CLI">

```
echo -e "Write me a poem about Machine Learning. Answer:" | transformers run --task text2text-generation --model google/vaultgemma-1b-pt --device 0
```

</hfoption>
</hfoptions>

## VaultGemmaConfig

[[autodoc]] VaultGemmaConfig

## VaultGemmaModel

[[autodoc]] VaultGemmaModel
    - forward

## VaultGemmaForCausalLM

[[autodoc]] VaultGemmaForCausalLM
