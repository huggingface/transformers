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
*This model was released on 2025-06-30 and added to Hugging Face Transformers on 2025-07-21.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Ernie 4.5 Moe

## Overview

The Ernie 4.5 Moe model was released in the [Ernie 4.5 Model Family](https://ernie.baidu.com/blog/posts/ernie4.5/) release by baidu.
This family of models contains multiple different architectures and model sizes. This model in specific targets the base text
model with mixture of experts (moe) - one with 21B total, 3B active parameters and another one with 300B total, 47B active parameters.
It uses the standard [Llama](./llama) at its core combined with a specialized MoE based on [Mixtral](./mixtral) with additional shared
experts.

Other models from the family can be found at [Ernie 4.5](./ernie4_5).

<div class="flex justify-center">
    <img src="https://ernie.baidu.com/blog/posts/ernie4.5/overview.png"/>
</div>


## Usage Tips

### Generate text

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "baidu/ERNIE-4.5-21B-A3B-PT"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# prepare the model input
inputs = tokenizer("Hey, are you conscious? Can you talk to me?", return_tensors="pt")
prompt = "Hey, are you conscious? Can you talk to me?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# decode the generated ids
generate_text = tokenizer.decode(output_ids, skip_special_tokens=True)
```

### Distributed Generation with Tensor Parallelism

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "baidu/ERNIE-4.5-21B-A3B-PT"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    tp_plan="auto",
)

# prepare the model input
inputs = tokenizer("Hey, are you conscious? Can you talk to me?", return_tensors="pt")
prompt = "Hey, are you conscious? Can you talk to me?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# decode the generated ids
generate_text = tokenizer.decode(output_ids, skip_special_tokens=True)
```

### Quantization with Bitsandbytes

```python
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "baidu/ERNIE-4.5-21B-A3B-PT"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
)

# prepare the model input
inputs = tokenizer("Hey, are you conscious? Can you talk to me?", return_tensors="pt")
prompt = "Hey, are you conscious? Can you talk to me?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# decode the generated ids
generate_text = tokenizer.decode(output_ids, skip_special_tokens=True)
```

This model was contributed by [Anton Vlasjuk](https://huggingface.co/AntonV).
The original code can be found [here](https://github.com/PaddlePaddle/ERNIE).


## Ernie4_5_MoeConfig

[[autodoc]] Ernie4_5_MoeConfig

## Ernie4_5_MoeModel

[[autodoc]] Ernie4_5_MoeModel
    - forward

## Ernie4_5_MoeForCausalLM

[[autodoc]] Ernie4_5_MoeForCausalLM
    - forward
    - generate
