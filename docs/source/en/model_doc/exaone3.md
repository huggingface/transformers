<!--Copyright 2024 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# EXAONE v3.0

_EXAONE stands for **EX**pert **A**I for Every**ONE**_

## Overview
EXAONE is the family of Large Language Models (LLMs) and Large Multimodal Models (LMMs) developed by LG AI Research. EXAONE stands for EXpert AI for EveryONE, a vision that LG is committed to realizing.
 
## Model Details
`EXAONE-3.0-7.8B-Instruct` is a pre-trained and instruction-tuned bilingual (English and Korean) generative model with 7.8 billion parameters. The model was pre-trained with 8T curated tokens and post-trained with supervised fine-tuning and direct preference optimization. It demonstrates highly competitive benchmark performance against other state-of-the-art open models of similar size.

## Usage tips

`EXAONE-v3.0-7.8B-Instruct` can be found on the [Huggingface Hub](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/)

We recommend to use `transformers >= 4.43.0`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

# Choose your prompt
prompt = "Explain who you are"  # English example
prompt = "너의 소원을 말해봐"   # Korean example

messages = [
    {"role": "system", 
     "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to("cuda"),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128
)
print(tokenizer.decode(output[0]))
```


## ExaoneConfig
[[autodoc]] ExaoneConfig

## ExaoneModel
[[autodoc]] ExaoneModel
    - forward

## ExaoneForCausalLM
[[autodoc]] ExaoneForCausalLM
    - forward

## ExaoneForSequenceClassification
[[autodoc]] ExaoneForSequenceClassification
    - forward

## ExaoneForQuestionAnswering
[[autodoc]] ExaoneForQuestionAnswering
    - forward
