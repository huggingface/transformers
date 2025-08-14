<!--Copyright 2025 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# EXAONE 4

## Overview

**[EXAONE 4.0](https://github.com/LG-AI-EXAONE/EXAONE-4.0)** model is the language model, which integrates a **Non-reasoning mode** and **Reasoning mode** to achieve both the excellent usability of [EXAONE 3.5](https://github.com/LG-AI-EXAONE/EXAONE-3.5) and the advanced reasoning abilities of [EXAONE Deep](https://github.com/LG-AI-EXAONE/EXAONE-Deep). To pave the way for the agentic AI era, EXAONE 4.0 incorporates essential features such as agentic tool use, and its multilingual capabilities are extended
to support Spanish in addition to English and Korean. 

The EXAONE 4.0 model series consists of two sizes: a mid-size **32B** model optimized for high performance, and a small-size **1.2B** model designed for on-device applications.

In the EXAONE 4.0 architecture, we apply new architectural changes compared to previous EXAONE models as below:

1. **Hybrid Attention**: For the 32B model, we adopt hybrid attention scheme, which combines *Local attention (sliding window attention)* with *Global attention (full attention)* in a 3:1 ratio. We do not use RoPE (Rotary Positional Embedding) for global attention for better global context understanding.
2. **QK-Reorder-Norm**: We reorder the LayerNorm position from the traditional Pre-LN scheme by applying LayerNorm directly to the attention and MLP outputs, and we add RMS normalization right after the Q and K projection. It helps yield better performance on downstream tasks despite consuming more computation.

For more details, please refer to our [technical report](https://arxiv.org/abs/2507.11407), [HuggingFace paper](https://huggingface.co/papers/2507.11407), [blog](https://www.lgresearch.ai/blog/view?seq=576), and [GitHub](https://github.com/LG-AI-EXAONE/EXAONE-4.0).

All model weights including quantized versions are available at [Huggingface Collections](https://huggingface.co/collections/LGAI-EXAONE/exaone-40-686b2e0069800c835ed48375).


## Model Details

### Model Specifications

| Model Configuration | 32B | 1.2B |
|:-------------------|:-----:|:------:|
| d_model | 5,120 | 2,048 |
| Number of layers | 64 | 30 |
| Normalization | QK-Reorder-LN | QK-Reorder-LN |
| Non-linearity | SwiGLU | SwiGLU |
| Feedforward dimension | 27,392 | 4,096 |
| Attention type | Hybrid (3:1 Local-Global) | Global |
| Head type | GQA | GQA |
| Number of heads | 40 | 32 |
| Number of KV heads | 8 | 8 |
| Head size | 128 | 64 |
| Max sequence length | 131,072 | 65,536 |
| RoPE theta | 1,000,000 | 1,000,000 |
| Tokenizer | BBPE | BBPE |
| Vocab size | 102,400 | 102,400 |
| Tied word embedding | False | True |
| Knowledge cut-off | Nov. 2024 | Nov. 2024 |


## Usage tips

### Non-reasoning mode

For general use, you can use the EXAONE 4.0 models with the following example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-4.0-32B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# choose your prompt
prompt = "Explain how wonderful you are"
prompt = "Explica lo increíble que eres"
prompt = "너가 얼마나 대단한지 설명해 봐"

messages = [
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=False,
)
print(tokenizer.decode(output[0]))
```

### Reasoning mode

The EXAONE 4.0 models have reasoning capabilities for handling complex problems. You can activate reasoning mode by using the `enable_thinking=True` argument with the tokenizer, which opens a reasoning block that starts with `<think>` tag without closing it.

```python
messages = [
    {"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=True,
    temperature=0.6,
    top_p=0.95
)
print(tokenizer.decode(output[0]))
```

> [!IMPORTANT]
> The model generation with reasoning mode can be affected sensitively by sampling parameters, so please refer to the [Usage Guideline](https://github.com/LG-AI-EXAONE/EXAONE-4.0#usage-guideline) on official GitHub page for better quality.

### Agentic tool use

The EXAONE 4.0 models can be used as agents with their tool calling capabilities. You can provide tool schemas to the model for effective tool calling.

```python
import random

def roll_dice(max_num: int):
    return random.randint(1, max_num)

tools = [
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "Roll a dice with the number 1 to N. User can select the number N.",
            "parameters": {
                "type": "object",
                "required": ["max_num"],
                "properties": {
                    "max_num": {
                        "type": "int",
                        "description": "Max number of the dice"
                    }
                }
            }
        }
    }
]

messages = [
    {"role": "user", "content": "Roll D6 dice twice!"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    tools=tools,
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
)
print(tokenizer.decode(output[0]))
```

## Exaone4Config

[[autodoc]] Exaone4Config

## Exaone4Model

[[autodoc]] Exaone4Model
    - forward

## Exaone4ForCausalLM

[[autodoc]] Exaone4ForCausalLM
    - forward

## Exaone4ForSequenceClassification

[[autodoc]] Exaone4ForSequenceClassification
    - forward

## Exaone4ForTokenClassification

[[autodoc]] Exaone4ForTokenClassification
    - forward

## Exaone4ForQuestionAnswering

[[autodoc]] Exaone4ForQuestionAnswering
    - forward