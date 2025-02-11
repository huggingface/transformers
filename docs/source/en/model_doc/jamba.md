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

# Jamba

## Overview

Jamba is a state-of-the-art, hybrid SSM-Transformer LLM. It is the first production-scale Mamba implementation, which opens up interesting research and application opportunities. While this initial experimentation shows encouraging gains, we expect these to be further enhanced with future optimizations and explorations.

For full details of this model please read the [release blog post](https://www.ai21.com/blog/announcing-jamba).

### Model Details

Jamba is a pretrained, mixture-of-experts (MoE) generative text model, with 12B active parameters and an overall of 52B parameters across all experts. It supports a 256K context length, and can fit up to 140K tokens on a single 80GB GPU.

As depicted in the diagram below, Jamba's architecture features a blocks-and-layers approach that allows Jamba to successfully integrate Transformer and Mamba architectures altogether. Each Jamba block contains either an attention or a Mamba layer, followed by a multi-layer perceptron (MLP), producing an overall ratio of one Transformer layer out of every eight total layers.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/jamba_architecture.png"
alt="drawing" width="600"/>

## Usage

### Prerequisites

Jamba requires you use `transformers` version 4.39.0 or higher:
```bash
pip install transformers>=4.39.0
```

In order to run optimized Mamba implementations, you first need to install `mamba-ssm` and `causal-conv1d`:
```bash
pip install mamba-ssm causal-conv1d>=1.2.0
```
You also have to have the model on a CUDA device.

You can run the model not using the optimized Mamba kernels, but it is **not** recommended as it will result in significantly lower latencies. In order to do that, you'll need to specify `use_mamba_kernels=False` when loading the model.

### Run the model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1")
tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

input_ids = tokenizer("In the recent Super Bowl LVIII,", return_tensors='pt').to(model.device)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=216)

print(tokenizer.batch_decode(outputs))
# ["<|startoftext|>In the recent Super Bowl LVIII, the Kansas City Chiefs emerged victorious, defeating the San Francisco 49ers in a thrilling overtime showdown. The game was a nail-biter, with both teams showcasing their skills and determination.\n\nThe Chiefs, led by their star quarterback Patrick Mahomes, displayed their offensive prowess, while the 49ers, led by their strong defense, put up a tough fight. The game went into overtime, with the Chiefs ultimately securing the win with a touchdown.\n\nThe victory marked the Chiefs' second Super Bowl win in four years, solidifying their status as one of the top teams in the NFL. The game was a testament to the skill and talent of both teams, and a thrilling end to the NFL season.\n\nThe Super Bowl is not just about the game itself, but also about the halftime show and the commercials. This year's halftime show featured a star-studded lineup, including Usher, Alicia Keys, and Lil Jon. The show was a spectacle of music and dance, with the performers delivering an energetic and entertaining performance.\n"]
```

<details>
<summary><strong>Loading the model in half precision</strong></summary>

The published checkpoint is saved in BF16. In order to load it into RAM in BF16/FP16, you need to specify `torch_dtype`:

```python
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1", torch_dtype=torch.bfloat16)
# you can also use torch_dtype=torch.float16
```

When using half precision, you can enable the [FlashAttention2](https://github.com/Dao-AILab/flash-attention) implementation of the Attention blocks. In order to use it, you also need the model on a CUDA device. Since in this precision the model is to big to fit on a single 80GB GPU, you'll also need to parallelize it using [accelerate](https://huggingface.co/docs/accelerate/index):
```python
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1",
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map="auto")
```

</details>
<details><summary><strong>Load the model in 8-bit</strong></summary>

**Using 8-bit precision, it is possible to fit up to 140K sequence lengths on a single 80GB GPU.** You can easily quantize the model to 8-bit using [bitsandbytes](https://huggingface.co/docs/bitsandbytes/index). In order to not degrade model quality, we recommend to exclude the Mamba blocks from the quantization:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["mamba"])
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/Jamba-v0.1", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", quantization_config=quantization_config
)
```
</details>

## JambaConfig

[[autodoc]] JambaConfig


## JambaModel

[[autodoc]] JambaModel
    - forward


## JambaForCausalLM

[[autodoc]] JambaForCausalLM
    - forward


## JambaForSequenceClassification

[[autodoc]] transformers.JambaForSequenceClassification
    - forward
