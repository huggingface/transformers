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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# Jamba

[Jamba](https://huggingface.co/papers/2403.19887) is a hybrid Transformer-Mamba mixture-of-experts (MoE) language model ranging from 52B to 398B total parameters. This model aims to combine the advantages of both model families, the performance of transformer models and the efficiency and longer context (256K tokens) of state space models (SSMs) like Mamba.

Jamba's architecture features a blocks-and-layers approach that allows Jamba to successfully integrate Transformer and Mamba architectures altogether. Each Jamba block contains either an attention or a Mamba layer, followed by a multi-layer perceptron (MLP), producing an overall ratio of one Transformer layer out of every eight total layers. MoE layers are mixed in to increase model capacity.

You can find all the original Jamba checkpoints under the [AI21](https://huggingface.co/ai21labs) organization.

> [!TIP]
> Click on the Jamba models in the right sidebar for more examples of how to apply Jamba to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
# install optimized Mamba implementations
# !pip install mamba-ssm causal-conv1d>=1.2.0
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="ai21labs/AI21-Jamba-Mini-1.6",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "ai21labs/AI21-Jamba-Large-1.6",
)
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/AI21-Jamba-Large-1.6",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model ai21labs/AI21-Jamba-Mini-1.6 --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 8-bits.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_skip_modules=["mamba"])

# a device map to distribute the model evenly across 8 GPUs
device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 2, 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2, 'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 3, 'model.layers.28': 3, 'model.layers.29': 3, 'model.layers.30': 3, 'model.layers.31': 3, 'model.layers.32': 3, 'model.layers.33': 3, 'model.layers.34': 3, 'model.layers.35': 3, 'model.layers.36': 4, 'model.layers.37': 4, 'model.layers.38': 4, 'model.layers.39': 4, 'model.layers.40': 4, 'model.layers.41': 4, 'model.layers.42': 4, 'model.layers.43': 4, 'model.layers.44': 4, 'model.layers.45': 5, 'model.layers.46': 5, 'model.layers.47': 5, 'model.layers.48': 5, 'model.layers.49': 5, 'model.layers.50': 5, 'model.layers.51': 5, 'model.layers.52': 5, 'model.layers.53': 5, 'model.layers.54': 6, 'model.layers.55': 6, 'model.layers.56': 6, 'model.layers.57': 6, 'model.layers.58': 6, 'model.layers.59': 6, 'model.layers.60': 6, 'model.layers.61': 6, 'model.layers.62': 6, 'model.layers.63': 7, 'model.layers.64': 7, 'model.layers.65': 7, 'model.layers.66': 7, 'model.layers.67': 7, 'model.layers.68': 7, 'model.layers.69': 7, 'model.layers.70': 7, 'model.layers.71': 7, 'model.final_layernorm': 7, 'lm_head': 7}
model = AutoModelForCausalLM.from_pretrained("ai21labs/AI21-Jamba-Large-1.6",
                                             dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                                             quantization_config=quantization_config,
                                             device_map=device_map)

tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Large-1.6")

messages = [
   {"role": "system", "content": "You are an ancient oracle who speaks in cryptic but wise phrases, always hinting at deeper meanings."},
   {"role": "user", "content": "Hello!"},
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)

outputs = model.generate(input_ids, max_new_tokens=216)

# Decode the output
conversation = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Split the conversation to get only the assistant's response
assistant_response = conversation.split(messages[-1]['content'])[1].strip()
print(assistant_response)
# Output: Seek and you shall find. The path is winding, but the journey is enlightening. What wisdom do you seek from the ancient echoes?
```

## Notes

- Don't quantize the Mamba blocks to prevent model performance degradation.
- It is not recommended to use Mamba without the optimized Mamba kernels as it results in significantly lower latencies. If you still want to use Mamba without the kernels, then set `use_mamba_kernels=False` in [`~AutoModel.from_pretrained`].

    ```py
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("ai21labs/AI21-Jamba-1.5-Large",
                                                 use_mamba_kernels=False)
    ```

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
