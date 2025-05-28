<!--Copyright 2025 The HuggingFace Team. All rights reserved.
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Normalized-Nemotron

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Normalized-Nemotron model is based on the normalized Transformer (nGPT) architecture proposed in [nGPT: Normalized Transformer with Representation Learning on the Hypersphere](https://arxiv.org/abs/2410.01131).

In nGPT, all vectors forming the embeddings, MLP, attention matrices and hidden states are unit norm normalized. The input stream of tokens travels on the surface of a hypersphere, with each layer contributing a displacement towards the target output predictions. These displacements are defined by the MLP and attention blocks, whose vector components also reside on the same hypersphere.

## How to use

The following code provides an example of how to load the Normalized-Nemotron-8B-Reasoning model and use it to perform text generation.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_path = 'nvidia/Normalized-Nemotron-8B-Reasoning'
tokenizer  = AutoTokenizer.from_pretrained(model_path)

model  = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

# Prepare the input text
prompt = 'Complete the paragraph: our solar system is'
inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

# Generate the output
outputs = model.generate(inputs, max_length=20)

# Decode and print the output
output_text = tokenizer.decode(outputs[0])
print(output_text)
```

## License

The use of this model is governed by [NVIDIA Internal Scientific Research and Development Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/).

### Citation

If you find our work helpful, please consider citing our paper:
```
@inproceedings{loshchilov2025ngpt,
  title={n{GPT}: Normalized Transformer with Representation Learning on the Hypersphere},
  author={Ilya Loshchilov and Cheng-Ping Hsieh and Simeng Sun and Boris Ginsburg},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## NGPTConfig

[[autodoc]] NGPTConfig


## NGPTModel

[[autodoc]] NGPTModel
    - forward


## NGPTForCausalLM

[[autodoc]] NGPTForCausalLM
    - forward