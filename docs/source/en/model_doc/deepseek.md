<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Copyright (c) 2024, DeepSeek-AI.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# DeepSeekMoE

## DeepSeekMoE

### License

The use of DeepSeekMoE models is subject to the [Model License](https://github.com/deepseek-ai/DeepSeek-MoE/blob/main/LICENSE-MODEL). DeepSeekMoE supports commercial use.

### Description

DeepSeekMoE 16B is a Mixture-of-Experts (MoE) language model with 16.4B parameters. It employs an innovative MoE architecture, which involves two principal strategies: fine-grained expert segmentation and shared experts isolation. It is trained from scratch on 2T English and Chinese tokens, and exhibits comparable performance with DeepSeek 7B and LLaMA2 7B, with only about 40% of computations. For research purposes, we release the model checkpoints of DeepSeekMoE 16B Base and DeepSeekMoE 16B Chat to the public, which can be deployed on a single GPU with 40GB of memory without the need for quantization.

### Usage

#### Text Completion

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

#### Chat Completion
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/deepseek-moe-16b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

messages = [
    {"role": "user", "content": "Who are you?"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)
```

### References

[arXiv paper](https://arxiv.org/abs/2401.06066)

### Citation

```
@article{dai2024deepseekmoe,
  author={Damai Dai and Chengqi Deng and Chenggang Zhao and R. X. Xu and Huazuo Gao and Deli Chen and Jiashi Li and Wangding Zeng and Xingkai Yu and Y. Wu and Zhenda Xie and Y. K. Li and Panpan Huang and Fuli Luo and Chong Ruan and Zhifang Sui and Wenfeng Liang},
  title={DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models}, 
  journal   = {CoRR},
  volume    = {abs/2401.06066},
  year      = {2024},
  url       = {https://arxiv.org/abs/2401.06066},
}
```

## DeepseekConfig

[[autodoc]] DeepseekConfig


## DeepseekModel

[[autodoc]] DeepseekModel
    - forward


## DeepseekForCausalLM

[[autodoc]] DeepseekForCausalLM
    - forward

## DeepseekForSequenceClassification

[[autodoc]] DeepseekForSequenceClassification
    - forward


## DeepseekForQuestionAnswering

[[autodoc]] DeepseekForQuestionAnswering
    - forward


## DeepseekForTokenClassification

[[autodoc]] DeepseekForTokenClassification
    - forward
