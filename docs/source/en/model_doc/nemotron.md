<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Nemotron

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

### License

The use of this model is governed by the [NVIDIA AI Foundation Models Community License Agreement](https://developer.nvidia.com/downloads/nv-ai-foundation-models-license).

### Description

Nemotron-4 is a family of enterprise ready generative text models compatible with [NVIDIA NeMo Framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/).

NVIDIA NeMo is an end-to-end, cloud-native platform to build, customize, and deploy generative AI models anywhere. It includes training and inferencing frameworks, guardrailing toolkits, data curation tools, and pretrained models, offering enterprises an easy, cost-effective, and fast way to adopt generative AI. To get access to NeMo Framework, please sign up at [this link](https://developer.nvidia.com/nemo-framework/join).

### References

[Announcement Blog](https://developer.nvidia.com/blog/nvidia-ai-foundation-models-build-custom-enterprise-chatbots-and-co-pilots-with-production-ready-llms/)

### Model Architecture

**Architecture Type:** Transformer

**Network Architecture:** Transformer Decoder (auto-regressive language model).

## Minitron

### Minitron 4B Base

Minitron is a family of small language models (SLMs) obtained by pruning NVIDIA's [Nemotron-4 15B](https://huggingface.co/papers/2402.16819) model. We prune model embedding size, attention heads, and MLP intermediate dimension, following which, we perform continued training with distillation to arrive at the final models.

Deriving the Minitron 8B and 4B models from the base 15B model using our approach requires up to **40x fewer training tokens** per model compared to training from scratch; this results in **compute cost savings of 1.8x** for training the full model family (15B, 8B, and 4B). Minitron models exhibit up to a 16% improvement in MMLU scores compared to training from scratch, perform comparably to other community models such as Mistral 7B, Gemma 7B and Llama-3 8B, and outperform state-of-the-art compression techniques from the literature. Please refer to our [arXiv paper](https://huggingface.co/papers/2407.14679) for more details.

Minitron models are for research and development only.

### HuggingFace Quickstart

The following code provides an example of how to load the Minitron-4B model and use it to perform text generation.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_path = 'nvidia/Minitron-4B-Base'
tokenizer  = AutoTokenizer.from_pretrained(model_path)

device = 'cuda'
dtype  = torch.bfloat16
model  = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device)

# Prepare the input text
prompt = 'Complete the paragraph: our solar system is'
inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

# Generate the output
outputs = model.generate(inputs, max_length=20)

# Decode and print the output
output_text = tokenizer.decode(outputs[0])
print(output_text)
```

### License

Minitron is released under the [NVIDIA Open Model License Agreement](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).

### Evaluation Results

*5-shot performance.* Language Understanding evaluated using [Massive Multitask Language Understanding](https://huggingface.co/papers/2009.03300):

| Average |
| :---- |
| 58.6 |

*Zero-shot performance.* Evaluated using select datasets from the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) with additions:

| HellaSwag | Winogrande | GSM8K| ARC-C | XLSum |
| :------------- | :------------- | :------------- | :------------- | :------------- |
| 75.0 | 74.0 | 24.1  | 50.9 | 29.5


*Code generation performance*. Evaluated using [HumanEval](https://github.com/openai/human-eval):

| p@1, 0-Shot |
| :------------- |
| 23.3 |

Please refer to our [paper](https://huggingface.co/papers/2407.14679) for the full set of results.

### Citation

If you find our work helpful, please consider citing our paper:
```
@article{minitron2024,
      title={Compact Language Models via Pruning and Knowledge Distillation},
      author={Saurav Muralidharan and Sharath Turuvekere Sreenivas and Raviraj Joshi and Marcin Chochowski and Mostofa Patwary and Mohammad Shoeybi and Bryan Catanzaro and Jan Kautz and Pavlo Molchanov},
      journal={arXiv preprint arXiv:2407.14679},
      year={2024},
      url={https://arxiv.org/abs/2407.14679},
}
```

## NemotronConfig

[[autodoc]] NemotronConfig


## NemotronModel

[[autodoc]] NemotronModel
    - forward


## NemotronForCausalLM

[[autodoc]] NemotronForCausalLM
    - forward

## NemotronForSequenceClassification

[[autodoc]] NemotronForSequenceClassification
    - forward


## NemotronForQuestionAnswering

[[autodoc]] NemotronForQuestionAnswering
    - forward


## NemotronForTokenClassification

[[autodoc]] NemotronForTokenClassification
    - forward