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
# Zamba

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

Zamba is a large language model (LLM) trained by Zyphra, and made available under an Apache 2.0 license. Please see the [Zyphra Hugging Face](https://huggingface.co/collections/zyphra/) repository for model weights.

This model was contributed by [pglo](https://huggingface.co/pglo).


## Model details

Zamba-7B-v1 is a hybrid between state-space models (Specifically [Mamba](https://github.com/state-spaces/mamba)) and transformer, and was trained using next-token prediction. Zamba uses a shared transformer layer after every 6 mamba blocks. It uses the [Mistral v0.1 tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.1). We came to this architecture after a series of ablations at small scales. Zamba-7B-v1 was pre-trained on 1T tokens of text and code data.

<img src=https://github.com/user-attachments/assets/c2cff209-b901-483c-87aa-774b82a0769f width=30% height=40% />

## Quick start


### Presequities

Zamba requires you use `transformers` version 4.46.0 or higher:
```bash
pip install transformers>=4.45.0
```

In order to run optimized Mamba implementations, you first need to install `mamba-ssm` and `causal-conv1d`:
```bash
pip install mamba-ssm causal-conv1d>=1.2.0
```
You also have to have the model on a CUDA device.

You can run the model not using the optimized Mamba kernels, but it is **not** recommended as it will result in significantly lower latencies. In order to do that, you'll need to specify `use_mamba_kernels=False` when loading the model.


## Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba-7B-v1")
model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba-7B-v1", device_map="auto", dtype=torch.bfloat16)

input_text = "A funny prompt would be "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```


## Model card

The model cards can be found at:
* [Zamba-7B](https://huggingface.co/Zyphra/Zamba-7B-v1)


## Issues
For issues with model output, or community discussion, please use the Hugging Face community [forum](https://huggingface.co/Zyphra/Zamba-7B-v1/discussions)


## License

The model weights are open-sourced via an Apache 2.0 license.


## ZambaConfig

[[autodoc]] ZambaConfig


## ZambaModel

[[autodoc]] ZambaModel
    - forward


## ZambaForCausalLM

[[autodoc]] ZambaForCausalLM
    - forward


## ZambaForSequenceClassification

[[autodoc]] transformers.ZambaForSequenceClassification
    - forward
