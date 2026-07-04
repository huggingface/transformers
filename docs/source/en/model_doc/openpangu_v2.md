<!--Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Copyright 2026 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# OpenPanguV2

OpenPanguV2 is an MoE model trained on Ascend. Its context length is 512k. During Post-training, OpenPanguV2 is 
trained through unified SFT with slow and fast thinking capability, multiple specialist RL traning, on-policy 
distillation combining multiple RL specialists.

## Architecture

OpenPanguV2 brings several major architectural improvements:

* **Efficient attention**: The model retains MLA for efficient inference and combines DSA and SWA in a 1:2 layer ratio. 
  SWA layers handle local-window modeling, while DSA layers capture sparse global context. This design lowers compute, 
  memory footprint, and memory access costs for long-context inference while preserving accuracy.
* **Residual topology**: TThe conventional residual path is replaced with a 4-stream mHC design, improving representation
  diversity and generalization.
* **Optimizer**: Training uses the Muon optimizer for faster convergence.

## Usage examples

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

```python
from transformers import pipeline


pipe = pipeline(
    task="text-generation",
    model="openpangu/openPangu-2.0-Flash",
    device_map="auto"
)
pipe("Give me a short introduction to large language model.")
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openpangu/openPangu-2.0-Flash"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
input_ids = tokenizer("Give me a short introduction to large language model.", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## OpenPanguV2Config

[[autodoc]] OpenPanguV2Config

## OpenPanguV2Model

[[autodoc]] OpenPanguV2Model
    - forward

## OpenPanguV2ForCausalLM

[[autodoc]] OpenPanguV2ForCausalLM
    - forward