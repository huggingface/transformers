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
*This model was released on 2025-05-21 and added to Hugging Face Transformers on 2025-05-21.*

# FalconH1

## Overview

The [FalconH1](https://huggingface.co/blog/tiiuae/falcon-h1) model was developed by the TII Pretraining team. A comprehensive research paper covering the architecture, pretraining dynamics, experimental results, and conclusions is forthcoming. You can read more about this series in [this website](https://github.com/tiiuae/Falcon-H1).

## Contributors

This model was contributed by [DhiyaEddine](https://huggingface.co/DhiyaEddine), [ybelkada](https://huggingface.co/ybelkada), [JingweiZuo](https://huggingface.co/JingweiZuo), [IlyasChahed](https://huggingface.co/IChahed), and [MaksimVelikanov](https://huggingface.co/yellowvm).
The original code can be found [here](https://github.com/tiiuae/Falcon-H1).


## FalconH1Config

| Model     | Depth | Dim  | Attn Heads | KV | Mamba Heads | d_head       | d_state | Ctx Len        |
|-----------|--------|------|------------|----|--------------|--------------|------|-----------------|
| H1 0.5B   | 36     | 1024 | 8          | 2  | 24           | 64 / 64      | 128  | 4K, 16K-SFT     |
| H1 1.5B   | 24     | 2048 | 8          | 2  | 48           | 128 / 64     | 256  | 128K            |
| H1 1.5B-d | 66     | 1280 | 6          | 2  | 24           | 128 / 64     | 256  | 128K            |
| H1 3B     | 32     | 2560 | 10         | 2  | 32           | 128 / 128    | 256  | 128K            |
| H1 7B     | 44     | 3072 | 12         | 2  | 24           | 128 / 128    | 256  | 256K            |
| H1 34B    | 72     | 5120 | 20         | 4  | 32           | 128 / 128    | 256  | 256K            |



[[autodoc]] FalconH1Config

<!---
## Usage Tips
Tips: 
- The architecture is based on Mamba-2 models.
## FalconH1Model
[[autodoc]] FalconH1Model
    - forward
-->

## FalconH1ForCausalLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon-H1-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon-H1-7B-Instruct")

message = ["Mamba is a snake with following properties  "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

[[autodoc]] FalconH1ForCausalLM
    - forward

This HF implementation is contributed by [younesbelkada](https://github.com/younesbelkada) and [DhiaEddineRhaiem](https://github.com/dhiaEddineRhaiem). 