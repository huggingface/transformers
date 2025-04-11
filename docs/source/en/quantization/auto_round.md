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

# AutoRound

Make sure you have auto-round installed:

```bash
pip install auto-round
```

AutoRound supports multiple devices, including CPU, XPU, and CUDA. It automatically selects the best available backend
based on the installed libraries and prompts the user to install additional libraries when a better backend is
available.

## Inference

### CPU

Supports 2,4,8 bits, we recommend to use IPEX(intel-extension-for-pytorch) for 4 bits or XPU and ITREX(
intel-extension-for-transformers) for 2/8 bits.

### XPU

Supports 4 bits, we recommend to use IPEX(intel-extension-for-pytorch) for 4 bits/XPU and 

### CUDA

Supports 2,3 4,8 bits, we recommend to use GPTQModel for 4,8 bits

### Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```