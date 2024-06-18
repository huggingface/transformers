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

# AutoRound
Make sure you have auto-round installed:

```bash
pip install auto-round
```


AutoRound supports various backends, including AutoRound, QBits (primarily for x86 CPUs), and GPTQ. Please note that models using the GPTQ backend cannot switch to other backends due to differences in packing.


## AutoRound Backend
This backend primarily reuses the AutoGPTQ kernel but addresses the asymmetry accuracy drop issue.




## Exllamav2
To use the backend, please install auto-round from source, otherwise it will switch to triton backend automatically. Only supports 4 bits.
```bash
git clone https://github.com/intel/auto-round.git
cd auto-round && pip install -vvv --no-build-isolation -e .
```

## Triton
Supports 2,4,8 bits

## Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Intel/Qwen2-7B-int4-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```


# Qbits backend
This backend is for CPU, mainly for Intel CPU. It supports 2,4,8 bits
Please install intel-extension-for-transformers first
```bash
pip3 install intel-extension-for-transformers
```
## Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Intel/Qwen2-7B-int4-inc"
from transformers import AutoRoundConfig
quantization_config = AutoRoundConfig(
   backend="cpu"
)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu",quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

# GPTQ backend
Support 2,4,8 bits
Please install auto-gptq first
```bash
pip3 install auto-gptq
```
## Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Intel/Mistral-7B-v0.1-int4-inc-lmhead"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```