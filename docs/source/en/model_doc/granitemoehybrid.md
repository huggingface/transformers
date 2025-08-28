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

# GraniteMoeHybrid

## Overview


The `GraniteMoeHybrid` model builds on top of `GraniteMoeSharedModel` and `Bamba`. Its decoding layers consist of state space layers or MoE attention layers with shared experts. By default, the attention layers do not use positional encoding.


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "ibm-granite/granite-4.0-tiny-preview"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()

# change input text as desired
prompt = "Write a code to find the maximum value in a list of numbers."

# tokenize the text
input_tokens = tokenizer(prompt, return_tensors="pt")
# generate output tokens
output = model.generate(**input_tokens, max_new_tokens=100)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# loop over the batch to print, in this example the batch size is 1
for i in output:
    print(i)
```

This HF implementation is contributed by [Sukriti Sharma](https://huggingface.co/SukritiSharma) and [Alexander Brooks](https://huggingface.co/abrooks9944).


## GraniteMoeHybridConfig

[[autodoc]] GraniteMoeHybridConfig

## GraniteMoeHybridModel

[[autodoc]] GraniteMoeHybridModel
    - forward

## GraniteMoeHybridForCausalLM

[[autodoc]] GraniteMoeHybridForCausalLM
    - forward