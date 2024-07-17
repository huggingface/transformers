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

# Sindibad

# Sindibad

## Overview

The Sindibad model was proposed by TII UAE (Technology Innovation Institute) in their release.

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

- Sindibad is mostly based on Mamba architecutre, the same [tips and best practices](./mamba) would be relevant here.

The model has been trained on approximtely 6T tokens consisting a mixture of many data sources such as RefineWeb, Cosmopedia and Math data.

For more details about the training procedure and the architecture, have a look at [the technical paper of Sindibad]() (coming soon).

# Usage

### A simple generation example: 

```python 
from transformers import SindibadForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/sindibad-7b")
model = SindibadForCausalLM.from_pretrained("tiiuae/sindibad-7b")

input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```


## SindibadConfig

[[autodoc]] SindibadConfig

## SindibadModel

[[autodoc]] SindibadModel
    - forward

## SindibadLMHeadModel

[[autodoc]] SindibadForCausalLM
    - forward
