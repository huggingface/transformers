<!--Copyright 2026 the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-05-06 and added to Hugging Face Transformers on 2026-05-09.*

# ZAYA

## Overview

ZAYA1 is a 760M active / 8.4B total parameter MoE language model trained by Zyphra. It combines Compressed
Convolutional Attention (CCA), a nonlinear ZAYA1 router, and residual scaling.

ZAYA1 uses the Gemma 3 tokenizer. For more details, see the [ZAYA1 model card](https://huggingface.co/Zyphra/ZAYA1-8B)
and Zyphra's technical reports.

## Usage examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "Zyphra/ZAYA1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("What factors contributed to the fall of the Roman Empire?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## ZayaConfig

[[autodoc]] ZayaConfig

## ZayaModel

[[autodoc]] ZayaModel
    - forward

## ZayaForCausalLM

[[autodoc]] ZayaForCausalLM
    - forward
