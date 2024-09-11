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

# BitNet

BitNet b1.58 is a quantization method for boosting the efficiency of LLMs. It uses ternary weights (-1, 0, 1), giving it 1.58 bits of precision. No extra libraries are needed for this. However, BitNet models can't be quantized on the fly—they need to be pre-trained or fine-tuned with the quantization applied (it's a Quantization aware training technique). Once trained, these models are already quantized and available as packed versions on the hub.

A quantized model can be load : 

```py
from transformers import AutoModelForCausalLM
path = "/path/to/model"
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
```