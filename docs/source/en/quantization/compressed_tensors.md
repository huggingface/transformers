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

# Compressed Tensors

Compressed tensors supports the quantization of models to a variety of formats and provides an extensible
framework for adding new formats and strategies.

Compressed models can be easily created using [llm-compressor](https://github.com/vllm-project/llm-compressor).
Alternatively models can be created indepedenty and serialized with a compressed tensors config.

Supported formats include:

 - FP8, INT4, INT8 (for Q/DQ arbitrary precision is allowed for INT)
 - Activation quantization (static)
 - Dynamic per-token activation quantization
 - Supports quantization of arbitrary layer types
 - Targeted support or ignoring of layers by name or class

## Installation

```bash
pip install compressed-tensors
```


## Sample Model Load
```python
from transformers import AutoModelForCausalLM
compressed_tensors_model = AutoModelForCausalLM.from_pretrained("nm-testing/tinyllama-oneshot-w4a16-group128-v3")
```


## More Coming Soon!
