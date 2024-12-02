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

# SpQR


This module contains a single-batch inference kernel for a model quantizated via the SpQR algorithm with a specific 16x16 tile and 3-bit configuration in mind, alsongside unstructured sparsity.
The compression algorithm is detailed in the research paper "[SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078)".

The instructions on how to quantize models yourself, as well as all the relevant code can be found in the corresponding GitHub [repository](https://github.com/Vahe1994/SpQR). To run SpQR models simply load a model that has been quantized with SpQR:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

quantized_model = AutoModelForCausalLM.from_pretrained(
    "elvircrn/Llama-2-7b-SPQR-3Bit-16x16-red_pajama-hf",
    torch_dtype=torch.half,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("elvircrn/Llama-2-7b-SPQR-3Bit-16x16-red_pajama-hf")
```