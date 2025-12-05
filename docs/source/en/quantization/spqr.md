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

# SpQR

The [SpQR](https://hf.co/papers/2306.03078) quantization algorithm involves a 16x16 tiled bi-level group 3-bit quantization structure with sparse outliers.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/spqr-diagram.png">
</div>

> [!TIP]
> To quantize a model with SpQR, refer to the [Vahe1994/SpQR](https://github.com/Vahe1994/SpQR) repository.

Load a SpQR-quantized model with [`~PreTrainedModel.from_pretrained`].

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

quantized_model = AutoModelForCausalLM.from_pretrained(
    "elvircrn/Llama-2-7b-SPQR-3Bit-16x16-red_pajama-hf",
    dtype=torch.half,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("elvircrn/Llama-2-7b-SPQR-3Bit-16x16-red_pajama-hf")
```
