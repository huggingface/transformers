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

# VPTQ 

> [!TIP]
> Try VPTQ on [Hugging Face](https://huggingface.co/spaces/microsoft/VPTQ)!
> Try VPTQ on [Google Colab](https://colab.research.google.com/github/microsoft/VPTQ/blob/main/notebooks/vptq_example.ipynb)!
> Know more about VPTQ on [ArXiv](https://arxiv.org/pdf/2409.17066)!

Vector Post-Training Quantization ([VPTQ](https://github.com/microsoft/VPTQ)) is a novel Post-Training Quantization method that leverages Vector Quantization to high accuracy on LLMs at an extremely low bit-width (<2-bit). VPTQ can compress 70B, even the 405B model, to 1-2 bits without retraining and maintain high accuracy.

Inference support for VPTQ is released in the `vptq` library. Make sure to install it to run the models:
```bash
pip install vptq
```

The library provides efficient kernels for NVIDIA/AMD GPU inference.

To run VPTQ models simply load a model that has been quantized with VPTQ:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-65536-woft",
    torch_dtype="auto", 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-65536-woft")
```


