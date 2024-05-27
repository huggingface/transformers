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

# AQLM

> [!TIP]
> Try AQLM on [Google Colab](https://colab.research.google.com/drive/1-xZmBRXT5Fm3Ghn4Mwa2KRypORXb855X?usp=sharing)!

Additive Quantization of Language Models ([AQLM](https://arxiv.org/abs/2401.06118)) is a Large Language Models compression method. It quantizes multiple weights together and take advantage of interdependencies between them. AQLM represents groups of 8-16 weights as a sum of multiple vector codes.

Inference support for AQLM is realised in the `aqlm` library. Make sure to install it to run the models (note aqlm works only with python>=3.10):
```bash
pip install aqlm[gpu,cpu]
```

The library provides efficient kernels for both GPU and CPU inference and training.

The instructions on how to quantize models yourself, as well as all the relevant code can be found in the corresponding GitHub [repository](https://github.com/Vahe1994/AQLM). To run AQLM models simply load a model that has been quantized with AQLM:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf",
    torch_dtype="auto", 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf")
```

## PEFT

Starting with version `aqlm 1.0.2`, AQLM supports Parameter-Efficient Fine-Tuning in a form of [LoRA](https://huggingface.co/docs/peft/package_reference/lora) integrated into the [PEFT](https://huggingface.co/blog/peft) library.

## AQLM configurations

AQLM quantization setups vary mainly on the number of codebooks used as well as codebook sizes in bits. The most popular setups, as well as inference kernels they support are:
 
| Kernel | Number of codebooks | Codebook size, bits | Notation | Accuracy | Speedup     | Fast GPU inference | Fast CPU inference |
|---|---------------------|---------------------|----------|-------------|-------------|--------------------|--------------------|
| Triton | K                   | N                  | KxN     | -        | Up to ~0.7x | ✅                  | ❌                  |
| CUDA | 1                   | 16                  | 1x16     | Best        | Up to ~1.3x | ✅                  | ❌                  |
| CUDA | 2                   | 8                   | 2x8      | OK          | Up to ~3.0x | ✅                  | ❌                  |
| Numba | K                   | 8                   | Kx8      | Good        | Up to ~4.0x | ❌                  | ✅                  |
