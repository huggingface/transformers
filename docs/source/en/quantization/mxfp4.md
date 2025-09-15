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

# MXFP4

| ![explanation of mxfp4 format](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/mxfp4.png) |
| :--: |
| Figure 1: The E2M1 format used in the MXFP4 format |

`MXFP4` is a 4-bit floating format with E2M1 layout: 1 sign bit, 2 exponent bits, and 1 mantissa bit, as shown in **Figure 1**. On its own, E2M1 is very coarse. MXFP4 compensates with **blockwise scaling**:

- Vectors are grouped into blocks of 32 elements.
- Each block stores a shared scale that restores dynamic range when dequantizing.
- Inside each block, 4-bit values represent numbers relative to that scale.

This blockwise scheme lets `MXFP4` keep range while using very few bits. In practice, GPT-OSS 20B fits in roughly `16 GB` of VRAM and GPT-OSS 120B fits in roughly `80 GB` when `MXFP4` is active, which is the difference between “cannot load” and “can run on a single GPU.” The catch is that matrix multiplies now have to respect block scales. Doing this efficiently at scale requires dedicated kernels.

Key implementation details:

- Quantizer logic: Found in the [MXFP4 quantizer file](https://github.com/huggingface/transformers/blob/0997c2f2ab08c32c8e2f90aaad06e29a7108535b/src/transformers/quantizers/quantizer_mxfp4.py), this handles the core quantization process for MXFP4.
- Integration hooks: The [MXFP4 integration file](https://github.com/huggingface/transformers/blob/0997c2f2ab08c32c8e2f90aaad06e29a7108535b/src/transformers/integrations/mxfp4.py) enables seamless use of MXFP4 within the transformers framework.

To check if a model supports `MXFP4`, inspect its configuration:
```py
from transformers import GptOssConfig

model_id = "openai/gpt-oss-120b"
cfg = GptOssConfig.from_pretrained(model_id)
print(cfg.quantization_config)

# Example output:
# {
#   'modules_to_not_convert': [
#     'model.layers.*.self_attn',
#     'model.layers.*.mlp.router',
#     'model.embed_tokens',
#     'lm_head'
#   ],
#   'quant_method': 'mxfp4'
# }
```

If `'quant_method': 'mxfp4'` is present, the model will automatically use the MXFP4 pathway with Triton kernels when supported.

## Requirements and fallbacks

To run `MXFP4` on GPU you need:

1. `accelerate`, `kernels`, and `triton>=3.4` installed. Note that `Pytorch 2.8` already comes with `triton 3.4`, so you only need to manually install triton if using `Pytorch 2.7`.
2. NVIDIA GPU with compute capability `≥ 7.5`. This goes all the way back to Tesla, so you can run `gpt-oss-20b` on the free tiers of Google Colab and Kaggle, and on many consumer GPUs.

If these constraints are not met, `transformers` falls back to a higher-precision path (`bfloat16` is used by default), which requires about 4× the memory of MXFP4.

The [snippet](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/memory-requirements-quantized-vs-dequantized.py) loads GPT-OSS twice on CUDA: once with `Mxfp4Config(dequantize=True)` (memory intensive) and once in the default quantized path (memory efficient). **Figure 2** shows the amount of used VRAM after each load so you can visualize the savings.

| ![memory used with quantized vs dequantized models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/quantization.png) |
| :--: |
| Figure 2: Memory requirements for the quantized and dequantized models |

## Kernels for MXFP4 (Advanced)

Efficient `MXFP4` requires kernels that understand 32-element blocks and their scales during GEMMs and fused ops. This is where **Kernels from the Hub** comes in again. `transformers` automatically pulls in the `MXFP4`-aware
Triton kernels from the community repository when you load a model that needs them. The repository will appear in your local cache and will be used during the forward pass. For the `MXFP4` kernels one does not need to use the `use_kernels=True` parameter like before, it is set to default in `transformers`.

Quick sanity check with the Hugging Face cache CLI,  after running `gpt-oss-20b` on a GPU compatible with the triton MXFP4 kernels:

```shell
hf cache scan
```

Sample output:

```shell
REPO ID                          REPO TYPE SIZE ON DISK
-------------------------------- --------- ------------
kernels-community/triton_kernels model           536.2K
openai/gpt-oss-20b               model            13.8G
```

This indicates the MXFP4 kernels were fetched and are available for execution.

Let's run some benchmarks and see how well the MXFP4 kernels perform. In **Figure 3**, we see that the `MXFP4` kernels are even better than the custom MoE and RMSNorm kernels for larger batches.

| ![benchmark mxfp4 kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/benchmark-mxfp4.png) |
| :--: |
| Figure 3: MXFP4 kernel benchmark |

> [!NOTE]
> You can explore and play with the benchmarking script [here](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/benchmark-mxfp4-kernels.py)
