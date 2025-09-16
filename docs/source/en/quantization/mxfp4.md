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

MXFP4 is a **4-bit** floating point format that dramatically reduces the memory requirements of large models. Large models (GPT-OSS-120B) can fit on a single 80GB GPU and smaller models (GPT-OSS-20B) only require 16GB of memory. It uses blockwise scaling to preserve it's range and accuracy, which typically becomes degraded at lower precisions.

To use MXPF4, make sure your hardware meets the following requirements.

- Install `accelerate`, `kernels`, and `triton≥ 3.4`. Only manually install `triton 3.4` if you're using `PyTorch 2.7` because it is already supported in `PyTorch 2.8`.
- NVIDIA GPU Compute Capability ≥ 7.5 which includes Tesla GPUs and newer.

Code below will check for NVIDIA GPU Compute Capability:

```python
from torch import cuda
cuda.get_device_capability()

# (7, 5)
```

Let's look at the quantization configuration of the gpt-oss series of models:

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

If `'quant_method': 'mxfp4'` is present, the model will automatically use the MXFP4 pathway.

## MXFP4 kernels

`transformers` automatically pulls in the `MXFP4`-aware Triton kernels from the community repository when you load a model that needs them. The repository will appear in your local cache and will be used during the forward pass. For the `MXFP4` kernels one does not need to use the `use_kernels=True` parameter like before, it is set to default in `transformers`.

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

## Resources

To learn more about it please refer [here](https://huggingface.co/blog/faster-transformers#mxfp4-quantization).
