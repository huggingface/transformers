<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GGUF

[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) is a single-file format used to store models for inference with [GGML](https://github.com/ggerganov/ggml), containing the model metadata and tensors. It supports many quantized data types (refer to the [quantization type table](https://hf.co/docs/hub/en/gguf#quantization-types)), which saves a significant amount of memory.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-spec.png"/>
</div>


## Load GGUF models

Install [kernels](https://huggingface.co/docs/kernels/index), otherwise the packed path falls back to full [dequantization](#dequantize) at load.

```bash
pip install kernels
```

Weights stay packed when the Hub kernel [transformers-community/ggml-quantization](https://huggingface.co/transformers-community/ggml-quantization) is available. The loader defaults to MPS when that kernel is present and runs matmuls directly on the packed blocks. If the kernel isn't available, the model is dequantized at load.

```py
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "unsloth/Qwen3.5-4B-GGUF"
filename = "Qwen3.5-4B-Q4_K_M.gguf"

model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename, dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
```

The same load works for Qwen3.5 MoE.

```py
moe_model_id = "unsloth/Qwen3.5-35B-A3B-GGUF"
moe_filename = "Qwen3.5-35B-A3B-Q4_K_M.gguf"

moe_model = AutoModelForCausalLM.from_pretrained(
    moe_model_id, gguf_file=moe_filename, dtype=torch.float32
)
moe_tokenizer = AutoTokenizer.from_pretrained(moe_model_id, gguf_file=moe_filename)
```

The packed path currently supports Qwen3.5 and Qwen3.5 MoE. Use `dtype=torch.float32` for packed loads. The loader warns when another dtype is requested because the kernels compute in float32. Other architectures go through the legacy loader.

## Attention

On Metal, with kernels installed, attention can use [ggml-attn](https://huggingface.co/transformers-community/ggml-attn), the same flash-attention kernel llama.cpp uses for decode and prefill.

```py
model = AutoModelForCausalLM.from_pretrained(
    model_id, gguf_file=filename, dtype=torch.float32, attn_implementation="transformers-community/ggml-attn"
)
```

## Dequantize

Dequantizing unpacks every weight at load time and gives back a plain dense model. It is the fallback whenever the fast, compressed path doesn't apply. You can also ask for it explicitly with [`GgufConfig`].

```py
import torch

from transformers import AutoModelForCausalLM, GgufConfig

quantization_config = GgufConfig(dequantize=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, gguf_file=filename, quantization_config=quantization_config, dtype=torch.bfloat16
)
```

You get a regular dense model in the `dtype` you passed.

Architectures other than Qwen3.5 and Qwen3.5 MoE go through the legacy loader, which always dequantizes.

> [!TIP]
> The legacy loader supports Llama, Mistral, Qwen2, Qwen2Moe, Phi3, Bloom, Falcon, StableLM, GPT2, Starcoder2, and [more](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/ggml.py).

## Serve

[transformers serve](../serve-cli/serving) lists each `.gguf` file in a repository as its own model. A GGUF model is named `<repo>:<file>.gguf`, since a repository holds
several quantizations and the id has to say which one to load. Requests name it the same way.

```shell
transformers serve unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf
```
