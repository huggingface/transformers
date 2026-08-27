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

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "unsloth/Qwen3.5-4B-GGUF"
filename = "Qwen3.5-4B-Q4_K_M.gguf"

model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
```

The weights only stay in their GGUF blocks on Metal (MPS) devices, where the [llama.cpp](https://github.com/ggerganov/llama.cpp) kernels, fetched from the Hub, run the matmuls directly on the packed blocks to keep inference fast.

Right now, the only architecture supported is Qwen3.5. Everything else falls back. On another device or quantization type, the model is [dequantized](#dequantize) at load time, and an architecture that isn't supported yet goes through the legacy loader.

## Dequantize

```py
from transformers import AutoModelForCausalLM, GgufConfig

quantization_config = GgufConfig(gguf_file=filename, dequantize=True)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename, quantization_config=quantization_config)
```

Architectures other than Qwen3.5 are read by the legacy loader and dequantize the model also.

> [!TIP]
> The legacy loader supports Llama, Mistral, Qwen2, Qwen2Moe, Phi3, Bloom, Falcon, StableLM, GPT2, Starcoder2, and [more](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/ggml.py).
