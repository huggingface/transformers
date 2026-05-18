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

# GGUF

[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) is a file format used to store models for inference with [GGML](https://github.com/ggerganov/ggml), a fast and lightweight inference framework written in C and C++. GGUF is a single-file format containing the model metadata and tensors.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-spec.png"/>
</div>

The GGUF format also supports many quantized data types (refer to [quantization type table](https://hf.co/docs/hub/en/gguf#quantization-types) for a complete list of supported quantization types) which saves a significant amount of memory, making inference with large models like Whisper and Llama feasible on local and edge devices.

Transformers supports loading models stored in the GGUF format for further training or finetuning. The GGUF checkpoint is **dequantized to fp32** where the full model weights are available and compatible with PyTorch.

> [!TIP]
> Models that support GGUF include Llama, Mistral, Qwen2, Qwen2Moe, Phi3, Bloom, Falcon, StableLM, GPT2, Starcoder2, and [more](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/ggml.py)

Add the `gguf_file` parameter to [`~PreTrainedModel.from_pretrained`] to specify the GGUF file to load.

```py
# pip install gguf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

dtype = torch.float32 # could be torch.float16 or torch.bfloat16 too
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename, dtype=dtype)
```

Once you're done tinkering with the model, save and convert it back to the GGUF format with the [convert-hf-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py) script.

```py
tokenizer.save_pretrained("directory")
model.save_pretrained("directory")

!python ${path_to_llama_cpp}/convert-hf-to-gguf.py ${directory}
```

## Metal fast paths

Transformers ships two optional Metal fast paths for GGUF on Apple Silicon. Both require the [kernels](https://github.com/huggingface/kernels) package. Unsupported quant types and non-MPS devices fall back to the pure PyTorch path.

Use native quantized inference for serving on MPS and the dequant fast path for training, finetuning, or any non-MPS device.

```sh
pip install kernels
```

### Native quantized inference

Loading a GGUF checkpoint without an explicit `dtype` keeps weights at their native quant and runs each `nn.Linear` through `GgufLinear`. On Apple Silicon, `GgufLinear` dispatches to the Metal kernels and decode is 1.37x faster under `torch.compile` on Llama-3.2-3B Q4_K_M, M3 Max. Packed weights cut resident memory ~3x — Qwen1.5-MoE-A2.7B Q4_K_M loads at 8.8 GB versus 28.6 GB when dequantized to `bfloat16`.

Pass `gguf_linear=True` in [`~AutoModelForCausalLM.from_pretrained`] to enable but passing an explicit `dtype` switches back to dequantize-on-load even if `gguf_linear=True`.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename, gguf_linear=True)
```

Supported quant types are `Q4_0`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, and `IQ4_XS`.

### Metal dequant fast path

The dequant path (used for training, finetuning, and non-MPS devices) routes each tensor through a single Metal compute kernel. The kernel ships in [kernels-community/gguf-dequant](https://huggingface.co/kernels-community/gguf-dequant) and runs 2-7x faster than the chained PyTorch path on M3 Max.

Supported quant types are `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, and `IQ4_XS`.

The fast path is enabled by default when the [kernels](https://github.com/huggingface/kernels) package is installed. Set the `TRANSFORMERS_GGUF_USE_METAL_KERNELS=0` environment variable to disable it.
