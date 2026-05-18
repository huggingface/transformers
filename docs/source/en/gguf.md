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

Transformers supports loading models stored in the GGUF format for further training or finetuning. The GGUF checkpoint is **dequantized lazily on the target device** by the conversion-mapping pipeline (see [Adding GGUF support for a new architecture](#adding-gguf-support-for-a-new-architecture) below for how that's plugged in).

> [!TIP]
> Architectures wired up for GGUF loading include Llama, Mistral, Phi3, Cohere, Qwen2, Qwen3, Deci, StableLM, Starcoder2, Nemotron, Gemma2, Gemma3 (text + multimodal), Bloom, GPT2, Mamba, LFM2, Falcon, Qwen2-MoE, Qwen3-MoE, MiniMax-M2, GPT-OSS, T5, UMT5. The full list can be fetched with `_GGUF_ARCH_CONVERTERS.keys()` in [`modeling_gguf_pytorch_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_gguf_pytorch_utils.py).

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

## Adding GGUF support for a new architecture

GGUF loading plugs into the same weight-conversion framework that the rest of `transformers` uses for safetensors / Fp8 / etc. To support a new architecture you only need to:

1. **Write a list of `WeightRenaming` / `WeightConverter` rules** in [`src/transformers/modeling_gguf_pytorch_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_gguf_pytorch_utils.py) and register it in `_GGUF_ARCH_CONVERTERS`. You can also just combine existing rules whenever relevant. You then map the `model_type` to the rules.
2. **Add the `model_type` to `EXPECTED_MODEL_TYPES`** in [`tests/quantization/ggml/test_gguf_arch_coverage.py`](https://github.com/huggingface/transformers/blob/main/tests/quantization/ggml/test_gguf_arch_coverage.py) so the fast regression test enforces it.

### Rule types

For more details you should check the doc [dynamic weight loading](https://huggingface.co/docs/transformers/weightconverter).
As a reminder it looks like this:

```python
from transformers.core_model_loading import WeightConverter, WeightRenaming

# Pure key rename (cheap — no op chain). Substring `re.sub` style: escape dots,
# use alternation when the same target template covers several source names.
WeightRenaming(r"^blk\.", "model.layers.")
WeightRenaming(r"\.ffn_(gate|up|down)\.weight", r".mlp.\1_proj.weight")

# With an actual tensor transform (transpose, permute, concat, etc.).
WeightConverter(
    source_patterns=r"\.attn_q\.weight",
    target_patterns=".self_attn.q_proj.weight",
    operations=[ReversePermuteAttnQ()],
)
```

`WeightRenaming`s are applied **sequentially** (each one operates on the previous rule's output), so structural prefix renames (`^blk\.` → `model.layers.`, etc.) should come first and per-tensor renames after. `WeightConverter`s are evaluated after all the renames; the first one whose source pattern matches is selected.

`GGUFDequantize` is automatically prepended to every `WeightConverter`'s op chain by `GGUFQuantizer.update_weight_conversions`, updating the keys to dequantize.

### Existing transform ops

We added a bunch of weight conversion operations that are for now sufficient to cover all the cases. They are in [`src/transformers/gguf_conversion_ops.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/gguf_conversion_ops.py).

| Op | Used for |
|----|----------|
| `ReversePermuteAttnQ`, `ReversePermuteAttnK` | Llama-family rope-permuted Q/K weights |
| `SubtractOne` | Gemma / Nemotron norm weights (stored as `weight + 1` in GGUF) |
| `Unsqueeze(dim)` | Mamba conv1d, LFM2 shortconv — add a singleton dim |
| `LogNegate` | Mamba SSM-A: `log(-x)` on load |
| `BloomReshapeQKVWeight`, `BloomReshapeQKVBias` | Bloom interleaved QKV |
| `Concatenate(dim)` (from `core_model_loading`) | MoE gate+up merge into `gate_up_proj` |
| `Transpose` (from `core_model_loading`) | GPT-2 transposed `c_attn` / `c_proj` / `c_fc` |

If your architecture needs a new tensor transform, add it to `gguf_conversion_ops.py` as a `ConversionOps` subclass with `convert(input_dict, source_patterns, target_patterns, **kwargs)` and a `reverse_op` property.

### Example: adding a new Llama-like architecture

For a model whose HF `model_type` is `"my_llama"` and which uses the standard rope-permuted Q/K, plain Llama norms, and `mlp.gate_proj` / `up_proj` / `down_proj`, the rule list is already provided as `_LLAMA_CONVERTERS`. You only need:

```python
# src/transformers/modeling_gguf_pytorch_utils.py
_GGUF_ARCH_CONVERTERS = {
    ...
    "my_llama": _LLAMA_CONVERTERS,
}
```

For a variant with `q_norm` / `k_norm` and attn biases (StableLM style) you compose:

```python
_MY_LLAMA_CONVERTERS = _LLAMA_CONVERTERS + [
    WeightRenaming(r"\.attn_(q|k)_norm\.weight", r".self_attn.\1_norm.weight"),
    WeightRenaming(r"\.attn_(q|k|v)\.bias", r".self_attn.\1_proj.bias"),
]
```

For a wholly new layout, start from one of the existing entries (`_BLOOM_CONVERTERS`, `_GPT2_CONVERTERS`, `_T5_CONVERTERS`, …) and adapt the renames + add any required `WeightConverter` ops.

### Dequantization performance

The actual byte-level dequant happens inside the `GGUFDequantize` op via [`integrations/gguf_dequant.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/gguf_dequant.py), a pure-PyTorch port of [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (same kernels diffusers uses). 

It supports Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, Q2_K…Q6_K, IQ4_NL, IQ4_XS, BF16, F16, F32 and runs on CPU or GPU/MPS.

If you add a new quant type, add the corresponding `_dq_<TYPE>` kernel to `gguf_dequant.py` and register it in `_build_dispatch()`.
