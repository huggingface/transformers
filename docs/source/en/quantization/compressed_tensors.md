<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# compressed-tensors

[compressed-tensors](https://github.com/neuralmagic/compressed-tensors) extends [safetensors](https://github.com/huggingface/safetensors) files to compressed tensor data types to provide a unified checkpoint format for storing and loading various quantization formats such as dense, int-quantized (int8), float-quantized (fp8), and pack-quantized (int4 or int8 weight-quantized packed into int32).

compressed-tensors supports fine-tuning with [PEFT](https://huggingface.co/docs/peft) and includes the following features as well.

- fp8, int4, int8 weight and activation precisions.
- Quantization scales and zero-points strategies for [tensor, channel, group, block, token](https://github.com/neuralmagic/compressed-tensors/blob/83b2e7a969d70606421a76b9a3d112646077c8de/src/compressed_tensors/quantization/quant_args.py#L43-L52).
- Dynamic per-token activation quantization (or any static strategy).
- Quantization of arbitrary modules, not just [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) modules.
- Targeted support for specific modules by name or class.

Install compressed-tensors from [PyPI](https://pypi.org/project/compressed-tensors) to get the latest stable release (recommended) or install it from source to get the latest features.

<hfoptions id="install">
<hfoption id="PyPI">

```bash
pip install compressed-tensors
```

</hfoption>
<hfoption id="source code">

```bash
git clone https://github.com/neuralmagic/compressed-tensors
cd compressed-tensors
pip install -e .
```

</hfoption>
</hfoptions>

Search using the compressed-tensors [tag](https://huggingface.co/models?other=compressed-tensors) to find a compatible model on the Hugging Face Hub.

Pre-quantized models can be loaded directly. To quantize a model into the compressed-tensors format, see [llm-compressor](https://github.com/vllm-project/llm-compressor). Alternatively, models can be created independently and serialized with a compressed-tensors config.

```python
from transformers import AutoModelForCausalLM

ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf", device_map="auto")

# measure memory usage
mem_params = sum([param.nelement()*param.element_size() for param in ct_model.parameters()])
print(f"{mem_params/2**30:.4f} GB")
# 8.4575 GB
```

## Loading modes

A compressed-tensors checkpoint stores its weights compressed (fp8, or packed int4/int8). How they are executed is up to two [`CompressedTensorsConfig`] arguments.

| Configuration | Weights after loading | Execution |
|---------------|-----------------------|-----------|
| default | left compressed | compressed-tensors owns the layers and decompresses the model on the first forward pass |
| `dequantize=True` | dequantized to the model dtype (e.g. BF16) | regular dense matmuls, and the model can be fine-tuned or saved in that dtype |
| `use_optimized_inference=True` | kept quantized | layers whose scheme has a kernel run through it, currently W8A8 fp8; inference only |

## FP8 kernel acceleration

Pass `use_optimized_inference=True` to keep an FP8 compressed-tensors model in FP8 and run its matmuls through hardware-accelerated FP8 kernels ([torch.nn.functional.scaled_mm](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_mm.html), which dispatches to `torch._scaled_mm_v2`; older torch versions fall back to `torch._scaled_mm`), instead of dequantizing the weights back to BF16. Keeping weights in FP8 throughout inference lowers memory usage and speeds up computation. This is inference only, so leave it off to fine-tune.

| Device | Kernel | Notes |
|--------|--------|-------|
| Intel XPU | `torch.nn.functional.scaled_mm` | All XPU devices with FP8 support |
| NVIDIA CUDA (SM89+) | `torch.nn.functional.scaled_mm` | Ada Lovelace (L4, L40), Hopper (H100), Blackwell and newer |
| CPU / CUDA SM80 (A100) | Fallback | `use_optimized_inference=True` is ignored, the model runs dequantized |

The FP8 kernel path supports these quantization layouts.

| Strategy | Example model |
|----------|---------------|
| Per-channel dynamic | [RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic) |
| Per-tensor static | [RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8) |

### Loading a pre-quantized FP8 model

The FP8 kernels are opt-in: ask for them with `use_optimized_inference=True`, and they are used when the model's config specifies FP8 quantization and a supported GPU is available.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

model = AutoModelForCausalLM.from_pretrained(
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
    quantization_config=CompressedTensorsConfig(use_optimized_inference=True),
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic")
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Dequantizing at load time

Without `use_optimized_inference=True`, the model takes the regular compressed-tensors route: the weights are left compressed and compressed-tensors decompresses them on the first forward pass. Pass `dequantize=True` to dequantize them during loading instead, which is what you want to fine-tune the model or save it in its original precision (e.g. BF16).

```python
from transformers import AutoModelForCausalLM, CompressedTensorsConfig

model = AutoModelForCausalLM.from_pretrained(
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
    quantization_config=CompressedTensorsConfig(dequantize=True),
    device_map="auto",
)
```

## Model checkpoint

Compressed-tensor models are defined through its configuration entry. The following example is taken from the [nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf/blob/main/config.json) `config.json` file.

There are a lot of entries to allow for flexible expression both during and after compression, but the entries for loading and inference can be simplified to focus on just a few key entries.

```json
"quantization_config": {
  "config_groups": {
    "group_0": {
      "input_activations": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      },
      "targets": ["Linear"],
      "weights": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      }
    }
  },
  "format": "naive-quantized",
  "ignore": ["lm_head"],
  "quant_method": "compressed-tensors",
  "quantization_status": "frozen"
},
```

The config file specifies the quantization of a config group (`group_0`), which includes weight and activation quantization to fp8 with a static per-tensor strategy. The `lm_head` module is unquantized as shown in the `ignore` key.

For a more detailed look at the model weights, use the [safetensors viewer](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf?show_file_info=model.safetensors.index.json) on the model card to see the quantized weights, input scale, and weight scale for all [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) modules.

| Tensors | Shape | Precision |
| ------- | ----- | --------- |
|model.layers.0.input_layernorm.weight | [4 096] | BF16|
|model.layers.0.mlp.down_proj.input_scale | [1] | BF16|
|model.layers.0.mlp.down_proj.weight | [4 096, 14 336] | F8_E4M3|
|model.layers.0.mlp.down_proj.weight_scale | [1] | BF16|
|model.layers.0.mlp.gate_proj.input_scale | [1] | BF16|
|model.layers.0.mlp.gate_proj.weight | [14 336, 4 096] | F8_E4M3|
|model.layers.0.mlp.gate_proj.weight_scale | [1] | BF16|
|model.layers.0.mlp.up_proj.input_scale| [1] |BF16|
|model.layers.0.mlp.up_proj.weight | [14 336, 4 096] | F8_E4M3|
|model.layers.0.mlp.up_proj.weight_scale | [1] | BF16|
|model.layers.0.post_attention_layernorm.weight | [4 096] |BF16|
|model.layers.0.self_attn.k_proj.input_scale | [1] |  BF16|
|model.layers.0.self_attn.k_proj.weight | [1 024, 4 096]| F8_E4M3|
|model.layers.0.self_attn.k_proj.weight_scale |[1] | BF16|
|model.layers.0.self_attn.o_proj.input_scale | [1] | BF16|
|model.layers.0.self_attn.o_proj.weight | [4 096, 4 096] | F8_E4M3|
|model.layers.0.self_attn.o_proj.weight_scale | [1] | BF16|
|model.layers.0.self_attn.q_proj.input_scale | [1] | BF16|
|model.layers.0.self_attn.q_proj.weight | [4 096, 4 096] | F8_E4M3|
|model.layers.0.self_attn.q_proj.weight_scale | [1] | BF16|
|model.layers.0.self_attn.v_proj.input_scale | [1] | BF16|
|model.layers.0.self_attn.v_proj.weight | [1 024, 4 096] | F8_E4M3|
|model.layers.0.self_attn.v_proj.weight_scale | [1] | BF16|

When loading a compressed-tensors model with the [`~quantizers.HFQuantizer`] integration, the targeted modules are handed over to compressed-tensors: it attaches the resolved `quantization_scheme`, sets `quantization_status`, registers the parameters the checkpoint stores (`weight` in fp8, plus `weight_scale` and, for a static strategy, `input_scale`) and installs its own forward pass over them. They stay [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) instances, so that is what `print` shows — recent compressed-tensors versions no longer wrap them in a `CompressedLinear` subclass. Modules listed under `ignore`, such as `lm_head`, are left untouched.

With `dequantize=False` (the default), the weights are still compressed once loading is over, and compressed-tensors decompresses the whole model on the first forward pass. `dequantize=True` does it during loading instead, so no forward pass is needed to get dense weights.

```python
import torch
from transformers import AutoModelForCausalLM, CompressedTensorsConfig

model_id = "nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf"

ct_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=CompressedTensorsConfig(dequantize=False),
    device_map="auto",
)
q_proj = ct_model.model.layers[0].self_attn.q_proj
print(q_proj, q_proj.quantization_status)
# Linear(in_features=4096, out_features=4096, bias=False) QuantizationStatus.COMPRESSED
# ^ compressed-tensors module: fp8 weight, weight_scale, and its own forward

ct_model(input_ids=torch.tensor([[0, 1, 2]], device=ct_model.device))
print(q_proj, q_proj.quantization_status)
# Linear(in_features=4096, out_features=4096, bias=False) QuantizationStatus.DECOMPRESSED
# ^ weight is BF16 now, decompressed by that forward pass

ct_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=CompressedTensorsConfig(dequantize=True),
    device_map="auto",
)
print(ct_model.model.layers[0].self_attn.q_proj)
# Linear(in_features=4096, out_features=4096, bias=False)      weight: BF16
```

With `use_optimized_inference=True`, the layers covered by an fp8 config group are replaced by `CompressedTensorsFP8Linear`, which holds the fp8 weight and its scale in the layout its row-wise matmul kernel expects. Those weights stay in fp8, forward passes included.

```python
from transformers import AutoModelForCausalLM, CompressedTensorsConfig

ct_model = AutoModelForCausalLM.from_pretrained(
    "nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf",
    quantization_config=CompressedTensorsConfig(use_optimized_inference=True),
    device_map="auto",
)
print(ct_model.model.layers[0].self_attn.q_proj)
# CompressedTensorsFP8Linear(in_features=4096, out_features=4096, bias=False)      weight: F8_E4M3
```
