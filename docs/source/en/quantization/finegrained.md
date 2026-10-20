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

# Fine-grained

[FP8](./fp8), [MXFP8](./mxfp8), [MXFP4](./mxfp4) and [NVFP4](./nvfp4) are block-scaled formats. They store
low-precision weights with one scale per block or group of values. The four formats share one integration and one
kernel bundle, [finegrained-kernels](https://huggingface.co/kernels-community/finegrained-kernels). Each format has its
own config that quantizes a whole model. [`FineGrainedConfig`] mixes several formats in one model.

This page covers what the formats share. Each format page covers its layouts and activation formats.

## Requirements

```bash
pip install --upgrade accelerate "kernels>=0.17.2,<0.18"
```

Quantized compute needs the `kernels` package and an Intel XPU or a CUDA GPU. A CUDA GPU needs compute capability 8.9
or newer. Without them, a pre-quantized checkpoint is dequantized to bf16 as it loads.
Quantizing on the fly raises an error.

## Load, quantize and save

A pre-quantized checkpoint carries its quantization config, so it loads without one. Pass a format config to
[`~PreTrainedModel.from_pretrained`] to quantize a full-precision checkpoint as it loads. Save the result with
[`~PreTrainedModel.save_pretrained`]. The saved model reloads without a config.

```py
from transformers import AutoModelForCausalLM, Fp8Config

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype="auto", device_map="auto", quantization_config=Fp8Config()
)
model.save_pretrained("qwen3-8b-fp8")
```

A config passed with a pre-quantized checkpoint does not requantize it. Its `activation_format` picks the kernels the
weights run on. Its loading options also apply.

- `dequantize=True` loads the weights in full precision, for example to fine-tune them.
- `modules_to_not_convert` replaces the checkpoint's list of modules kept in full precision. If it is unset, the
  checkpoint's list stays.

The other options, such as the block size or the scale format, come from the checkpoint. Passing a config warns that
they are ignored.

## Mix formats in one model

The `dtype_plan` of [`FineGrainedConfig`] maps module name patterns to format configs. Each matching module is
quantized in the format of its entry. A typical plan puts the MoE experts in a 4-bit format and every other linear in
8 bits. The experts hold most of the weights.

```py
from transformers import AutoModelForCausalLM, Fp8Config, FineGrainedConfig, Mxfp4Config

quantization_config = FineGrainedConfig(
    dtype_plan={
        r"\.experts($|\.)": Mxfp4Config(),
        ".*": Fp8Config(),
    },
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B", dtype="auto", device_map="auto", quantization_config=quantization_config
)
```

- A pattern matches module names with `re.search`.
- A targeted entry wins over the `".*"` catch-all, whatever the order of the plan. A plan saved to `config.json`
  loses its order.
- Two targeted entries that match the same module raise an error.
- A module that no entry matches stays in full precision. So does a module listed in `modules_to_not_convert`.
- `dequantize` and `modules_to_not_convert` belong to the plan, not to its entries.

Some checkpoints load as a [`FineGrainedConfig`] without passing one.

- DeepSeek-V4 ships block-FP8 weights with MXFP4 routed experts. Its checkpoint declares `quant_method="fp8"` and its
  model config declares `expert_dtype="fp4"`. DeepSeek-V4.1 declares `expert_dtype` in its quantization config instead.
  Together they read as a plan of MXFP4 experts over block FP8. See [DeepSeek-V4](../model_doc/deepseek_v4) for the
  full plan.
- A ModelOpt export with `quant_algo="NVFP4"` can give different modules different formats in its `config_groups`. It
  reads as a plan with one entry per group target.
- A ModelOpt export with `quant_algo="MIXED_PRECISION"` reads its `quantized_layers` mapping. Each module name covers
  that module and its descendants. Modules missing from the mapping stay in full precision. The supported algorithms
  are `FP8`, `NVFP4`, `MXFP8`, `MXFP4`, `W4A16_NVFP4` and `W4A16_MXFP4`. `FP8` is calibrated per-tensor FP8. The
  `W4A16` variants are weight-only.

```json
{
  "quant_method": "modelopt",
  "quant_algo": "MIXED_PRECISION",
  "quantized_layers": {
    "layers.8.attention.wq": {"quant_algo": "FP8"},
    "layers.9.experts": {"quant_algo": "NVFP4", "group_size": 16}
  }
}
```

Module names resolve through the checkpoint renames of the model. Loading raises an error in these cases.

- An assignment matches no supported linear, expert or embedding module.
- A per-expert assignment does not fit the fused expert module of the model.
- The kernels do not serve the assigned format.

## Tensor and expert parallelism

Pass a [`DistributedConfig`] to [`~PreTrainedModel.from_pretrained`] to shard a quantized model. Each scale is sharded
with its weight. Quantized experts use [masking and all-reduce](../expert_parallelism) expert parallelism instead of
token dispatch. This is why `ep_size` must equal `tp_size`.

```py
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-FP8", dtype="auto", distributed_config=DistributedConfig(tp_size=8, ep_size=8)
)
```

## Kernel autotuning

The kernels tune each new input shape and cache the chosen configuration. Set `FINEGRAINED_AUTOTUNE_CONFIG=fast`
before the kernels load to shrink the search. Use it for correctness tests or quick experiments. Leave it unset when
measuring performance. See the
[kernel autotuning documentation](https://huggingface.co/kernels-community/finegrained-kernels#autotuning).
