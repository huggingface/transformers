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

[FP8](./fp8), [MXFP8](./mxfp8), [MXFP4](./mxfp4) and [NVFP4](./nvfp4) are block-scaled formats: low-precision
weights with one scale per block or group of values. They share one integration and one kernel bundle,
[finegrained-kernels](https://huggingface.co/kernels-community/finegrained-kernels), and each has its own config that
quantizes a whole model. [`FineGrainedConfig`] mixes them in one model. This page covers what they share; the format
pages cover each format's layouts and activation formats.

## Requirements

```bash
pip install --upgrade accelerate "kernels>=0.17.2,<0.18"
```

- Quantized compute needs a CUDA GPU with compute capability 8.9 or newer (Ada, Hopper, Blackwell) or an Intel XPU.
  Weight-only MXFP4 (W4A16, how GPT-OSS runs) only needs compute capability 7.5.
- Without a GPU, or below the required compute capability, a pre-quantized checkpoint is dequantized to bf16 as it
  loads. Quantizing a full-precision checkpoint on the fly raises an error without a GPU, and below the required
  compute capability loads the model unquantized.
- The kernels are loaded from the Hub on the first forward. A missing or incompatible `kernels` package is not caught
  at load time: the first forward raises an `ImportError`.

## Load, quantize and save

A pre-quantized checkpoint carries its quantization config, so it loads with no config. Pass a format config to
[`~PreTrainedModel.from_pretrained`] to quantize a full-precision checkpoint as it loads, and save it with
[`~PreTrainedModel.save_pretrained`]; the saved model reloads with no config.

```py
from transformers import AutoModelForCausalLM, Fp8Config

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype="auto", device_map="auto", quantization_config=Fp8Config()
)
model.save_pretrained("qwen3-8b-fp8")
```

A config passed with a pre-quantized checkpoint does not requantize it. Its `activation_format` picks the kernels the
weights run on, and its loading options apply:

- `dequantize=True` loads the weights dequantized to full precision, for example to fine-tune them.
- `modules_to_not_convert` replaces the checkpoint's list of modules kept in full precision; left unset, the
  checkpoint's list stays.

The format options (block size, scale format, activation scheme) are the checkpoint's own. A passed value that
differs from them is ignored with a warning.

## Mix formats in one model

[`FineGrainedConfig`]'s `dtype_plan` maps module name patterns to the format config each matching module is
quantized in. A typical plan holds the MoE experts, which carry most of the weights, in a 4-bit format and every other
linear in 8 bits.

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

- A pattern names the modules it matches with `re.search`. A targeted entry wins over the `".*"` catch-all, whatever
  order the plan is written in, since a plan saved to `config.json` loses its order. Two targeted entries matching one
  module raise an error.
- A module no entry matches, or that `modules_to_not_convert` names, stays in full precision.
- The loading options (`dequantize`, `modules_to_not_convert`) belong to the plan, not to its entries.

These checkpoints load as a [`FineGrainedConfig`] without one being passed:

- DeepSeek-V4 ships block-FP8 weights with MXFP4 routed experts. Its checkpoint declares `quant_method="fp8"` and its
  model config `expert_dtype="fp4"` (DeepSeek-V4.1 declares it in its quantization config), which together read as a
  plan of MXFP4 experts over block FP8. See [DeepSeek-V4](../model_doc/deepseek_v4) for the plan written out.
- A ModelOpt export labelled `quant_algo="NVFP4"` whose `config_groups` give different modules different formats
  reads as a plan with one entry per group target.
- ModelOpt `quant_algo="MIXED_PRECISION"` reads its `quantized_layers` mapping. Each module name covers that module
  and its descendants, and modules absent from the mapping stay in full precision. The supported algorithms are `FP8`
  (calibrated per-tensor FP8), `NVFP4`, `MXFP8`, `MXFP4`, and the
  `W4A16_NVFP4` / `W4A16_MXFP4` weight-only variants.

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

Module names resolve through the model's checkpoint renames. An assignment that matches no supported linear, expert,
or embedding module raises an error, as does a per-expert assignment the model's fused expert module cannot hold.
Unsupported algorithms and config groups, AWQ smoothing (`pre_quant_scale`), SVD residuals, asymmetric zero points,
sparse or transformed layouts, and compressed scale encodings such as `weight_scale_encoding="csf"` raise an error too.

## Tensor and expert parallelism

Pass a [`DistributedConfig`] to [`~PreTrainedModel.from_pretrained`] to shard a quantized model; each
scale is sharded with its weight. Quantized experts use [masking and all-reduce](../expert_parallelism) expert
parallelism instead of token dispatch, so `ep_size` must equal `tp_size`.

```py
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-FP8", dtype="auto", distributed_config=DistributedConfig(tp_size=8, ep_size=8)
)
```

## Kernel autotuning

The kernels tune each new input shape and cache the chosen configuration. For correctness tests or quick experiments,
set `FINEGRAINED_AUTOTUNE_CONFIG=fast` before the kernels load to shrink the search. Leave it unset when measuring
performance. See the [kernel autotuning documentation](https://huggingface.co/kernels-community/finegrained-kernels#autotuning).
