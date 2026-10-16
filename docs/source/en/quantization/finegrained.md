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

[Block FP8](./blockfp8), [MXFP8](./mxfp8), [MXFP4](./mxfp4) and [NVFP4](./nvfp4) each have their own config and
quantize a whole model on their own. [`FineGrainedConfig`] mixes them in one model: its `dtype_plan` maps module name
patterns to the format config each matching module is quantized in. A typical plan holds the MoE experts, which carry
most of the weights, in a 4-bit format and every other linear in 8 bits.

```py
from transformers import AutoModelForCausalLM, BlockFp8Config, FineGrainedConfig, Mxfp4Config

quantization_config = FineGrainedConfig(
    dtype_plan={
        r"\.experts($|\.)": Mxfp4Config(),
        ".*": BlockFp8Config(),
    },
    modules_to_not_convert=["lm_head"],
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B", dtype="auto", device_map="auto", quantization_config=quantization_config
)
model.save_pretrained("qwen3-30b-a3b-mixed")
```

The saved model records its plan, so it reloads with no config.

> [!TIP]
> The requirements are those of the formats in the plan: a CUDA GPU with compute capability 8.9 or newer or an Intel
> XPU, and the [kernels](https://github.com/huggingface/kernels) package.

## How a plan resolves

- A pattern names the modules it matches with `re.search`. A targeted entry wins over the `".*"` catch-all, whatever
  order the plan is written in, since a plan saved to `config.json` loses its order. Two targeted entries matching one
  module raise an error.
- A module no entry matches, or that `modules_to_not_convert` names, stays in full precision.
- `modules_to_convert` names embedding tables to hold in FP8 with one per-tensor scale, whatever the plan's formats.
- `dequantize=True` loads every entry dequantized to full precision.

## Checkpoints that mix formats

These load as a [`FineGrainedConfig`] without one being passed:

- DeepSeek-V4 ships block-FP8 weights with MXFP4 routed experts. Its checkpoint declares `quant_method="fp8"` and its
  model config `expert_dtype="fp4"`, which together read as a plan of MXFP4 experts over block FP8 (DeepSeek-V4.1
  declares `expert_dtype` in its quantization config instead, and reads the same way). See the
  [DeepSeek-V4](../model_doc/deepseek_v4) page for the plan written out.
- A ModelOpt export labelled `quant_algo="NVFP4"` whose `config_groups` give different modules different formats
  reads as a plan with one entry per group target. Other `quant_algo` values, such as `"MIXED_PRECISION"`, raise an
  error.

To change one format of such a checkpoint, pass the plan you want; to dequantize it, pass
`FineGrainedConfig(dequantize=True)`.

## Kernel autotuning

Fine-grained kernels tune each new input shape and cache the chosen configuration. For correctness tests or quick
experiments, set `FINEGRAINED_AUTOTUNE_TRIALS=1` before loading the kernel bundle to benchmark only one configuration
per shape. Leave it unset when measuring performance so the tuner can search its full budget. See the
[kernel autotuning documentation](https://huggingface.co/kernels-community/finegrained-kernels#autotuning).
