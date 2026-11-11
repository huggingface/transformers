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

# Add a quantized format

A new quantization format has 3 things to define: what it stores, which module to replace, how to read and write the weights.
The following should guide you in your addition.

## 1. It is `FineGrainedFP8` with other settings

Take MXFP8: e4m3 weights, one ue8m0 scale per 32 values of a row. That is FP8 with a `(1, 32)` block and power-of-two scales, which [`FineGrainedFP8Config`] already describes:

```py
from transformers import AutoModelForCausalLM, FineGrainedFP8Config

config = FineGrainedFP8Config(weight_block_size=(1, 32), scale_fmt="ue8m0")
model = AutoModelForCausalLM.from_pretrained("my-org/model-mxfp8", quantization_config=config)
```

Check the kernel actually supports your block size on the GPUs you target, and most probably you just have to add a setting to `FineGrainedFP8`.
We'll add anything that reflects a "setting" and not an actual difference in computation.

## 2. It is another dtype the FP8 kernels can run

DeepSeek-V4's MXFP4 experts store two e2m1 values per byte, with a ue8m0 scale per 32. The finegrained-fp8 kernels run
them as W4A8, so the module only changes how its params are allocated:

```py
class Mxfp4Experts(FP8Experts):
    "MXFP4 experts: e2m1 pairs packed in `int8`, a ue8m0 scale per 32, run W4A8 on the FP8 kernels"

    def __init__(self, config, has_bias=False, has_gate=True):
        super().__init__(
            config, block_size=(1, 32), scale_fmt="ue8m0", has_bias=has_bias, has_gate=has_gate, weight_dtype=torch.int8
        )
```

The quantizer then says which modules become `Mxfp4Experts` and what the checkpoint stores (simplified):

```py
class Mxfp4HfQuantizer(HfQuantizer):
    # what each param is stored as: a linear's `weight`, and the fused experts' params
    serialized_names = {
        "weight": ["weight", "weight_scale_inv"],
        "gate_up_proj": ["gate_up_proj", "gate_up_proj_scale_inv"],
        "down_proj": ["down_proj", "down_proj_scale_inv"],
    }
    renamed_names = {"scale": "weight_scale_inv"}  # DeepSeek-V4 saves `.scale`

    def get_load_operations(self):
        return {"deserialize": Mxfp4Deserialize, "dequantize": Mxfp4Dequantize}

    def quantized_module(self, name, module):
        if name.endswith(".experts"):
            return Mxfp4Experts(module.config)
```

- `serialized_names` lists the tensors a param is stored as. They load together, with the same ops.
- `renamed_names` maps globs on the last names of a checkpoint key to ours, first match wins. It is applied per layer
  (`rename_layer`), before anything else, so the rest of loading only sees our names. A glob can look further up the
  key, e.g. NVFP4's `{"experts.*.input_scale": "weight_input_scale"}` renames only the experts' `input_scale`.
- `get_load_operations` returns the op classes for `"deserialize"` and `"dequantize"`. `get_load_operation()` picks the
  one for the config (`dequantize=True` or not). Here `Mxfp4Deserialize` passes DeepSeek-V4's tensors through as is,
  since `Mxfp4Experts` stores them as the checkpoint does.
- `quantized_module` is called on meta for each module the plan gives the quantizer; returning `None` keeps the module.
  All quantizers of a plan swap their modules in one pass, so a quantizer never walks the model itself. Only formats
  that swap modules this way can be mixed with others in a plan.

The experts need nothing else: the model's own converters take each expert's scales along with its weight. A converter
pattern ending in `.weight` also matches the tensors named after it (`weight_scale_inv`, ...), and they land in the
target named like it:

```text
checkpoint keys (after rename_layer)              model converter                   targets
mlp.experts.0.gate_proj.weight         ─┐
mlp.experts.0.up_proj.weight            ├─ MergeModulelist + Concatenate ─→  mlp.experts.gate_up_proj
mlp.experts.1.gate_proj.weight ...     ─┘   then deserialize
mlp.experts.0.gate_proj.scale  → weight_scale_inv ─┐
mlp.experts.0.up_proj.scale    → weight_scale_inv  ├─ same ops ──────────→  mlp.experts.gate_up_proj_scale_inv
...                                               ─┘
```

With `dequantize=True` the converter reads the weights and their scales into the one target `gate_up_proj`, and the
dequantize op runs first, on each expert, before the merge. `update_weight_transform` attaches the load op to a param
once its first key matches a converter: after the converter's ops for deserialize, before them for dequantize.

Under tensor parallelism, a param named after a planned one (`gate_up_proj_scale_inv`, GPT-OSS's `gate_up_proj_blocks`)
shards like it. A format whose scales can't be sharded that way overrides `update_parallel_plan`, e.g. NVFP4 replicates
its modules.

## 3. It needs a new kernel

NVFP4 does not fit the FP8 kernels: its block scales must be swizzled for the GEMM and it has a second, per-tensor scale.
Three pieces:

**The kernel**, from the Hub, like any [kernels](https://huggingface.co/docs/kernels) kernel. Add it to
`_HUB_KERNEL_MAPPING` in `integrations/hub_kernels.py`, then load it lazily where the module needs it:

```py
_HUB_KERNEL_MAPPING = {
    "finegrained-fp8": {"repo_id": "kernels-community/finegrained-fp8", "version": 4},
    "nvfp4": {"repo_id": "kernels-community/nvfp4-gemm", "version": 1},
}

def load_nvfp4_kernel():
    return lazy_load_kernel("nvfp4")
```

**The module**, holding the tensors under the checkpoint's names and calling the kernel (simplified):

```py
class NVFP4Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Buffer(torch.empty(out_features, in_features // 2, dtype=torch.uint8))  # e2m1 pairs
        self.weight_scale = nn.Buffer(torch.empty(out_features, in_features // 16, dtype=torch.uint8))
        self.weight_scale_2 = nn.Buffer(torch.empty(1))
        self.weight_sf = nn.Buffer(...)  # `weight_scale`, swizzled for the GEMM

    def forward(self, input):
        return nvfp4_linear(load_nvfp4_kernel(), input, self.weight, self.weight_sf, ...)
```

**The load ops**, which turn the checkpoint's tensors into the module's. `deserialize` runs on the pre-quantized
tensors of one param, after the model's own merges. `dequantize` runs before them, on the weight and its scales
collected together, so `dequantize=True` gets plain weights and needs no kernel (or GPU):

```py
class NVFP4HfQuantizer(HfQuantizer):
    serialized_names = {"weight": ["weight", "weight_scale", "weight_scale_2", "input_scale"]}

    def get_load_operations(self):
        return {"deserialize": NVFP4Deserialize, "dequantize": NVFP4Dequantize}

    def get_quantize_ops(self):  # optional: quantize a full-precision checkpoint on load
        return NVFP4Quantize(self)
```

For MoE checkpoints, the experts module replacement requires a bit more work as you need a grouped forward over the kernel (see `NVFP4Experts.grouped_forward`), with a per-expert eager loop as the fallback.

## 4. It is stored per layer in a mixed checkpoint

For example ModelOpt checkpoints name their format with a `quant_algo`: one for the whole model, or with `MIXED_PRECISION`, one per layer under `quantized_layers`. Both are converted to `transformers'` format using the following:

```py
MODELOPT_ALGOS = {
    "NVFP4": NVFP4Config,
    "W4A16_NVFP4": NVFP4Config,
    "FP8": lambda: FineGrainedFP8Config(activation_scheme="static", weight_block_size=None),
    "FP8_PB_WO": lambda: FineGrainedFP8Config(activation_scheme="none", weight_block_size=(128, 128)),
}
```

A new `quant_algo` that is another FP8 variant (step 1) is one line here, e.g.
`"MXFP8": lambda: FineGrainedFP8Config(weight_block_size=(1, 32), scale_fmt="ue8m0")`. 

Layers ModelOpt leaves out keep the base `quant_method` if it is one (e.g. DeepSeek-V4's block `fp8`), else stay unquantized like its `ignore`d ones.

A checkpoint format that describes its layers some other way adds a function turning its `quantization_config` into a
plan, `{module pattern: config}`, or `None` for configs it doesn't handle (like `fp4_experts_plan` for DeepSeek-V4's
`expert_dtype: "fp4"`), and calls it next to `modelopt_plan` in `get_hf_quantizer`. The plan's names are checkpoint
module names: `AutoHfQuantizer` renames them to the model's, like the keys of their params.

`AutoHfQuantizer.from_plan` then builds one quantizer per distinct config, and the plan gives each module and param the
quantizer of its pattern, for the module swap, the load ops and the parallel plan alike.
