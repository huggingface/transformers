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

This page adds a format the way a new vLLM / ModelOpt dtype usually arrives: as a checkpoint whose layers store
low-precision weights plus scales. It goes step by step, from the cheapest case to the most work, and stops as soon as
the format runs. Most formats stop at step 1 or 2.

Loading is already generic: renaming checkpoint names, merging per-expert tensors into fused experts, tensor parallel
sharding, per-layer mixing of formats and `dequantize=True` all come from the quantization plan. A format only says
what it stores, which module computes with it, and how its tensors become that module's.

```text
step  the format is…                                   you add                                     ~lines
1     FP8 with another block size / scale format        a config (or a plan entry)                  0-5
2     another dtype the FP8 kernels can run             a module subclass                           10
3     a new kernel                                      a hub kernel entry + module + ops           100
4     stored per layer in a mixed checkpoint            a `MODELOPT_ALGOS` / `PLAN_CONVERTERS` entry  1-5
```

## 1. It is FineGrainedFP8 with other settings

Take MXFP8: e4m3 weights, one ue8m0 scale per 32 values of a row. That is FP8 with a `(1, 32)` block and power-of-two
scales, which [`FineGrainedFP8Config`] already describes:

```py
from transformers import AutoModelForCausalLM, FineGrainedFP8Config

config = FineGrainedFP8Config(weight_block_size=(1, 32), scale_fmt="ue8m0")
model = AutoModelForCausalLM.from_pretrained("my-org/model-mxfp8", quantization_config=config)
```

Why it works: `FP8Linear` / `FP8Experts` allocate their scales from `block_size` and `scale_fmt`, and the
`kernels-community/finegrained-fp8` kernels take any block size. If the checkpoint names its scales differently (e.g.
`weight_scale` instead of `weight_scale_inv`), that is a rename, not a new format: add it to the quantizer's
`renamed_names`, which renames only on the layers of that quantizer.

Check the kernel actually supports your block size on the GPUs you target before stopping here.

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
    serialized_names = {"weight": ["weight", "weight_scale_inv"]}  # what a param is stored as
    renamed_names = {"scale": "weight_scale_inv"}                 # DeepSeek-V4 saves `.scale`

    def quantized_module(self, name, module):
        if name.endswith(".experts"):
            return Mxfp4Experts(module.config)
```

Why nothing else: `serialized_names` is enough for the plan to load each expert's scale with its weight, through the
same merge and concatenation the model applies to the weights. No load op is needed because the module stores the
tensors exactly as the checkpoint does.

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
tensors of one param, after the model's own merges. `dequantize` runs before them, so `dequantize=True` (and GPUs
without the kernel) get plain weights:

```py
class NVFP4HfQuantizer(HfQuantizer):
    serialized_names = {"weight": ["weight", "weight_scale", "weight_scale_2", "input_scale"]}

    def get_load_operations(self):
        # `quantize`: a full-precision checkpoint quantized on load (optional)
        return {"deserialize": NVFP4Deserialize, "dequantize": NVFP4Dequantize, "quantize": NVFP4Quantize}
```

Why ops and not a post-processing pass: the plan gives each param its ops while loading, so every format runs the same
way under tensor parallelism (each op sees this rank's shard) and nothing walks the model after loading.

For MoE checkpoints, the experts module is the main extra work: a grouped forward over the kernel (see
`NVFP4Experts.grouped_forward`), with a per-expert eager loop as the fallback.

## 4. It is stored per layer in a mixed checkpoint

ModelOpt's `MIXED_PRECISION` checkpoints list a `quant_algo` per layer. Map the name to a config:

```py
MODELOPT_ALGOS = {
    "NVFP4": NVFP4Config,
    "FP8": lambda: FineGrainedFP8Config(activation_scheme="static", weight_block_size=None),
    "MXFP8": lambda: FineGrainedFP8Config(weight_block_size=(1, 32), scale_fmt="ue8m0"),  # step 1
}
```

A checkpoint format that describes its layers some other way gets a function in `PLAN_CONVERTERS` that turns its
`quantization_config` into a plan, `{module pattern: config}` (see `fp4_experts_plan` for DeepSeek-V4's
`expert_dtype`). The plan then matches every module once, renames the patterns to the model's names, and loads each
layer through its own quantizer.

## Checklist

- `tests/quantization/<format>/`: load a tiny checkpoint quantized and with `dequantize=True`, and compare the weights to
  an independent dequantization.
- A kernel test on the GPUs you target, with more than one token per expert (grouped kernels often fail only then).
- A page in `docs/source/en/quantization/` and an entry in `_toctree.yml`.
