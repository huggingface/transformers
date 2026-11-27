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

A quantized format is a recipe: what a checkpoint stores, which modules run it, and how its weights are read and
written. Each recipe has its own config and quantizer, even if an existing kernel could run it. A config describes one
recipe, and its options (block size, activation scheme, ...) are settings of that recipe, not other formats. Don't
load a new format through an existing quantizer such as fine-grained FP8.

| Part | Where | Example |
|---|---|---|
| Config | `utils/quantization_config.py` | [`NVFP4Config`] |
| Quantizer | `quantizers/quantizer_<format>.py` | `NVFP4HfQuantizer`, `Mxfp4HfQuantizer` |
| Modules, kernels, load ops | `integrations/<format>.py` | `NVFP4Linear`, `NVFP4Deserialize` |
| Checkpoint scheme reader (optional) | `quantizers/schema_readers.py` | `modelopt_plan`, `fp4_experts_plan` |

See [Contribute](./contribute) for the config, registration and tests. This page covers the format itself.

> [!WARNING]
> MXFP8 checkpoints currently load through fine-grained FP8 (`quant_method: "mxfp8"`). Don't take it as a model for a
> new format: MXFP8 is not just another dtype and granularity, it is natively accelerated on Blackwell and needs its
> scales interleaved and swizzled for the best performance.

## 1. The quantizer: what the checkpoint stores

The quantizer declares what the checkpoint stores for each parameter and how those tensors are loaded. A simplified
NVFP4 quantizer looks like this:

```py
class NVFP4HfQuantizer(HfQuantizer):
    # what each param is stored as: a linear's `weight`, plus its scales
    serialized_names = {"weight": ["weight", "weight_scale", "weight_scale_2", "input_scale"]}
    checkpoint_renames = {"experts.*.input_scale": "weight_input_scale"}

    def get_load_operations(self):
        return {"deserialize": NVFP4Deserialize, "dequantize": NVFP4Dequantize}

    def quantized_module(self, name, module):
        if isinstance(module, torch.nn.Linear):
            return NVFP4Linear(module.in_features, module.out_features)
```

- `serialized_names` lists the tensors a param is stored as. They load together, with the same ops.
- `checkpoint_renames` maps globs on the last names of a checkpoint key to ours, first match wins. It is applied per layer
  (`rename_layer`), before anything else, so the rest of loading only sees our names. A glob can look further up the
  key, e.g. `{"*.input_scale": "weight_input_scale"}` renames only the `input_scale` tensors.
- `get_load_operations` returns the op classes for `"deserialize"` and `"dequantize"`. `get_load_operation()` picks the
  one for the config (`dequantize=True` or not). Here `NVFP4Deserialize` turns the checkpoint's packed weights and
  scales into the swizzled layout that `NVFP4Linear` runs.
- `quantized_module` is called on meta for each module the plan gives the quantizer; returning `None` keeps the module.
  All quantizers of a plan swap their modules in one pass, so a quantizer never walks the model itself. Only formats
  that swap modules this way can be mixed with others in a plan.

## 2. The modules and kernels

The module holds the tensors under the checkpoint's names and runs the format's kernel. Add it to
`_HUB_KERNEL_MAPPING` in `integrations/hub_kernels.py`, then load it lazily where the module needs it:

```py
_HUB_KERNEL_MAPPING = {
    "nvfp4": {"repo_id": "kernels-community/nvfp4-gemm", "version": 1},
}

def load_nvfp4_kernel():
    return lazy_load_kernel("nvfp4")
```

NVFP4 needs its block scales swizzled for the GEMM and has a second, per-tensor scale, so `NVFP4Linear` keeps both
the checkpoint's scales and the swizzled ones (simplified):

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

## 3. The load ops

The load ops turn the checkpoint's tensors into the module's. `deserialize` runs on the pre-quantized tensors of one
param, after the model's own merges. `dequantize` runs before them, on the weight and its scales collected together,
so `dequantize=True` gets plain weights and needs no kernel (or GPU):

```py
class NVFP4HfQuantizer(HfQuantizer):
    serialized_names = {"weight": ["weight", "weight_scale", "weight_scale_2", "input_scale"]}

    def get_load_operations(self):
        return {"deserialize": NVFP4Deserialize, "dequantize": NVFP4Dequantize}

    def get_quantize_ops(self):  # optional: quantize a full-precision checkpoint on load
        return NVFP4Quantize(self)
```

A converter pattern ending in `.weight` also matches the tensors named after it (`weight_scale_inv`, ...), and they
land in the target named like it. `update_weight_transform` attaches the load op to a param once its first key matches
a converter: after the converter's ops for deserialize, before them for dequantize.

Under tensor parallelism, a param named after a planned one shards like it. A format whose scales can't be sharded
that way overrides `update_parallel_plan`, e.g. NVFP4 replicates its modules.

## 4. Read the checkpoint's own scheme

Transformers' format for a model quantized with several methods is a [quantization plan](./overview#quantization-plans),
`{module pattern: config}`. Some checkpoints describe their quantization in their own scheme instead, for example:

| Checkpoint scheme | Reader | Plan |
|---|---|---|
| ModelOpt `quant_algo`, or `MIXED_PRECISION` with one per layer under `quantized_layers` | `modelopt_plan` | each layer's `quant_algo` → its config, e.g. `NVFP4` → [`NVFP4Config`] |
| DeepSeek-V4's FP8 config with `expert_dtype: "fp4"` | `fp4_experts_plan` | `{"*.mlp.experts": Mxfp4Config(), "*": <the FP8 config>}` |

Such a scheme gets a small reader in `quantizers/schema_readers.py`, converting it into a plan, instead of a special
case in the loading code. A reader takes the checkpoint's `quantization_config` dict and returns a plan, or `None` for
configs it doesn't handle:

```py
def fp4_experts_plan(config: dict) -> dict | None:
    "DeepSeek-V4's FP8 checkpoints, whose routed experts are MXFP4 (`expert_dtype: fp4`)"
    if config.get("quant_method") != "fp8" or config.get("expert_dtype") != "fp4":
        return None
    return {
        "*.mlp.experts": Mxfp4Config(),
        "*": {key: value for key, value in config.items() if key != "expert_dtype"},
    }


SCHEMA_READERS = [modelopt_plan, fp4_experts_plan]  # tried in order, the first plan wins
```

Add the reader to `SCHEMA_READERS`. `get_hf_quantizer` calls `read_quantization_schema(config)`, which returns the plan
of the first reader that handles the config. The plan's names are checkpoint module names: the
`QuantizationPlanHfQuantizer` renames them to the model's, like the keys of their params. A new method in an existing
scheme, e.g. a new ModelOpt `quant_algo`, maps to the config of its own quantizer.

`AutoHfQuantizer.from_plan` then builds one quantizer per distinct config, and the plan gives each module and param the
quantizer of its pattern, for the module swap, the load ops and the parallel plan alike.
