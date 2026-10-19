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

# Contribute

Transformers supports many quantization methods such as QLoRA, GPTQ, LLM.int8, and AWQ. However, there are still many more quantization approaches that haven't been integrated yet. To make adding and using these quantization methods with Transformers easier, use the [`~quantizers.HfQuantizer`] class. [`~quantizers.HfQuantizer`] is designed to be an internal helper class for adding a quantization method instead of something applied to every PyTorch module.

This guide will show you how to integrate a new quantization method with [`~quantizers.HfQuantizer`].

## Requirements

Before integrating a new quantization method into Transformers, ensure the method meets the following requirements. Only quantization methods that can be run with PyTorch modules are supported.

- The quantization method is available through a Python package that is pip-installable (it is also fine if you can only install the package from source). Ideally, pre-compiled kernels are included in the pip package.
- The method can run on commonly-used hardware (CPU, GPU, etc.).
- The method is wrapped in a [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) ([`~bitsandbytes.nn.Linear8bitLt`], [`~bitsandbytes.nn.Linear4bit`]), and the quantized linear layer should have the following definition.

  ```py
  class Linear4bit(nn.Module):
      def __init__(self, ...):
          ...

      def forward(self, x):
          return my_4bit_kernel(x, self.weight, self.bias)
  ```

  This way, Transformers models are easily quantized by replacing instances of [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) with a target class.

- The quantization method should be serializable. You can save the quantized weights locally or push them to the Hub.
- Make sure the package containing the quantization kernels/primitive is stable (no frequent breaking changes).

Some quantization methods may require "pre-quantizing" the model through data calibration (AWQ). In this case, we prefer to only support inference in Transformers and let the third-party library maintained by the ML community deal handle the model quantization itself.

## Create new HFQuantizer class

0. The best starting point would be to have a look at another quantization method such as Finegrained Fp8. You will have to update or create three files in total: the [config file](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py), the [integration file](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/finegrained_fp8.py) and the [quantizer file](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_finegrained_fp8.py).

1. Create a new quantization config class inside [src/transformers/utils/quantization_config.py](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py). Add the new quantization config to the [\_import_structure](https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py) inside Transformers' [src/transformers/\_\_init\_\_.py](https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py) file.

2. Create a new file inside [src/transformers/quantizers/](https://github.com/huggingface/transformers/tree/main/src/transformers/quantizers) named `quantizer_your_method.py`, and make it inherit from [`~quantizers.HfQuantizer`]. Make sure to add the new quantizer and quantization config in the quantization auto-mapping in [src/transformers/quantizers/auto.py](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/auto.py).

3. Define the following class attributes and property methods for your quantization method:

   - `requires_calibration`: Whether the quantization method requires a data calibration process. If set to `True`, you can only support inference (with quantized weights) and not inference and quantization.
   - `is_serializable`: A property method to determine whether the method is serializable or not.
   - `is_trainable`: A property method to determine whether you can fine-tune models on top of the quantization method (with or without PEFT approaches).

4. Write the `validate_environment` and `update_dtype` methods. These methods are called before creating the quantized model to ensure users use the right configuration. Refer to other quantizers for an example of it is implemented.

5. Write the `_process_model_before_weight_loading` method. In Transformers, the quantized models are initialized first on the `"meta"` device before loading the weights. This means the `_process_model_before_weight_loading` method takes care of manipulating the model skeleton to replace some modules ([nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)) with the target modules (quantization modules).

You can define module replacement logic or any other utility method by creating a new file in [transformers/src/integrations/](https://github.com/huggingface/transformers/tree/main/src/transformers/integrations) and exposing the relevant methods in that folder's `__init__.py` file.

6. Set `serialized_names` to the names a quantized parameter is saved under, and add the load methods for each case your method supports. See [Choose how weights load](#choose-how-weights-load) for which methods to add.

7. Set `quantized_element_size` to the bytes per element of a quantized weight, for example `1` for FP8. Transformers uses it to estimate memory when it builds a device map.

8. Write the `_process_model_after_weight_loading` method if needed. This method enables implementing additional features that require manipulating the model after loading the weights.

9. Document everything! Make sure your quantization method is documented by adding a new file under `docs/source/en/quantization`.

10. You should add tests by adding the package in our nightly Dockerfile inside `docker/transformers-quantization-latest-gpu` and then adding a new test file in `tests/quantization/xxx`. Feel free to check out existing quantization methods to see how it is implemented.

## Files overview

| File                                         | Purpose                                                                                          |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `utils/quantization_config.py`               | Define `YourMethodConfig` inheriting from `QuantizationConfigMixin`                              |
| `quantizers/quantizer_your_method.py`        | Implement `YourMethodHfQuantizer` inheriting from `HfQuantizer`                                  |
| `integrations/your_method.py`                | Implement `ConversionOps` subclasses and helper functions                                        |
| `quantizers/auto.py`                         | Register quantizer and config in `AUTO_QUANTIZER_MAPPING` and `AUTO_QUANTIZATION_CONFIG_MAPPING` |
| `docs/source/en/quantization/your_method.md` | Document usage for users                                                                         |
| `tests/quantization/your_method/`            | Add integration tests                                                                            |

## Choose how weights load

Transformers loads every checkpoint tensor through [`WeightConverter`] and [`ConversionOps`], as described in [Dynamic weight loading](../weightconverter). Your quantizer adds operations to this pipeline. Which operation runs depends on the checkpoint and the quantization config.

| Case | When it applies | What you implement | Where it runs |
|---|---|---|---|
| Quantize | The checkpoint is full precision (`pre_quantized=False`). | `get_quantize_ops` and `param_needs_quantization` | After the model's conversion ops, on each parameter that `param_needs_quantization` accepts |
| Deserialize | The checkpoint is pre-quantized, and its tensors need to be rebuilt or recast. | `serialized_names` and a `"deserialize"` op in `get_load_operations` | On parameters that the model's conversion mapping doesn't match |
| Dequantize | The checkpoint is pre-quantized, and the config sets `dequantize=True`. | `serialized_names` and a `"dequantize"` op in `get_load_operations` | Before the model's conversion ops, and on parameters that the mapping doesn't match |

A pre-quantized checkpoint doesn't always need a load op. If your quantized modules store the saved tensors as parameters with the same names, the loader copies them as they are. FP8 works this way, because `FP8Linear` has both a `weight` and a `weight_scale_inv` parameter, so it only defines a dequantize op.

### Declare serialized names

`serialized_names` maps the name of a full-precision parameter to the names it's saved under in a quantized checkpoint. FP8 saves each `weight` with its scales.

```py
serialized_names = {"weight": ["weight", "weight_scale_inv", "activation_scale"]}
```

MXFP4 saves expert weights as blocks and scales.

```py
serialized_names = {
    "gate_up_proj": ["gate_up_proj_blocks", "gate_up_proj_scales"],
    "down_proj": ["down_proj_blocks", "down_proj_scales"],
}
```

Make `serialized_names` a property when the names depend on the config or the checkpoint. The torchao quantizer reads them from the checkpoint metadata.

[`~quantizers.HfQuantizer`] builds the rest of the loading setup from this dict.

- `get_weight_conversions` creates one [`WeightConverter`] per entry. It collects the saved names into the parameter and runs your load op.
- `update_weight_conversions` extends the model's own conversion mapping to load the saved names. See [Load MoE and fused weights](#load-moe-and-fused-weights).
- `update_parallel_plan` adds the saved names to the model's tensor parallel and expert parallel plans, with the same style as their parameter. `experts.gate_up_proj_scale_inv` is sharded like `experts.gate_up_proj`, so you don't need to update the plans yourself.

### Quantize on the fly

`get_quantize_ops` returns an instance of your quantize op. During loading, Transformers calls `param_needs_quantization(model, param_name)` for each parameter and attaches the op when it returns `True`. The op runs after the model's conversion ops, so it receives the final parameter, such as the merged `experts.gate_up_proj` tensor.

```py
def get_quantize_ops(self):
    from ..integrations.your_method import YourQuantize

    return YourQuantize(self)
```

`convert` receives a dict of full parameter names and their tensors. A value can be a one-element list. Return every tensor the module needs, including scales. The example below is simplified from `Fp8Quantize`.

```py
class YourQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, **kwargs):
        result = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            quantized, scale_inv = your_quantize_fn(tensor)
            result[key] = quantized
            result[key + "_scale_inv"] = scale_inv
        return result
```

Transformers loads tensors one at a time during on-the-fly quantization. Otherwise, worker threads would move full-precision tensors to the GPU faster than the main thread quantizes them.

### Load pre-quantized weights

`get_load_operations` returns your op classes, not instances, under the `"deserialize"` and `"dequantize"` keys. Both keys are optional. [`~quantizers.HfQuantizer`] creates each op with the quantizer as its only argument.

```py
def get_load_operations(self):
    from ..integrations.your_method import YourDeserialize

    return {"deserialize": YourDeserialize}
```

The deserialize op receives the saved tensors keyed by their names from `serialized_names`, and `full_layer_name` is the parameter they load into. It returns the rebuilt parameter. The torchao quantizer uses this to rebuild a tensor subclass from `_weight_qdata`, `_weight_scale`, and the other saved parts.

```py
class YourDeserialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, full_layer_name=None, **kwargs):
        tensors = {k: v[0] if isinstance(v, list) else v for k, v in input_dict.items()}
        weight = rebuild_from_parts(tensors["_weight_qdata"], tensors["_weight_scale"])
        return {full_layer_name: weight}
```

Keep the quantizer on the op as `self.hf_quantizer`. The loader checks for this attribute (in `get_new_dtype`) to decide which tensors your op handles. It casts floating-point tensors that no op claims, such as scales, to the dtype of the parameter they load into.

### Dequantize to full precision

When the config sets `dequantize=True`, the quantizer uses the `"dequantize"` op instead of the `"deserialize"` op. The op returns full-precision weights, and the loaded model has no quantization config. [`CompressedTensorsConfig`], [`FineGrainedFP8Config`], and [`Mxfp4Config`] support this. To support it, add a `dequantize` argument to your config.

The dequantize op also runs before the model's own conversion ops. Those ops, like `MergeModulelist` and `Concatenate`, merge per-expert tensors into one parameter. After a merge, the per-expert scales no longer line up with the weights, so the op dequantizes each expert first. It receives the weights and their saved companions and returns only the weights. `Fp8Dequantize` is a reference implementation.

### Load MoE and fused weights

The model's conversion mapping in [conversion_mapping.py](https://github.com/huggingface/transformers/blob/main/src/transformers/conversion_mapping.py) matches full-precision names. For example, Mixtral merges `experts.*.w1.weight` and `experts.*.w3.weight` into `experts.gate_up_proj`. `update_weight_conversions` extends each converter whose source names end in a key of `serialized_names`.

- If the model has a parameter for a companion of the target, like `experts.gate_up_proj_scale_inv`, a copy of the converter with the same ops merges the saved companions into it.
- Otherwise, if the quantizer has a dequantize op, the converter also collects the companions and runs the dequantize op first.

Override `update_weight_conversions` only to handle a checkpoint-specific quirk, and call `super()`. Some FP8 checkpoints save scales under `.scale` instead of `.weight_scale_inv`, so the FP8 quantizer adds a rename in front.

```py
def update_weight_conversions(self, weight_conversions, model):
    from ..core_model_loading import WeightRenaming

    scale_rename = WeightRenaming(source_patterns=r"^(.+)\.scale", target_patterns=r"\1.weight_scale_inv")
    return [scale_rename] + super().update_weight_conversions(weight_conversions, model)
```

### Match saved names

A source pattern that ends in a letter or digit only matches up to the end of a name. `weight` matches `q_proj.weight` and doesn't match `q_proj.weight_scale_inv`, so you don't need to anchor patterns with `$`. A pattern that ends in `_`, like `hc_`, matches the start of a name such as `hc_fn`.
