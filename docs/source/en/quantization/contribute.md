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

This guide will show you how to integrate a new quantization method with [`~quantizers.HfQuantizer`]. If your method is a new low-precision dtype (e.g. a vLLM or ModelOpt format), start with [Add a quantized format](./add_format): it may only need a config or a module subclass.

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

1. Create a new quantization config class inside [src/transformers/utils/quantization_config.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/utils/quantization_config.py). Add the new quantization config to the [\_import_structure](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py#L1088) inside Transformers' [src/transformers/\_\_init\_\_.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py) file.

2. Create a new file inside [src/transformers/quantizers/](https://github.com/huggingface/transformers/tree/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers) named `quantizer_your_method.py`, and make it inherit from [`~quantizers.HfQuantizer]. Make sure to add the new quantizer and quantization config in the quantization auto-mapping in [src/transformers/quantizers/auto.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/auto.py).

3. Define the following class attributes and property methods for your quantization method:

   - `requires_calibration`: Whether the quantization method requires a data calibration process. If set to `True`, you can only support inference (with quantized weights) and not inference and quantization.
   - `is_serializable`: A property method to determine whether the method is serializable or not.
   - `is_trainable`: A property method to determine whether you can fine-tune models on top of the quantization method (with or without PEFT approaches).

4. Write the `validate_environment` and `update_dtype` methods. These methods are called before creating the quantized model to ensure users use the right configuration. Refer to other quantizers for an example of it is implemented.

5. Write the `quantized_module(name, module)` method. In Transformers, the quantized models are initialized first on the `"meta"` device before loading the weights. `quantized_module` returns the quantized module replacing `module` (e.g. an [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)), or `None` to keep it. It is called on meta, only for the modules your quantizer owns (see [Several methods in one model](./overview#several-methods-in-one-model)) and that `modules_to_not_convert` doesn't skip, so don't walk the model yourself. Only quantizers implementing `quantized_module` can be mixed with other methods in a quantization plan.

Define the quantized modules and load operations in a new file in [transformers/src/integrations/](https://github.com/huggingface/transformers/tree/main/src/transformers/integrations).

6. Add the `get_quantize_ops` method to the quantizer class if the quantization supports quantizing on the fly. In transformers, we materialize each tensor and apply a sequence of different operations on it. In our case, the quantization operation happens at the end. You need to create a `XXXQuantize`, a subclass of `ConversionOps`, and add a `convert` method. In the `convert` method, you need to quantize the weights and return a dictionary of quantized params.

7. If the quantization supports loading pre-quantized weights, set `serialized_names` (what each param is stored as) and `get_load_operations` (the `"deserialize"` and `"dequantize"` ops). Add `renamed_names` if the checkpoint names a tensor differently than your module. Loading builds the converters from them, see [Load pre-quantized checkpoints](#load-pre-quantized-checkpoints).

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

## Understanding `get_quantize_ops` vs `get_load_operations`

These two methods handle different scenarios for loading weights. Understanding when to use each is essential. The quantization plan (`AutoHfQuantizer`) gives each param the ops of its module's quantizer while loading, so the same ops run when your method quantizes only part of a model, or mixes with others.

### `get_quantize_ops` — Quantize on the fly

Use this when loading a **non-quantized checkpoint** (e.g., float16/bfloat16 weights) and quantizing during load.

```text
Checkpoint: model.safetensors (float16 weights for example)
     ↓
get_quantize_ops → YourQuantize.convert()
     ↓
Result: Quantized weights in memory
```

The `convert` method receives one tensor at a time, quantizes it, and can return a dictionary of quantized params, for example:

```py
class YourQuantize(ConversionOps):
    def convert(self, input_dict, model, full_layer_name, missing_keys, **kwargs):
        # input_dict = {"layer.weight": <float16 tensor>}
        value = list(input_dict.values())[0]
        module, tensor_name = get_module_from_name(model, full_layer_name)

        # Quantize and assign
        quantized, scale, zero_point = your_quantize_fn(value)
        return {full_layer_name: quantized, full_layer_name + ".scale": scale, full_layer_name + ".zero_point": zero_point}
```

### Load pre-quantized checkpoints

Use this when loading a **pre-quantized checkpoint**, where each param is saved as several tensors (such as data, scale, and zero point). Declare them in `serialized_names` and the ops turning them into your module's tensors in `get_load_operations`. You don't write the `WeightConverter`s: loading collects a param's serialized names together, also through the model's own converters (a scale goes through the same expert merge as its weight), and gives each param the op of its module's quantizer. See [Add a quantized format](./add_format) for how a scale follows its weight through the experts merge.

```text
Checkpoint: model.safetensors (quantized components)
  - layer.weight_qdata
  - layer.weight_scale
  - layer.weight_zero_point
     ↓
serialized_names collect them → "deserialize" op: YourDeserialize.convert()
     ↓
Result: Reconstructed quantized tensor → layer.weight
```

- `"deserialize"` runs after the model's converters (e.g. the experts merge), on the pre-quantized tensors of one param. Leave it out if your module stores the tensors exactly as the checkpoint does.
- `"dequantize"` runs before the model's converters, on the weight and its serialized names, and returns the full-precision weight.
- `get_load_operation()` picks the one for the config: `"dequantize"` when it sets `dequantize=True`, else `"deserialize"`. `update_weight_transform` attaches it to each param once the param's first key matches a converter.
- `renamed_names` maps globs on a checkpoint key's last names to yours, first match wins. Keys are renamed per layer (`rename_layer`) before anything else, so `serialized_names` only uses your names.

```py
class YourHfQuantizer(HfQuantizer):
    serialized_names = {
        "weight": ["weight_qdata", "weight_scale", "weight_zero_point"],
        # fused experts: the model's converters merge each expert's tensors into these
        "gate_up_proj": ["gate_up_proj_qdata", "gate_up_proj_scale", "gate_up_proj_zero_point"],
    }
    renamed_names = {"scale": "weight_scale", "zp": "weight_zero_point"}  # the checkpoint's `.scale` / `.zp`

    def get_load_operations(self):
        return {"deserialize": YourDeserialize, "dequantize": YourDequantize}


class YourDeserialize(ConversionOps):
    def convert(self, input_dict, model, full_layer_name, **kwargs):
        # input_dict contains all collected tensors
        # Reconstruct the quantized tensor from components
        reconstructed_tensor = reconstruct_from_components(input_dict)
        return {full_layer_name: reconstructed_tensor}
```
