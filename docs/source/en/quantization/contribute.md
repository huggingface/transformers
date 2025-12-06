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

# Contribute

Transformers supports many quantization methods such as QLoRA, GPTQ, LLM.int8, and AWQ. However, there are still many more quantization approaches that haven't been integrated yet. To make adding and using these quantization methods with Transformers easier, use the [`~quantizers.HfQuantizer`] class.  [`~quantizers.HfQuantizer`] is designed to be an internal helper class for adding a quantization method instead of something applied to every PyTorch module.

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

1. Create a new quantization config class inside [src/transformers/utils/quantization_config.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/utils/quantization_config.py). Add the new quantization config to the [_import_structure](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py#L1088) inside Transformers' [src/transformers/__init__.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py) file.

2. Create a new file inside [src/transformers/quantizers/](https://github.com/huggingface/transformers/tree/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers) named `quantizer_your_method.py`, and make it inherit from [`~quantizers.HfQuantizer]. Make sure to add the new quantizer and quantization config in the quantization auto-mapping in [src/transformers/quantizers/auto.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/auto.py).

3. Define the following class attributes and property methods for your quantization method:

    - `requires_calibration`: Whether the quantization method requires a data calibration process. If set to `True`, you can only support inference (with quantized weights) and not inference and quantization.
    - `is_serializable`: A property method to determine whether the method is serializable or not.
    - `is_trainable`:  A property method to determine whether you can fine-tune models on top of the quantization method (with or without PEFT approaches).

4. Write the `validate_environment` and `update_dtype` methods. These methods are called before creating the quantized model to ensure users use the right configuration. Refer to other quantizers for an example of it is implemented.

5. Write the `_process_model_before_weight_loading` method. In Transformers, the quantized models are initialized first on the `"meta"` device before loading the weights. This means the `_process_model_before_weight_loading` method takes care of manipulating the model skeleton to replace some modules ([nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)) with the target modules (quantization modules).

You can define module replacement logic or any other utility method by creating a new file in [transformers/src/integrations/](https://github.com/huggingface/transformers/tree/abbffc4525566a48a9733639797c812301218b83/src/transformers/integrations) and exposing the relevant methods in that folder's `__init__.py` file. 

6. Add the `get_quantize_ops` method to the quantizer class if the quantization supports quantizing on the fly. In transformers, we materialize each tensor and apply a sequence of different operations on it. In our case, the quantization operation happens at the end. You need to create a `XXXQuantize`,  a subclass of `ConversionOps`, and add a `convert` method. In the `convert` method, you need to quantize the weights and return a dictionary of quantized params.

7. Add the `get_weight_conversions` method to the quantizer class if the quantization supports loading pre-quantized weights. In transformers, we can collect multiple tensors and apply operations on them. This is particularly useful when we have tensors in the checkpoint that require to be regrouped to re-create the quantized tensors.

8. Write the `_process_model_after_weight_loading` method if needed. This method enables implementing additional features that require manipulating the model after loading the weights.

9. Document everything! Make sure your quantization method is documented by adding a new file under `docs/source/en/quantization`.

10. You should add tests by adding the package in our nightly Dockerfile inside `docker/transformers-quantization-latest-gpu` and then adding a new test file in `tests/quantization/xxx`. Feel free to check out existing quantization methods to see how it is implemented.
