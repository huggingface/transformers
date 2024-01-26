<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Build a new `HfQuantizer` class to add quantization support for a new quantization method.

Through this document, you will learn how to work on a transformers integration of a new quantization method. Note that currently the `HfQuantizer` is not meant to be used for any PyTorch module, but you should rather see it as an internal utility class that is used in the core modeling code to easily quantize transformers models with different SoTA approaches (e.g. QLoRA, GPTQ, LLM.int8, AWQ, ...). 


## Pre-requisities 

Before you start integrating a new quantization method into transformers, make sure that the method you are trying to add meet the following pre-requisities. Note we only support quantization methods that can be run with PyTorch modules for now.

- The quantization method is available through a Python package that is pip-installable by anyone (it is also fine if you can only install the package from source), ideally with pre-compiled kernels included in the pip package.
- The method can at least run on a commonly-used hardware (CPU, GPU, ..).
- The method is wrapped in a `nn.Module` (e.g. `Linear8bitLt`, `Linear4bit`). Ideally your quantized linear layer should have the following definition
```py
class Linear4bit(nn.Module):
    def __init__(self, ...):
        ...
    
    def forward(self, x):
        return my_4bit_kernel(x, self.weight, self.bias)
```
That way, transformers models can be easily quantizable by simply replacing some instances of `nn.Linear` with a target class.
- Ideally the quantization method should be serializable, i.e. you can save the quantized weights locally or push them on the Hub.
- Make sure the package that contains the quantization kernels / primitive is mature enough (e.g. no frequent breaking changes).

Note that for some quantization methods it is a strong requirement to "pre-quantize" the models through data calibration (e.g. AWQ). In that case, we prefer to support only inference through transformers and let third-party libraries maintained by the ML community deal with the model quantization itself.

## How should I get started?

0- 📕 Create a new quantization config class inside `src/transformers/utils/quantization_config.py`, and make sure to expose that new quantization config inside transformers main init, by adding it on the `_import_structure` object of `src/transformers/__init__.py`.

1-  🗃 Create a new file inside `src/transformers/quantizers/` named `quantizer_your_method.py` and make it inherit from `src/transformers/quantizers/base.py::HfQuantizer`. Make sure to add the new quantizer and quantization config in the quantization auto-mapping in `src/transformers/quantizers/auto.py`

2- 🔩 Define the following class attributes / property methods:

2.1. `requires_calibration`: Whether the quantization method requires a data-calibration process. If set to `True` you'll be able to only support inference (with quantized weights) and not inference + quantization.
2.2. `required_packages`: A list of strings of the required packages to use the quantized weights. You might need to define some new utility methods such as `is_auto_awq_available` in `transformers/src/utils/import_utils.py`
2.3 `requires_parameters_quantization`: (Advanced - defaults to `False`) Only required if your quantization methods requires a special care of the underlying `nn.Parameter` object. For example bitsandbytes uses `Params4bit` and `Int8Param` that requires some special care when quantizing the model. Most of the recent quantization method packs int2 / int4 weights inside `torch.uint8` weights so that flag should not be really required
2.4 `is_serializable`: A property method to determine whether the method is serializable or not
2.5. `is_trainable`:  A property method to determine whether you can fine-tune models on top of that quantization methods (with or without PEFT approaches).


3- 🪛 Write the `validate_environment` and `set_torch_dtype` methods. These methods are called before creating the quantized model to make sure users are on the right configuration. You can have a look at how this is done on other quantizers.

4- 🖋 Write the `_process_model_before_weight_loading` method. In transformers, the quantized models are first initialized on the `"meta"` device before loading the weights. Therefore `_process_model_before_weight_loading` can take care of manipulating the model skeleton to replace some modules (e.g. nn.Linear) with target modules (quantization modules). You can define a module replacement logic or any other utility method by creating a new file in `transformers/src/integrations/` and make sure to expose the relevant methods in the `__init__.py` file of that folder. Again the best starting point would be to have a look at what is done for other quantization methods such as `quantizer_awq.py`

5- 🖊 Write the `_process_model_after_weight_loading` method: in case you want to implement additional features that requires to manipulate the model post loading the weight, you can define that whole logic there!

6- 📖 Document eveything! Make sure that your quantization method is documented in the `docs/source/en/quantization.md` file.

7- 🟢 Add tests! You should add tests by first adding the package in our nightly Dockerfile inside `docker/transformers-all-latest-gpu` then adding a new test file in `tests/quantization/xxx`. Feel free to check out what is done on other quantization methods (e.g. awq)

