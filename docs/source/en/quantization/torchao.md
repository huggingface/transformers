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

# torchao

[torchao](https://github.com/pytorch/ao) is a PyTorch architecture optimization library with support for custom high performance data types, quantization, and sparsity. It is composable with native PyTorch features such as [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for even faster inference and training. You can checkout torchao [README](https://github.com/pytorch/ao#torchao-pytorch-architecture-optimization) for an overall introduction to the library and recent highlights and updates.

## Main Functionalities

| Feature | Description |
|--------|-------------|
| **Post-Training Quantization (PTQ)** | Quantize weights and activations  with just one line, see below for examples |
| **Quantization Aware Training (QAT)** | Train quantized models with minimal accuracy loss, for more details see [QAT README](https://github.com/pytorch/ao/blob/main/torchao/quantization/qat/README.md) |
| **Float8 Training** | High-throughput training with float8 formats, for more details see [torchtitan](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md) |
| **Sparsity Support** | Semi-structured (2:4) sparsity for faster inference, details in the [blog](https://pytorch.org/blog/accelerating-neural-network-training/) |
| **Optimizer Quantization** | Reduce optimizer state memory with 4/8-bit variants of Adam. |
| **KV Cache Quantization** | Enables long context inference with lower memory, some details [here](https://github.com/pytorch/ao/blob/main/torchao/_models/llama/README.md) |
| **Custom Kernels Support** | use your own ops compatible with `torch.compile`. |
| **FSDP2** | Composable with FSDP2 for training|

## Hardware Compatibility

| Component | Compatibility |
|----------|----------------|
| **CUDA Versions** | ✅ cu118, cu124, cu126, cu128 |
| **CPU** | ✅ Quantization on cpu is supported (different layouts for Int4, see below) |
| **CPU / GPU Offloading** | ✅ Optimizers and gradients can be offloaded |

## Supported Quantization techniques

TorchAO supports several quantization techniques:

- **Int8 Weight Only**
- **Int4 Weight Only**
- **Int8 Dynamic Quantization**
- **Float8 (E4M3 / E5M2)**
- **KV Cache Quantization**
- **Custom intx**: Support for int2, int3, etc., via pure PyTorch bitpacking (alpha feature)



# Use in transformers

## Installation

Install torchao with the following command.

Stable release from Pypi which will default to CUDA 12.4
```bash
# Updating 🤗 Transformers to the latest version, as the example script below uses the new auto compilation
pip install --upgrade torchao transformers
```
Stable Release from the PyTorch index
```bash
pip install torchao --extra-index-url https://download.pytorch.org/whl/cu124 # options are cpu/cu118/cu124/cu126
```

## Available Quantization Schemes

TorchAO provides a variety of quantization configurations:

- `Int4WeightOnlyConfig`
- `Int8WeightOnlyConfig`
- `Int8DynamicActivationInt8WeightConfig`
- `Float8WeightOnlyConfig`

Each configuration can be further customized with parameters such as `group_size`, `scheme`, and `layout` to optimize for specific hardware and model architectures.

For a complete list of available configurations, see our [quantization API documentation](https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py).

> **⚠️ DEPRECATION WARNING**
>
> Starting with version 0.10.0, the string-based API for quantization configuration (e.g., `TorchAoConfig("int4_weight_only", group_size=128)`) is **deprecated** and will be removed in a future release.
>
> Please use the new `AOBaseConfig`-based approach instead:
>
> ```python
> # Old way (deprecated)
> quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
>
> # New way (recommended)
> from torchao.quantization import Int4WeightOnlyConfig
> quant_config = Int4WeightOnlyConfig(group_size=128)
> quantization_config = TorchAoConfig(quant_type=quant_config)
> ```
>
> The new API offers greater flexibility, better type safety, and access to the full range of features available in torchao.
>
> ## Migration Guide
>
> Here's how to migrate from common string identifiers to their `AOBaseConfig` equivalents:
>
> | Old String API | New `AOBaseConfig` API |
> |----------------|------------------------|
> | `"int4_weight_only"` | `Int4WeightOnlyConfig()` |
> | `"int8_weight_only"` | `Int8WeightOnlyConfig()` |
> | `"int8_dynamic_activation_int8_weight"` | `Int8DynamicActivationInt8WeightConfig()` |
>
> All configuration objects accept parameters for customization (e.g., `group_size`, `scheme`, `layout`).

## Quantization Examples 

You can manually choose the quantization types and settings or automatically select the quantization types.

Create a [`TorchAoConfig`] and specify the quantization type and `group_size` of the weights to quantize. Set the `cache_implementation` to `"static"` to automatically [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) the forward method.

<hfoptions id="examples">
<hfoption id="int8-weight-only cuda">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

quant_config = Int8WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>

<hfoption id="int8-weight-only cpu">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

quant_config = Int8WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="cpu",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>
<hfoption id="int4-weight-only cuda">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig

quant_config = Int4WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>

<hfoption id="int4-weight-only cpu">

> [!TIP]
> Run the quantized model on a CPU by changing `device_map` to `"cpu"` and `layout` to `Int4CPULayout()`. This is only available in torchao 0.8.0+.

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig
from torchao.dtypes import Int4CPULayout

quant_config = Int4WeightOnlyConfig(group_size=128, layout=Int4CPULayout())
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="cpu",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>
<hfoption id="int8-dynamic-quantization cuda">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8DynamicActivationInt8WeightConfig

quant_config = Int8DynamicActivationInt8WeightConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
<hfoption id="int8-dynamic-quantization cpu">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8DynamicActivationInt8WeightConfig

quant_config = Int8DynamicActivationInt8WeightConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="cpu",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>
<hfoption id="float8-weight-only cuda">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Float8WeightOnlyConfig

quant_config = Float8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))

```
</hfoption>
<hfoption id="float8-weight-only cpu">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Float8WeightOnlyConfig

quant_config = Float8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="cpu",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>

<hfoption id="automatic">

The [autoquant](https://pytorch.org/ao/stable/generated/torchao.quantization.autoquant.html#torchao.quantization.autoquant) API automatically chooses a quantization type for quantizable layers (`nn.Linear`) by micro-benchmarking on input type and shape and compiling a single linear layer.

Create a [`TorchAoConfig`] and set to `"autoquant"`. Set the `cache_implementation` to `"static"` to automatically [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) the forward method. Finally, call `finalize_autoquant` on the quantized model to finalize the quantization and log the input shapes.


```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

quantization_config = TorchAoConfig("autoquant", min_sqnr=None)
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
# explicitly call `finalize_autoquant` (may be refactored and removed in the future)
quantized_model.finalize_autoquant()
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>

</hfoptions>

## Loading Quantized Models Examples

```py

```



## Benchmark

```py
from torch._inductor.utils import do_bench_using_profiling
from typing import Callable

def benchmark_fn(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3

MAX_NEW_TOKENS = 1000
print("int4wo-128 model:", benchmark_fn(quantized_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS, cache_implementation="static"))

bf16_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
output = bf16_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static") # auto-compile
print("bf16 model:", benchmark_fn(bf16_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS, cache_implementation="static"))
```

> [!TIP]
> For best performance, you can use recommended settings by calling `torchao.quantization.utils.recommended_inductor_config_setter()`



## Serialization

torchao implements [torch.Tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor) for maximum flexibility in supporting new quantized torch.Tensor formats. [Safetensors](https://huggingface.co/docs/safetensors/en/index) serialization and deserialization does not work with torchao.

To avoid arbitrary user code execution, torchao sets `weights_only=True` in [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html) to ensure only tensors are loaded. Any known user functions can be whitelisted with [add_safe_globals](https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals).

```py
# don't serialize model with Safetensors
output_dir = "llama3-8b-int4wo-128"
quantized_model.save_pretrained("llama3-8b-int4wo-128", safe_serialization=False)
```

## Resources

For a better sense of expected performance, view the [benchmarks](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks) for various models with CUDA and XPU backends.

Refer to [Other Available Quantization Techniques](https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques) for more examples and documentation.
