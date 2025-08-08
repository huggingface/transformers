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

[![Open In Colab: Torchao Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/quantization/torchao.ipynb)

[torchao](https://github.com/pytorch/ao) is a PyTorch architecture optimization library with support for custom high performance data types, quantization, and sparsity. It is composable with native PyTorch features such as [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for even faster inference and training.

See the table below for additional torchao features.

| Feature | Description |
|--------|-------------|
| **Quantization Aware Training (QAT)** | Train quantized models with minimal accuracy loss (see [QAT README](https://github.com/pytorch/ao/blob/main/torchao/quantization/qat/README.md)) |
| **Float8 Training** | High-throughput training with float8 formats (see [torchtitan](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md) and [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/low_precision_training#configuring-torchao) docs) |
| **Sparsity Support** | Semi-structured (2:4) sparsity for faster inference (see [Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity](https://pytorch.org/blog/accelerating-neural-network-training/) blog post) |
| **Optimizer Quantization** | Reduce optimizer state memory with 4 and 8-bit variants of Adam |
| **KV Cache Quantization** | Enables long context inference with lower memory (see [KV Cache Quantization](https://github.com/pytorch/ao/blob/main/torchao/_models/llama/README.md)) |
| **Custom Kernels Support** | use your own `torch.compile` compatible ops |
| **FSDP2** | Composable with FSDP2 for training|

> [!TIP]
> Refer to the torchao [README.md](https://github.com/pytorch/ao#torchao-pytorch-architecture-optimization) for more details about the library.


torchao supports the [quantization techniques](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md) below.

- A16W8 Float8 Dynamic Quantization
- A16W8 Float8 WeightOnly Quantization
- A8W8 Int8 Dynamic Quantization
- A16W8 Int8 Weight Only Quantization
- A16W4 Int4 Weight Only Quantization
- A16W4 Int4 Weight Only Quantization + 2:4 Sparsity
- Autoquantization

torchao also supports module level configuration by specifying a dictionary from fully qualified name of module and its corresponding quantization config. This allows skip quantizing certain layers and using different quantization config for different modules.


Check the table below to see if your hardware is compatible.

| Component | Compatibility |
|----------|----------------|
| CUDA Versions | ✅ cu118, cu126, cu128 |
| XPU Versions | ✅ pytorch2.8 |
| CPU | ✅ change `device_map="cpu"` (see examples below) |



Install torchao from PyPi or the PyTorch index with the following commands.

<hfoptions id="install torchao">
<hfoption id="PyPi">

```bash
# Updating 🤗 Transformers to the latest version, as the example script below uses the new auto compilation
# Stable release from Pypi which will default to CUDA 12.6
pip install --upgrade torchao transformers
```
</hfoption>
<hfoption id="PyTorch Index">
Stable Release from the PyTorch index
    
```bash
pip install torchao --index-url https://download.pytorch.org/whl/cu126 # options are cpu/cu118/cu126/cu128
```
</hfoption>
</hfoptions>

If your torchao version is below 0.10.0, you need to upgrade it, please refer to the [deprecation notice](#deprecation-notice) for more details.

## Quantization examples

TorchAO provides a variety of quantization configurations. Each configuration can be further customized with parameters such as `group_size`, `scheme`, and `layout` to optimize for specific hardware and model architectures.

For a complete list of available configurations, see the [quantization API documentation](https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py).

You can manually choose the quantization types and settings or automatically select the quantization types.

Create a [`TorchAoConfig`] and specify the quantization type and `group_size` of the weights to quantize (for int8 weight only and int4 weight only). Set the `cache_implementation` to `"static"` to automatically [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) the forward method.

We'll show examples for recommended quantization methods based on hardwares, e.g. A100 GPU, H100 GPU, CPU.

### H100 GPU
<hfoptions id="examples-H100-GPU">
<hfoption id="float8-dynamic-and-weight-only">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig

quant_config = Float8DynamicActivationFloat8WeightConfig()
# or float8 weight only quantization
# quant_config = Float8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
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
<hfoption id="int4-weight-only">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import GemliteUIntXWeightOnlyConfig

# We integrated with gemlite, which optimizes for batch size N on A100 and H100
quant_config = GemliteUIntXWeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
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
</hfoptions>

</hfoption>
<hfoption id="int4-weight-only-24sparse">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig
from torchao.dtypes import MarlinSparseLayout

quant_config = Int4WeightOnlyConfig(layout=MarlinSparseLayout())
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model with sparsity. A sparse checkpoint is needed to accelerate without accuraccy loss
quantized_model = AutoModelForCausalLM.from_pretrained(
    "RedHatAI/Sparse-Llama-3.1-8B-2of4",
    dtype=torch.float16,
    device_map="cuda",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("RedHatAI/Sparse-Llama-3.1-8B-2of4")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>
</hfoptions>

### A100 GPU
<hfoptions id="examples-A100-GPU">
<hfoption id="int8-dynamic-and-weight-only">
    
```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig

quant_config = Int8DynamicActivationInt8WeightConfig()
# or int8 weight only quantization
# quant_config = Int8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
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

<hfoption id="int4-weight-only">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import GemliteUIntXWeightOnlyConfig, Int4WeightOnlyConfig

# For batch size N, we recommend gemlite, which may require autotuning
# default is 4 bit, 8 bit is also supported by passing `bit_width=8`
quant_config = GemliteUIntXWeightOnlyConfig(group_size=128)

# For batch size 1, we also have custom tinygemm kernel that's only optimized for this
# We can set `use_hqq` to `True` for better accuracy
# quant_config = Int4WeightOnlyConfig(group_size=128, use_hqq=True)

quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
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
</hfoptions>

</hfoption>
<hfoption id="int4-weight-only-24sparse">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig
from torchao.dtypes import MarlinSparseLayout

quant_config = Int4WeightOnlyConfig(layout=MarlinSparseLayout())
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model with sparsity. A sparse checkpoint is needed to accelerate without accuraccy loss
quantized_model = AutoModelForCausalLM.from_pretrained(
    "RedHatAI/Sparse-Llama-3.1-8B-2of4",
    dtype=torch.float16,
    device_map="cuda",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("RedHatAI/Sparse-Llama-3.1-8B-2of4")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>
</hfoptions>

### Intel XPU
<hfoptions id="examples-Intel-XPU">
<hfoption id="int8-dynamic-and-weight-only">
    
```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig

quant_config = Int8DynamicActivationInt8WeightConfig()
# or int8 weight only quantization
# quant_config = Int8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("xpu")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>

<hfoption id="int4-weight-only">

```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig
from torchao.dtypes import Int4XPULayout
from torchao.quantization.quant_primitives import ZeroPointDomain


quant_config = Int4WeightOnlyConfig(group_size=128, layout=Int4XPULayout(), zero_point_domain=ZeroPointDomain.INT)
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("xpu")

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>
</hfoptions>


### CPU
<hfoptions id="examples-CPU">
<hfoption id="int8-dynamic-and-weight-only">
    
```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig

quant_config = Int8DynamicActivationInt8WeightConfig()
# quant_config = Int8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
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
<hfoption id="int4-weight-only">

> [!TIP]
> Run the quantized model on a CPU by changing `device_map` to `"cpu"` and `layout` to `Int4CPULayout()`.

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
    dtype="auto",
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
</hfoptions>

### Per Module Quantization
#### 1. Skip quantization for certain layers
With `ModuleFqnToConfig` we can specify a default configuration for all layers while skipping quantization for certain layers.
```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"

from torchao.quantization import Int4WeightOnlyConfig, ModuleFqnToConfig
config = Int4WeightOnlyConfig(group_size=128)

# set default to int4 (for linears), and skip quantizing `model.layers.0.self_attn.q_proj`
quant_config = ModuleFqnToConfig({"_default": config, "model.layers.0.self_attn.q_proj": None})
quantization_config = TorchAoConfig(quant_type=quant_config)
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16, quantization_config=quantization_config)
# lm_head is not quantized and model.layers.0.self_attn.q_proj is not quantized
print("quantized model:", quantized_model)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Manual Testing
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to(quantized_model.device.type)
generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
output_text = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

#### 2. Quantizing different layers with different quantization configs
```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

model_id = "facebook/opt-125m"

from torchao.quantization import Int4WeightOnlyConfig, ModuleFqnToConfig, Int8DynamicActivationInt4WeightConfig, IntxWeightOnlyConfig, PerAxis, MappingType

weight_dtype = torch.int8
granularity = PerAxis(0)
mapping_type = MappingType.ASYMMETRIC
embedding_config = IntxWeightOnlyConfig(
    weight_dtype=weight_dtype,
    granularity=granularity,
    mapping_type=mapping_type,
)
linear_config = Int8DynamicActivationInt4WeightConfig(group_size=128)
quant_config = ModuleFqnToConfig({"_default": linear_config, "model.decoder.embed_tokens": embedding_config, "model.decoder.embed_positions": None})
# set `include_embedding` to True in order to include embedding in quantization
# when `include_embedding` is True, we'll remove input embedding from `modules_not_to_convert` as well
quantization_config = TorchAoConfig(quant_type=quant_config, include_embedding=True)
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", dtype=torch.bfloat16, quantization_config=quantization_config)
print("quantized model:", quantized_model)
# make sure embedding is quantized
print("embed_tokens weight:", quantized_model.model.decoder.embed_tokens.weight)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Manual Testing
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
generated_ids = quantized_model.generate(**inputs, max_new_tokens=128, cache_implementation="static")
output_text = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### Autoquant

If you want to automatically choose a quantization type for quantizable layers (`nn.Linear`) you can use the [autoquant](https://pytorch.org/ao/stable/generated/torchao.quantization.autoquant.html#torchao.quantization.autoquant) API.

The `autoquant` API automatically chooses a quantization type by micro-benchmarking on input type and shape and compiling a single linear layer.

Note: autoquant is for GPU only right now.

Create a [`TorchAoConfig`] and set to `"autoquant"`. Set the `cache_implementation` to `"static"` to automatically [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) the forward method. Finally, call `finalize_autoquant` on the quantized model to finalize the quantization and log the input shapes.


```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

quantization_config = TorchAoConfig("autoquant", min_sqnr=None)
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to(quantized_model.device.type)

# auto-compile the quantized model with `cache_implementation="static"` to get speed up
output = quantized_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static")
# explicitly call `finalize_autoquant` (may be refactored and removed in the future)
quantized_model.finalize_autoquant()
print(tokenizer.decode(output[0], skip_special_tokens=True))
```


## Serialization

torchao implements [torch.Tensor subclasses](https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor) for maximum flexibility in supporting new quantized torch.Tensor formats. [Safetensors](https://huggingface.co/docs/safetensors/en/index) serialization and deserialization does not work with torchao.

To avoid arbitrary user code execution, torchao sets `weights_only=True` in [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html) to ensure only tensors are loaded. Any known user functions can be whitelisted with [add_safe_globals](https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals).

<hfoptions id="serialization-examples">
<hfoption id="save-locally">
    
```py
# don't serialize model with Safetensors
output_dir = "llama3-8b-int4wo-128"
quantized_model.save_pretrained("llama3-8b-int4wo-128", safe_serialization=False)
```
</hfoption>
<hfoption id="push-to-huggingface-hub">
    
```py
# don't serialize model with Safetensors
USER_ID = "your_huggingface_user_id"
REPO_ID = "llama3-8b-int4wo-128"
quantized_model.push_to_hub(f"{USER_ID}/llama3-8b-int4wo-128", safe_serialization=False)
tokenizer.push_to_hub(f"{USER_ID}/llama3-8b-int4wo-128")
```
</hfoption>
</hfoptions>


## Loading quantized models

Loading a quantized model depends on the quantization scheme. For quantization schemes, like int8 and float8, you can quantize the model on any device and also load it on any device. The example below demonstrates quantizing a model on the CPU and then loading it on CUDA or XPU.
```py
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

quant_config = Int8WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_config)

# Load and quantize the model
quantized_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    device_map="cpu",
    quantization_config=quantization_config
)
# save the quantized model
output_dir = "llama-3.1-8b-torchao-int8"
quantized_model.save_pretrained(output_dir, safe_serialization=False)

# reload the quantized model
reloaded_model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to(reloaded_model.device.type)

output = reloaded_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))

```
For int4, the model can only be loaded on the same device it was quantized on because the layout is specific to the device. The example below demonstrates quantizing and loading a model on the CPU.

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
    dtype="auto",
    device_map="cpu",
    quantization_config=quantization_config
)
# save the quantized model
output_dir = "llama-3.1-8b-torchao-int4-cpu"
quantized_model.save_pretrained(output_dir, safe_serialization=False)

# reload the quantized model
reloaded_model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    device_map="cpu",
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt")

output = reloaded_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))

```

## ⚠️ Deprecation Notice

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
> [Migration Guide](#migration-guide)
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



## Resources

For a better sense of expected performance, view the [benchmarks](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks) for various models with CUDA and XPU backends. You can also run the code below to benchmark a model yourself.

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

bf16_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.bfloat16)
output = bf16_model.generate(**input_ids, max_new_tokens=10, cache_implementation="static") # auto-compile
print("bf16 model:", benchmark_fn(bf16_model.generate, **input_ids, max_new_tokens=MAX_NEW_TOKENS, cache_implementation="static"))
```

> [!TIP]
> For best performance, you can use recommended settings by calling `torchao.quantization.utils.recommended_inductor_config_setter()`

Refer to [Other Available Quantization Techniques](https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques) for more examples and documentation.

## Issues

If you encounter any issues with the Transformers integration, please open an issue on the [Transformers](https://github.com/huggingface/transformers/issues) repository. For issues directly related to torchao, please open an issue on the [torchao](https://github.com/pytorch/ao/issues) repository.
