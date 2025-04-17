<!--Copyright 2025 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# AutoRound

[AutoRound](https://github.com/intel/auto-round) is an advanced quantization algorithm that delivers strong accuracy, even at 2-bit precision. 
It leverages sign gradient descent to fine-tune both rounding values and min-max clipping thresholds in just 200 steps. Designed for broad compatibility, it seamlessly supports a wide range of LLMs and is actively expanding to cover more VLMs as well. 
It also supports tuning and inference across multiple hardware platforms, including CPU, XPU, and CUDA.

AutoRound also offers a variety of useful features, including mixed-bit tuning and inference, lm-head quantization, support for exporting to formats like GPTQ/AWQ/GGUF, and flexible tuning recipes. 
For a comprehensive overview and the latest updates, check out the AutoRound [README](https://github.com/intel/auto-round).


AutoRound is part of the [Intel Neural Compressor](https://github.com/intel/neural-compressor). For more model compression technicals optimized for Intel devices, please check it out

## Installation

```bash
pip install auto-round
```

## Supported Quantization Configurations

AutoRound supports several quantization configurations:

- **Int8 Weight Only**
- **Int4 Weight Only**
- **Int3 Weight Only**
- **Int2 Weight Only**
- **Mixed bits Weight only**

## Hardware Compatibility

CPU, XPU, and CUDA for both quantization and inference.

## Quantization and Serialization (offline)

Currently, only offline mode is supported to generate quantized models.

<hfoptions id="quantization">
<hfoption id="quantization cmd">

### Command Line Usage
```bash
auto-round \
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --output_dir ./tmp_autoround
```

AutoRound also offer two configurations, `auto-round-best` and `auto-round-light`, designed for optimal accuracy and improved speed, respectively. 
For 2 bits, we recommend using `auto-round-best` and `auto-round`.
</hfoption>

<hfoption id="quantization api">

### API Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

bits, group_size, sym = 4, 128, True
# mixed bits config
# layer_config = {"model.decoder.layers.6.self_attn.out_proj": {"bits": 2, "group_size": 32}}

autoround = AutoRound(
    model, 
    tokenizer, 
    bits=bits, 
    group_size=group_size, 
    sym=sym,
    # enable_torch_compile=True,
    # layer_config=layer_config,
)

## the best accuracy, 3X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

## 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
## format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format='auto_round') 
```

</hfoption>

</hfoptions>

W4G128 Average Accuracy of 13 tasks (mmlu-pro,if_eval,gsm8k,etc) and Time Cost Results(Testing was conducted on the Nvidia A100 80G using the version of PyTorch 2.6.0 with enable_torch_compile):

| Model   | Qwen2.5-0.5B-Instruct | Falcon3-3B      | Qwen2.5-7B-Instruct | Meta-Llama-3.1-8B-Instruct | Falcon3-10B     | Qwen2.5-72B-Instruct |
|---------|-----------------------|-----------------|---------------------|----------------------------|-----------------|----------------------|
| 16bits  | 0.4192                | 0.5203          | 0.6470              | 0.6212                     | 0.6151          | 0.7229               |
| Best    | **0.4137**(7m)        | **0.5142**(23m) | 0.6426(58m)         | **0.6116**(81m)            | **0.6092**(81m) | 0.7242(575m)         |
| Default | 0.4129(2m)            | 0.5133(6m)      | 0.6441(13m)         | 0.6106(13m)                | 0.6080(18m)     | **0.7252**(118m)     |
| Light   | 0.4052(2m)            | 0.5108(3m)      | **0.6453**(5m)      | 0.6104(6m)                 | 0.6063(6m)      | 0.7243(37m)          |

## Inference

AutoRound automatically selects the best available backend based on the installed libraries and prompts the user to install additional libraries when a better backend is found.
<hfoptions id="inference">
<hfoption id="inference cpu">

### CPU

Supports 2,4,8 bits, we recommend to use intel-extension-for-pytorch for 4 bits 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

<hfoption>

### XPU

<hfoption id="inference xpu">
Supports 4 bits,  IPEX(intel-extension-for-pytorch) is required

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="xpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

<hfoption>

### CUDA

<hfoption id="inference cuda">
Supports 2,3 4,8 bits, we recommend to use GPTQModel for 4,8 bits

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

<hfoption>

### Specify Backend

<hfoption id="specific backend">

The automatically selected backend may not always be the most suitable for certain devices. 
You can specify your preferred backend, such as "ipex"(cpu/xpu), "itrex"(cpu), "marlin"(cuda), "exllamav2"(cuda), "triton"(cuda) and others, according to your needs or hardware compatibility. Please note that some corresponding libraries  may need to be installed.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoRoundConfig

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
quantization_config = AutoRoundConfig(backend="ipex")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

<hfoption>


### Convert GPTQ/AWQ to AutoRound

<hfoption id="format convert">
Most GPTQ/AWQ models can be converted to the AutoRound format for better compatibility and support with Intel devices. Please note that the quantization config will be changed if the model is serialized.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoRoundConfig

model_name = "ybelkada/opt-125m-gptq-4bit"
quantization_config = AutoRoundConfig()
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

<hfoption>



<hfoptions>

## Issues

If you encounter any issues with the transformers integration, please open an issue on
the [transformers](https://github.com/huggingface/transformers/issues) repository.  
If you encounter any issues with auto-round, please open an issue on
the [auto-round](https://github.com/intel/auto-round/issues) repository.


## Acknowledgement
Special thanks to AutoGPTQ, AutoAWQ, GPTQModel, Triton, Marlin, and ExLLaMAV2 for providing low-precision CUDA kernels, which were leveraged in AutoRound.