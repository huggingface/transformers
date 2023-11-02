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

# Quantize 🤗 Transformers models

## `AutoGPTQ` Integration

🤗 Transformers has integrated `optimum` API to perform GPTQ quantization on language models. You can load and quantize your model in 8, 4, 3 or even 2 bits without a big drop of performance and faster inference speed! This is supported by most GPU hardwares.

To learn more about the quantization model, check out: 
- the [GPTQ](https://arxiv.org/pdf/2210.17323.pdf) paper
- the `optimum` [guide](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization) on GPTQ quantization
- the [`AutoGPTQ`](https://github.com/PanQiWei/AutoGPTQ) library used as the backend

### Requirements

You need to have the following requirements installed to run the code below: 

- Install latest `AutoGPTQ` library
`pip install auto-gptq`

- Install latest `optimum` from source 
`pip install git+https://github.com/huggingface/optimum.git`

- Install latest `transformers` from source 
`pip install git+https://github.com/huggingface/transformers.git`

- Install latest `accelerate` library 
`pip install --upgrade accelerate`

Note that GPTQ integration supports for now only text models and you may encounter unexpected behaviour for vision, speech or multi-modal models.

### Load and quantize a model

GPTQ is a quantization method that requires weights calibration before using the quantized models. If you want to quantize transformers model from scratch, it might take some time before producing the quantized model (~5 min on a Google colab for `facebook/opt-350m` model). 

Hence, there are two different scenarios where you want to use GPTQ-quantized models. The first use case would be to load models that has been already quantized by other users that are available on the Hub, the second use case would be to quantize your model from scratch and save it or push it on the Hub so that other users can also use it.
#### GPTQ Configuration

In order to load and quantize a model, you need to create a [`GPTQConfig`]. You need to pass the number of `bits`, a `dataset` in order to calibrate the quantization and the `tokenizer` of the model in order prepare the dataset.

```python 
model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
```

Note that you can pass your own dataset as a list of string. However, it is highly recommended to use the dataset from the GPTQ paper. 
```python
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
quantization = GPTQConfig(bits=4, dataset = dataset, tokenizer=tokenizer)
```

#### Quantization

You can quantize a model by using `from_pretrained` and setting the `quantization_config`. 

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=gptq_config)
```
Note that you will need a GPU to quantize a model. We will put the model in the cpu and move the modules back and forth to the gpu in order to quantize them.

If you want to maximize your gpus usage while using cpu offload, you can set `device_map = "auto"`.
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)
```
Note that disk offload is not supported. Furthermore, if you are out of memory because of the dataset, you may have to pass `max_memory` in `from_pretained`. Checkout this [guide](https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map) to learn more about `device_map` and `max_memory`.

<Tip warning={true}>
GPTQ quantization only works for text model for now. Futhermore, the quantization process can a lot of time depending on one's hardware (175B model = 4 gpu hours using NVIDIA A100). Please check on the hub if there is not a GPTQ quantized version of the model. If not, you can submit a demand on github. 
</Tip>

### Push quantized model to 🤗 Hub

You can push the quantized model like any 🤗 model to Hub with `push_to_hub`. The quantization config will be saved and pushed along the model. 

```python
quantized_model.push_to_hub("opt-125m-gptq")
tokenizer.push_to_hub("opt-125m-gptq")
```

If you want to save your quantized model on your local machine, you can also do it with `save_pretrained`: 
```python
quantized_model.save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")
```

Note that if you have quantized your model with a `device_map`, make sure to move the entire model to one of your gpus or the `cpu` before saving it. 
```python
quantized_model.to("cpu")
quantized_model.save_pretrained("opt-125m-gptq")
```

### Load a quantized model from the 🤗 Hub

You can load a quantized model from the Hub by using `from_pretrained`.
Make sure that the pushed weights are quantized, by checking that the attribute `quantization_config` is present in the model configuration object.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq")
```

If you want to load a model faster and without allocating more memory than needed, the `device_map` argument also works with quantized model. Make sure that you have `accelerate` library installed.
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto")
```

### Exllama kernels for faster inference

For 4-bit model, you can use the exllama kernels in order to a faster inference speed. It is activated by default. You can change that behavior by passing `disable_exllama` in [`GPTQConfig`]. This will overwrite the quantization config stored in the config. Note that you will only be able to overwrite the attributes related to the kernels. Furthermore, you need to have the entire model on gpus if you want to use exllama kernels. 

```py
import torch
gptq_config = GPTQConfig(bits=4, disable_exllama=False)
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config = gptq_config)
```

Note that only 4-bit models are supported for now. Furthermore, it is recommended to deactivate the exllama kernels if you are finetuning a quantized model with peft. 

#### Fine-tune a quantized model 

With the official support of adapters in the Hugging Face ecosystem, you can fine-tune models that have been quantized with GPTQ. 
Please have a look at [`peft`](https://github.com/huggingface/peft) library for more details.

### Example demo

Check out the Google Colab [notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) to learn how to quantize your model with GPTQ and how finetune the quantized model with peft. 

### GPTQConfig

[[autodoc]] GPTQConfig


## `bitsandbytes` Integration

🤗 Transformers is closely integrated with most used modules on `bitsandbytes`. You can load your model in 8-bit precision with few lines of code.
This is supported by most of the GPU hardwares since the `0.37.0` release of `bitsandbytes`.

Learn more about the quantization method in the [LLM.int8()](https://arxiv.org/abs/2208.07339) paper, or the [blogpost](https://huggingface.co/blog/hf-bitsandbytes-integration) about the collaboration.

Since its `0.39.0` release, you can load any model that supports `device_map` using 4-bit quantization, leveraging FP4 data type.

If you want to quantize your own pytorch model, check out this [documentation](https://huggingface.co/docs/accelerate/main/en/usage_guides/quantization) from 🤗 Accelerate library. 

Here are the things you can do using `bitsandbytes` integration

### General usage

You can quantize a model by using the `load_in_8bit` or `load_in_4bit` argument when calling the [`~PreTrainedModel.from_pretrained`] method as long as your model supports loading with 🤗 Accelerate and contains `torch.nn.Linear` layers. This should work for any modality as well.

```python
from transformers import AutoModelForCausalLM

model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_8bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_4bit=True)
```

By default all other modules (e.g. `torch.nn.LayerNorm`) will be converted in `torch.float16`, but if you want to change their `dtype` you can overwrite the `torch_dtype` argument:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM

>>> model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_8bit=True, torch_dtype=torch.float32)
>>> model_8bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
torch.float32
```


### FP4 quantization 

#### Requirements

Make sure that you have installed the requirements below before running any of the code snippets below.

- Latest `bitsandbytes` library
`pip install bitsandbytes>=0.39.0`

- Install latest `accelerate`
`pip install --upgrade accelerate`

- Install latest `transformers`
`pip install --upgrade transformers`

#### Tips and best practices

- **Advanced usage:** Refer to [this Google Colab notebook](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf) for advanced usage of 4-bit quantization with all the possible options.

- **Faster inference with `batch_size=1` :** Since the `0.40.0` release of bitsandbytes, for `batch_size=1` you can benefit from fast inference. Check out [these release notes](https://github.com/TimDettmers/bitsandbytes/releases/tag/0.40.0) and make sure to have a version that is greater than `0.40.0` to benefit from this feature out of the box. 

- **Training:** According to [QLoRA paper](https://arxiv.org/abs/2305.14314), for training 4-bit base models (e.g. using LoRA adapters) one should use `bnb_4bit_quant_type='nf4'`. 

- **Inference:** For inference, `bnb_4bit_quant_type` does not have a huge impact on the performance. However for consistency with the model's weights, make sure you use the same `bnb_4bit_compute_dtype` and `torch_dtype` arguments.

#### Load a large model in 4bit

By using `load_in_4bit=True` when calling the `.from_pretrained` method, you can divide your memory use by 4 (roughly).

```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
```

<Tip warning={true}>

Note that once a model has been loaded in 4-bit it is currently not possible to push the quantized weights on the Hub. Note also that you cannot train 4-bit weights as this is not supported yet. However you can use 4-bit models to train extra parameters, this will be covered in the next section.

</Tip>

### Load a large model in 8bit

You can load a model by roughly halving the memory requirements by using `load_in_8bit=True` argument when calling `.from_pretrained` method


```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)
```

Then, use your model as you would usually use a [`PreTrainedModel`].

You can check the memory footprint of your model with `get_memory_footprint` method.

```python
print(model.get_memory_footprint())
```

With this integration we were able to load large models on smaller devices and run them without any issue.  

<Tip warning={true}>

Note that once a model has been loaded in 8-bit it is currently not possible to push the quantized weights on the Hub except if you use the latest `transformers` and `bitsandbytes`. Note also that you cannot train 8-bit weights as this is not supported yet. However you can use 8-bit models to train extra parameters, this will be covered in the next section.
Note also that `device_map` is optional but setting `device_map = 'auto'` is prefered for inference as it will dispatch efficiently the model on the available ressources.

</Tip>

#### Advanced use cases

Here we will cover some advanced use cases you can perform with FP4 quantization 

##### Change the compute dtype

The compute dtype is used to change the dtype that will be used during computation. For example, hidden states could be in `float32` but computation can be set to bf16 for speedups. By default, the compute dtype is set to `float32`.

```python
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

##### Using NF4 (Normal Float 4) data type 

You can also use the NF4 data type, which is a new 4bit datatype adapted for weights that have been initialized using a normal distribution. For that run:

```python
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

##### Use nested quantization for more memory efficient inference

We also advise users to use the nested quantization technique. This saves more memory at no additional performance - from our empirical observations, this enables fine-tuning llama-13b model on an NVIDIA-T4 16GB with a sequence length of 1024, batch size of 1 and gradient accumulation steps of 4.

```python
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
```


### Push quantized models on the 🤗 Hub

You can push a quantized model on the Hub by naively using `push_to_hub` method. This will first push the quantization configuration file, then push the quantized model weights.
Make sure to use `bitsandbytes>0.37.2` (at this time of writing, we tested it on `bitsandbytes==0.38.0.post1`) to be able to use this feature. 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model.push_to_hub("bloom-560m-8bit")
```

<Tip warning={true}>

Pushing 8bit models on the Hub is strongely encouraged for large models. This will allow the community to benefit from the memory footprint reduction and loading for example large models on a Google Colab.

</Tip>

### Load a quantized model from the 🤗 Hub

You can load a quantized model from the Hub by using `from_pretrained` method. Make sure that the pushed weights are quantized, by checking that the attribute `quantization_config` is present in the model configuration object.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```
Note that in this case, you don't need to specify the arguments `load_in_8bit=True`, but you need to make sure that `bitsandbytes` and `accelerate` are installed.
Note also that `device_map` is optional but setting `device_map = 'auto'` is prefered for inference as it will dispatch efficiently the model on the available ressources.

### Advanced use cases

This section is intended to advanced users, that want to explore what it is possible to do beyond loading and running 8-bit models.

#### Offload between `cpu` and `gpu`

One of the advanced use case of this is being able to load a model and dispatch the weights between `CPU` and `GPU`. Note that the weights that will be dispatched on CPU **will not** be converted in 8-bit, thus kept in `float32`. This feature is intended for users that want to fit a very large model and dispatch the model between GPU and CPU.

First, load a [`BitsAndBytesConfig`] from `transformers` and set the attribute `llm_int8_enable_fp32_cpu_offload` to `True`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

Let's say you want to load `bigscience/bloom-1b7` model, and you have just enough GPU RAM to fit the entire model except the `lm_head`. Therefore write a custom device_map as follows:
```python
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}
```

And load your model as follows:
```python
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

And that's it! Enjoy your model!

#### Play with `llm_int8_threshold`

You can play with the `llm_int8_threshold` argument to change the threshold of the outliers. An "outlier" is a hidden state value that is greater than a certain threshold. 
This corresponds to the outlier threshold for outlier detection as described in `LLM.int8()` paper. Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models (small models, fine-tuning).
This argument can impact the inference speed of the model. We suggest to play with this parameter to find which one is the best for your use case.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_threshold=10,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### Skip the conversion of some modules

Some models has several modules that needs to be not converted in 8-bit to ensure stability. For example Jukebox model has several `lm_head` modules that should be skipped. Play with `llm_int8_skip_modules` 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### Fine-tune a model that has been loaded in 8-bit

With the official support of adapters in the Hugging Face ecosystem, you can fine-tune models that have been loaded in 8-bit. 
This enables fine-tuning large models such as `flan-t5-large` or `facebook/opt-6.7b` in a single google Colab. Please have a look at [`peft`](https://github.com/huggingface/peft) library for more details.

Note that you don't need to pass `device_map` when loading the model for training. It will automatically load your model on your GPU. You can also set the device map to a specific device if needed (e.g. `cuda:0`, `0`, `torch.device('cuda:0')`). Please note that `device_map=auto` should be used for inference only. 

### BitsAndBytesConfig

[[autodoc]] BitsAndBytesConfig


## Quantization with 🤗 `optimum` 

Please have a look at [Optimum documentation](https://huggingface.co/docs/optimum/index) to learn more about quantization methods that are supported by `optimum` and see if these are applicable for your use case.
