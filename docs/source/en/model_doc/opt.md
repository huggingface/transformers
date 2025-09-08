<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-05-02 and added to Hugging Face Transformers on 2022-05-12.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
           <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
           <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# OPT

[OPT](https://huggingface.co/papers/2205.01068) is a suite of open-source decoder-only pre-trained transformers whose parameters range from 125M to 175B. OPT models are designed for causal language modeling and aim to enable responsible and reproducible research at scale. OPT-175B is comparable in performance to GPT-3 with only 1/7th the carbon footprint.

You can find all the original OPT checkpoints under the [OPT](https://huggingface.co/collections/facebook/opt-66ed00e15599f02966818844) collection.

> [!TIP]
> This model was contributed by [ArthurZ](https://huggingface.co/ArthurZ), [ybelkada](https://huggingface.co/ybelkada), and [patrickvonplaten](https://huggingface.co/patrickvonplaten).
>
> Click on the OPT models in the right sidebar for more examples of how to apply OPT to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.


<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="facebook/opt-125m", dtype=torch.float16, device=0)
pipeline("Once upon a time, in a land far, far away,", max_length=50, num_return_sequences=1)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

prompt = ("Once upon a time, in a land far, far away, ")

model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
tokenizer.batch_decode(generated_ids)[0]
```
</hfoption>
<hfoption id="transformers CLI">

```py
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model facebook/opt-125m --device 0
```
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](..quantization/bitsandbytes) to quantize the weights to 8-bits.

```py
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, infer_device

device = infer_device()

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", dtype=torch.float16, attn_implementation="sdpa", quantization_config=bnb_config).to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")

prompt = ("Once upon a time, in a land far, far away, ")

model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
tokenizer.batch_decode(generated_ids)[0]
```

## Notes

- OPT adds an `EOS` token `</s>` to the beginning of every prompt.

- The `head_mask` argument is ignored if the attention implementation isn't `"eager"`. Set `attn_implementation="eager"` to enable the `head_mask`.

## Resources

- Refer to this [notebook](https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing) for an example of fine-tuning OPT with PEFT, bitsandbytes, and Transformers.
- The [How 🤗 Accelerate runs very large models thanks to PyTorch](https://huggingface.co/blog/accelerate-large-models) blog post demonstrates how to run OPT for inference.

## OPTConfig

[[autodoc]] OPTConfig

## OPTModel

[[autodoc]] OPTModel
    - forward

## OPTForCausalLM

[[autodoc]] OPTForCausalLM
    - forward

## OPTForSequenceClassification

[[autodoc]] OPTForSequenceClassification
    - forward

## OPTForQuestionAnswering

[[autodoc]] OPTForQuestionAnswering
    - forward
