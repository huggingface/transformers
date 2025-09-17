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
*This model was released on 2025-08-05 and added to Hugging Face Transformers on 2025-08-05.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# GptOss

## Overview

[GptOss](https://huggingface.co/papers/2508.10925) is a mixture-of-experts (MoEs) model released by OpenAI designed for powerful reasoning, agentic tasks, and versatile developer use cases. It comprises of two model variants, namely gpt-oss-120b and gpt-oss-20b.

GptOss architecture enables sparse computation by activating only a subset of parameters for each token and has a
You can find all the original GptOss checkpoints under the [State Space Models](https://huggingface.co/openai) organization.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`] and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
# install optimized MXFP4
#pip install --upgrade transformers kernels accelerate "triton>=3.4"
from transformers import pipeline

pipe = pipeline("text-generation", model="openai/gpt-oss-20b")
messages = [
    {"role": "user", "content": "Plants create energy through a process known as"},
]
pipe(messages)
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b" device_map="auto",
    torch_dtype="auto")
messages = [
    {"role": "user", "content": "Plants create energy through a process known as"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```
</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model openai/gpt-oss-20b --device 0
```

</hfoption>
</hfoptions>

> [!TIP]
> Click on the GptOss models in the right sidebar for more examples of how to apply GptOss to different language tasks.

Quantization reduces the memory burden of large models by representing the weights in a lower precision. GptOss utilizes MXFP4 on the MoE weights to reduce the memory requirement for running inference of these models. Refer to the [Quantization](../quantization/mxfp4) guide on MXFP4 for more information.


## Notes

- To check if MXFP4 kernels are loaded, you can use the Hub CLI with the following command:

```
hf cache scan
```

- Sample output:

```
REPO ID                          REPO TYPE SIZE ON DISK
-------------------------------- --------- ------------
kernels-community/triton_kernels model           536.2K
openai/gpt-oss-20b               model            13.8G
```

This indicates the MXFP4 kernels were fetched and are available for execution.

- You can view additional info here in this blogpost as well.


## GptOssConfig

[[autodoc]] GptOssConfig

## GptOssModel

[[autodoc]] GptOssModel
    - forward

## GptOssForCausalLM

[[autodoc]] GptOssForCausalLM
    - forward

## GptOssForSequenceClassification

[[autodoc]] GptOssForSequenceClassification
    - forward

## GptOssForTokenClassification

[[autodoc]] GptOssForTokenClassification
    - forward
