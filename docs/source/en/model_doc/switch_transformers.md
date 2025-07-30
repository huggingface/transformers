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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Switch Transformers

[Switch Transformers](https://huggingface.co/papers/2101.03961) is a sparse T5 model where the MLP layer is replaced by a Mixture-of-Experts (MoE). A routing mechanism associates each token with an expert and each expert is a dense MLP. Sparsity enables better scaling and the routing mechanism allows the model to select relevant weights on the fly which increases model capacity.

You can find all the original Switch Transformers checkpoints under the [Switch Transformer](https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f) collection.


> [!TIP]
> This model was contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).
>
> Click on the Switch Transformers models in the right sidebar for more examples of how to apply Switch Transformers to different natural language tasks.

The example below demonstrates how to predict the masked token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text2text-generation", 
    model="google/switch-base-8",
    dtype=torch.float16,
    device=0
)
print(pipeline("The capital of France is <extra_id_0>."))
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8", device_map="auto", dtype=torch.float16)

input_text = "The capital of France is <extra_id_0>."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "The capital of France is <extra_id_0>." | transformers run --task text2text-generation --model google/switch-base-8 --device 0
# [{'generated_text': 'Paris.'}]
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes/) to only quantize the weights to 8-bits.

```py
# pip install bitsandbytes
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8", device_map="auto", quantization_config=quantization_config)

input_text = "The capital of France is <extra_id_0>."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```


## SwitchTransformersConfig

[[autodoc]] SwitchTransformersConfig

## SwitchTransformersTop1Router

[[autodoc]] SwitchTransformersTop1Router
    - _compute_router_probabilities
    - forward

## SwitchTransformersSparseMLP

[[autodoc]] SwitchTransformersSparseMLP
    - forward

## SwitchTransformersModel

[[autodoc]] SwitchTransformersModel
    - forward

## SwitchTransformersForConditionalGeneration

[[autodoc]] SwitchTransformersForConditionalGeneration
    - forward

## SwitchTransformersEncoderModel

[[autodoc]] SwitchTransformersEncoderModel
    - forward
