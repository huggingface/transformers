<!--Copyright 2024 JetMoe team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-06-07 and added to Hugging Face Transformers on 2024-05-14.*


# add appropriate badges
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
            <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
            <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
            <img alt="MoE" src="https://img.shields.io/badge/MoE-8B-blue?style=flat">
    </div>
</div>


# JetMoe

[JetMoe-8b](https://huggingface.co/papers/2404.07413) is an 8B-parameter Mixture-of-Experts (MoE) language model for efficient text generation. The model activates a subset of specialized “experts” for each input, which improves computational efficiency while keeping performance comparable to dense models of similar size. Each block consists of two MoE layers: Mixture of Attention Heads and Mixture of MLP Experts. This sparse activation allows the model to process input tokens effectively and achieve faster training and inference with fewer resources compared to standard dense models.

You can find all the original JetMoe checkpoints under the [JetMoe](https://huggingface.co/jetmoe) collection.

> [!TIP]
> This model was contributed by [Yikang Shen](https://huggingface.co/YikangS).
>
> Click on the JetMoe models in the right sidebar for more examples of how to apply JetMoe to different text generation tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline>

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="jetmoe/jetmoe-8b",
    dtype=torch.float16,
    device=0
)
pipeline("The stock market rallied today after positive economic news")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b")
model = AutoModelForCausalLM.from_pretrained(
    "jetmoe/jetmoe-8b",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
input_ids = tokenizer("The stock market rallied today after positive economic news", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers-cli">

```bash
echo -e "The stock market rallied today after positive economic news" | transformers run --task text-generation --model jetmoe/jetmoe-8b --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```py
# pip install torchao
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = AutoModelForCausalLM.from_pretrained(
    "jetmoe/jetmoe-8b",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b")
input_ids = tokenizer("The stock market rallied today after positive economic news", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

# add if this is supported for your model
Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("jetmoe/jetmoe-8b")
visualizer("The stock market rallied today after positive economic news")
```

# upload image to https://huggingface.co/datasets/huggingface/documentation-images/tree/main/transformers/model_doc and ping me to merge
<div class="flex justify-center">
    <img src=""/>
</div>

## Notes

- JetMOE uses sparse expert routing; only a subset of experts is activated per input.
- The training throughput of JetMoe-8B is around 100B tokens per day on a cluster of 96 H100 GPUs with a straightforward 3-way pipeline parallelism strategy.


## JetMoeConfig

[[autodoc]] JetMoeConfig

## JetMoeModel

[[autodoc]] JetMoeModel
    - forward

## JetMoeForCausalLM

[[autodoc]] JetMoeForCausalLM
    - forward

## JetMoeForSequenceClassification

[[autodoc]] JetMoeForSequenceClassification
    - forward
