
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
*This model was released on 2024-03-13 and added to Hugging Face Transformers on 2024-02-21.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Gemma

[Gemma](https://huggingface.co/papers/2403.08295) is a family of lightweight language models with pretrained and instruction-tuned variants, available in 2B and 7B parameters. The architecture is based on a transformer decoder-only design. It features Multi-Query Attention, rotary positional embeddings (RoPE), GeGLU activation functions, and RMSNorm layer normalization.

The instruction-tuned variant was fine-tuned with supervised learning on instruction-following data, followed by reinforcement learning from human feedback (RLHF) to align the model outputs with human preferences.

You can find all the original Gemma checkpoints under the [Gemma](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b) release.


> [!TIP]
> Click on the Gemma models in the right sidebar for more examples of how to apply Gemma to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`] class, and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="google/gemma-2b",
    dtype=torch.bfloat16,
    device_map="auto",
)

pipeline("LLMs generate text through a process known as", max_new_tokens=50)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)

input_text = "LLMs generate text through a process known as"
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=50, cache_implementation="static")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "LLMs generate text through a process known as" | transformers run --task text-generation --model google/gemma-2b --device 0
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```py
#!pip install bitsandbytes
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b",
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="sdpa"
)

input_text = "LLMs generate text through a process known as."
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **input_ids,
    max_new_tokens=50,
    cache_implementation="static"
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("google/gemma-2b")
visualizer("LLMs generate text through a process known as")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/gemma-attn-mask.png"/>
</div>

## Notes

- The original Gemma models support standard kv-caching used in many transformer-based language models. You can use use the default [`DynamicCache`] instance or a tuple of tensors for past key values during generation. This makes it compatible with typical autoregressive generation workflows.

   ```py
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

   tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
   model = AutoModelForCausalLM.from_pretrained(
       "google/gemma-2b",
       dtype=torch.bfloat16,
       device_map="auto",
       attn_implementation="sdpa"
   )
   input_text = "LLMs generate text through a process known as"
   input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
   past_key_values = DynamicCache(config=model.config)
   outputs = model.generate(**input_ids, max_new_tokens=50, past_key_values=past_key_values)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```

## GemmaConfig

[[autodoc]] GemmaConfig

## GemmaTokenizer

[[autodoc]] GemmaTokenizer


## GemmaTokenizerFast

[[autodoc]] GemmaTokenizerFast

## GemmaModel

[[autodoc]] GemmaModel
    - forward

## GemmaForCausalLM

[[autodoc]] GemmaForCausalLM
    - forward

## GemmaForSequenceClassification

[[autodoc]] GemmaForSequenceClassification
    - forward

## GemmaForTokenClassification

[[autodoc]] GemmaForTokenClassification
    - forward
