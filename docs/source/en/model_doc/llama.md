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
*This model was released on 2023-02-27 and added to Hugging Face Transformers on 2023-03-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Llama

[Llama](https://huggingface.co/papers/2302.13971) is a family of large language models ranging from 7B to 65B parameters. These models are focused on efficient inference (important for serving language models) by training a smaller model on more tokens rather than training a larger model on fewer tokens. The Llama model is based on the GPT architecture, but it uses pre-normalization to improve training stability, replaces ReLU with SwiGLU to improve performance, and replaces absolute positional embeddings with rotary positional embeddings (RoPE) to better handle longer sequence lengths.

You can find all the original Llama checkpoints under the [Huggy Llama](https://huggingface.co/huggyllama) organization.

> [!TIP]
> Click on the Llama models in the right sidebar for more examples of how to apply Llama to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="huggyllama/llama-7b",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "huggyllama/llama-7b",
)
model = AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model huggyllama/llama-7b --device 0
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
    "huggyllama/llama-30b",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("huggyllama/llama-7b")
visualizer("Plants create energy through a process known as")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llama-attn-mask.png"/>
</div>

## Notes

- The tokenizer is a byte-pair encoding model based on [SentencePiece](https://github.com/google/sentencepiece). During decoding, if the first token is the start of the word (for example, "Banana"), the tokenizer doesn't prepend the prefix space to the string.

## LlamaConfig

[[autodoc]] LlamaConfig

## LlamaTokenizer

[[autodoc]] LlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LlamaTokenizerFast

[[autodoc]] LlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary

## LlamaModel

[[autodoc]] LlamaModel
    - forward

## LlamaForCausalLM

[[autodoc]] LlamaForCausalLM
    - forward

## LlamaForSequenceClassification

[[autodoc]] LlamaForSequenceClassification
    - forward

## LlamaForQuestionAnswering

[[autodoc]] LlamaForQuestionAnswering
    - forward

## LlamaForTokenClassification

[[autodoc]] LlamaForTokenClassification
    - forward
