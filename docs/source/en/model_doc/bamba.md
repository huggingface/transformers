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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Bamba

[Bamba](https://huggingface.co/blog/bamba) is a 9B parameter decoder-only language model built on the [Mamba-2](./mamba2) architecture. It is pretrained in two stages - it starts by training on 2T tokens from the [Dolma v1.7](https://huggingface.co/datasets/allenai/dolma) dataset and then trained on an additional 200B tokens from [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia).

You can find all the original Bamba checkpoints under the [Bamba](https://huggingface.co/collections/ibm-ai-platform/bamba-674f1388b9bbc98b413c7bab) collection.

> [!TIP]
> This model was contributed by [ani300](https://github.com/ani300) and [fabianlim](https://github.com/fabianlim).
>
> Click on the Bamba models in the right sidebar for more examples of how to apply Bamba to different text generation tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="ibm-ai-platform/Bamba-9B-v2",
    dtype=torch.bfloat16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>

<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ibm-ai-platform/Bamba-9B-v2")
model = AutoModelForCausalLM.from_pretrained("ibm-ai-platform/Bamba-9B-v2", dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>

<hfoption id="transformers CLI">
```bash
echo "Plants create energy through a process known as" | transformers-cli run --task text-generation --model ibm-ai-platform/Bamba-9B-v2 --device 0
```
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
tokenizer = AutoTokenizer.from_pretrained("ibm-ai-platform/Bamba-9B-v2")
model = AutoModelForCausalLM.from_pretrained(
   "ibm-ai-platform/Bamba-9B-v2",
   quantization_config=quantization_config,
   device_map="auto",
   attn_implementation="sdpa"
)

inputs = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")
output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- Bamba supports padding-free training which concatenates distinct training examples while still processing inputs as separate batches. It can significantly accelerate inference by [~2x](https://github.com/huggingface/transformers/pull/35861#issue-2807873129) (depending on model and data distribution) and reduce memory-usage if there are examples of varying lengths by avoiding unnecessary compute and memory overhead from padding tokens.

  Padding-free training requires the `flash-attn`, `mamba-ssm`, and `causal-conv1d` packages and the following arguments must be passed to the model in addition to `input_ids` and `labels`.

  - `position_ids: torch.LongTensor`: the position index of each token in each sequence.
  - `seq_idx: torch.IntTensor`: the index of each sequence in the batch.
  - Each of the [`FlashAttentionKwargs`]
    - `cu_seq_lens_q: torch.LongTensor`: the cumulative sequence lengths of all queries.
    - `cu_seq_lens_k: torch.LongTensor`: the cumulative sequence lengths of all keys.
    - `max_length_q: int`: the longest query length in the batch.
    - `max_length_k: int`: the longest key length in the batch.

  The `attention_mask` inputs should not be provided. The [`DataCollatorWithFlattening`] programmatically generates the set of additional arguments above using `return_seq_idx=True` and `return_flash_attn_kwargs=True`. See the [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-with-FA2) blog post for additional information.

  ```python
  from transformers import DataCollatorWithFlattening

  # Example of using padding-free training
  data_collator = DataCollatorWithFlattening(
      tokenizer=tokenizer,
      return_seq_idx=True,
      return_flash_attn_kwargs=True
  )
  ```

## BambaConfig

[[autodoc]] BambaConfig

## BambaModel

[[autodoc]] BambaModel
    - forward

## BambaForCausalLM

[[autodoc]] BambaForCausalLM
    - forward
