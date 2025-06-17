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

# Bamba

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Bamba-9B is a decoder-only language model based on the [Mamba-2](https://github.com/state-spaces/mamba) architecture and is designed to handle a wide range of text generation tasks. It is trained from scratch using a two-stage training approach. In the first stage, the model is trained on 2 trillion tokens from the Dolma v1.7 dataset. In the second stage, it undergoes additional training on 200 billion tokens, leveraging a carefully curated blend of high-quality data to further refine its performance and enhance output quality.

Checkout all Bamba-9B model checkpoints [here](https://github.com/foundation-model-stack/bamba).

## BambaConfig

| Model            | Params       | # Layers | Hidden Dim. | Attention Heads | GQA | KV Heads | Context Length |  Tied Embeddings |
|-------------------|--------------|----------|-------------|-----------------|-----|----------|----------------|------------------|
| Bamba  | 9B (9.78B)   | 32       | 4096        | 32              | Yes | 8        | 4096           | True |

[[autodoc]] BambaConfig

<!---
## Usage Tips

Tips:

- The architecture is based on Mamba-2 models.

## BambaModel

[[autodoc]] BambaModel
    - forward
-->

## BambaForCausalLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibm-fms/Bamba-9B")
tokenizer = AutoTokenizer.from_pretrained("ibm-fms/Bamba-9B")

message = ["Mamba is a snake with following properties  "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```


## Padding-Free Training

Bamba supports padding-free training in which distinct training examples can be concatenated
together while nevertheless processing the inputs as though they belonged to separate batches. When
the examples are of varying lengths, padding-free training can provide significant speed ups and
memory savings compared to batching the examples together and using padding, as the unnecessary
compute and memory due to padding is avoided entirely. The performance gains depend on factors such
as the model and the data distribution, but throughput gains up to [~2x are commonly
seen](https://github.com/huggingface/transformers/pull/35861#issue-2807873129).

Using padding-free training with Bamba requires the `flash-attn`, `mamba-ssm`, and `causal-conv1d`
packages, and the following arguments must be passed to the model in addition to `input_ids` and
`labels`:
* `position_ids: torch.LongTensor`: the position index of each token in each sequence.
* `seq_idx: torch.IntTensor`: the index of each sequence in the batch.
* Each of the [`FlashAttentionKwargs`]
    * `cu_seq_lens_q: torch.LongTensor`: The cumulative sequence lengths of all queries.
    * `cu_seq_lens_k: torch.LongTensor`: The cumulative sequence lengths of all keys.
    * `max_length_q: int`: the longest query length in the batch.
    * `max_length_k: int`: the longest key length in the batch.

The `attention_mask` inputs should not be provided. The [`DataCollatorWithFlattening`] can be used
to programmatically generate the above set of additional arguments using `return_seq_idx=True` and
`return_flash_attn_kwargs=True`. See [this blog post](https://huggingface.co/blog/packing-with-FA2)
for additional information.


[[autodoc]] BambaForCausalLM
    - forward

This HF implementation is contributed by [ani300](https://github.com/ani300) and [fabianlim](https://github.com/fabianlim).
