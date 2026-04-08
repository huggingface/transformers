<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-05-22 and added to Hugging Face Transformers on 2020-11-16.*

# RAG

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
  </div>
</div>

[Retrieval-Augmented Generation (RAG)](https://huggingface.co/papers/2005.11401) combines a pretrained language model (parametric memory) with access to an external data source (non-parametric memory) by means of a pretrained neural retriever. RAG fetches relevant passages and conditions its generation on them during inference. This often makes the answers more factual and lets you update knowledge by changing the index instead of retraining the whole model.

You can find all the original RAG checkpoints under the [AI at Meta](https://huggingface.co/facebook/models?search=rag) organization.

> [!TIP]
> This model was contributed by [ola13](https://huggingface.co/ola13).
>
> Click on the RAG models in the right sidebar for more examples of how to apply RAG to different language tasks.

The examples below demonstrates how to generate text with [`AutoModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-token-nq",
    retriever=retriever,
    dtype="auto",
    attn_implementation="flash_attention_2",
)
input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", return_tensors="pt")
generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

Quantization reduces memory by storing weights in lower precision. See the [Quantization](../quantization/overview) overview for supported backends.
The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 4-bits.

```py
import torch
from transformers import BitsAndBytesConfig, RagTokenizer, RagRetriever, RagSequenceForGeneration

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-token-nq",
    retriever=retriever,
    quantization_config=bnb,   # quantizes generator weights
    device_map="auto",
)
input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", return_tensors="pt")
generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
```

## RagConfig

[[autodoc]] RagConfig

## RagTokenizer

[[autodoc]] RagTokenizer

## Rag specific outputs

[[autodoc]] models.rag.modeling_rag.RetrievAugLMMarginOutput

[[autodoc]] models.rag.modeling_rag.RetrievAugLMOutput

## RagRetriever

[[autodoc]] RagRetriever

## RagModel

[[autodoc]] RagModel
    - forward

## RagSequenceForGeneration

[[autodoc]] RagSequenceForGeneration
    - forward
    - generate

## RagTokenForGeneration

[[autodoc]] RagTokenForGeneration
    - forward
    - generate
