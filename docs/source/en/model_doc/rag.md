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
*This model was released on 2020-05-22 and added to Hugging Face Transformers on 2020-11-16 and contributed by [ola13](https://huggingface.co/ola13).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# RAG

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://huggingface.co/papers/2005.11401) explores a fine-tuning recipe for RAG models, integrating pre-trained seq2seq models with a dense vector index of Wikipedia accessed via a neural retriever. The study compares two RAG formulations, demonstrating superior performance on open domain QA tasks and generating more specific, diverse, and factual language in language generation tasks compared to parametric-only seq2seq baselines.

<hfoptions id="usage">
<hfoption id="RagSequenceForGeneration">

```py
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
)

model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever, dtype="auto",)
input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", return_tensors="pt")
generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

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

