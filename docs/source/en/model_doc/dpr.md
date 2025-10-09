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
*This model was released on 2020-04-10 and added to Hugging Face Transformers on 2020-11-16 and contributed by [lhoestq](https://huggingface.co/lhoestq).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DPR

[DPR](https://huggingface.co/papers/2004.04906) demonstrates that dense representations can be effectively used for open-domain question answering retrieval, surpassing traditional sparse models like TF-IDF or BM25. Using a dual-encoder framework, embeddings are learned from a limited set of questions and passages. This dense retriever achieves a significant improvement of 9%-19% in top-20 passage retrieval accuracy across various datasets, enhancing the overall performance of end-to-end QA systems.

<hfoptions id="usage">
<hfoption id="DPRReader">

```py
import torch
from transformers import DPRReader, DPRReaderTokenizer

tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base", dtype="auto")
encoded_inputs = tokenizer(
    questions=["What is photosynthesis?"],
    titles=["Plant Biology"],
    texts=["Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll in their leaves."],
    return_tensors="pt",
)
outputs = model(**encoded_inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
relevance_logits = outputs.relevance_logits
```

</hfoption>
</hfoptions>


## DPRConfig

[[autodoc]] DPRConfig

## DPRContextEncoderTokenizer

[[autodoc]] DPRContextEncoderTokenizer

## DPRContextEncoderTokenizerFast

[[autodoc]] DPRContextEncoderTokenizerFast

## DPRQuestionEncoderTokenizer

[[autodoc]] DPRQuestionEncoderTokenizer

## DPRQuestionEncoderTokenizerFast

[[autodoc]] DPRQuestionEncoderTokenizerFast

## DPRReaderTokenizer

[[autodoc]] DPRReaderTokenizer

## DPRReaderTokenizerFast

[[autodoc]] DPRReaderTokenizerFast

## DPR specific outputs

[[autodoc]] models.dpr.modeling_dpr.DPRContextEncoderOutput

[[autodoc]] models.dpr.modeling_dpr.DPRQuestionEncoderOutput

[[autodoc]] models.dpr.modeling_dpr.DPRReaderOutput

## DPRContextEncoder

[[autodoc]] DPRContextEncoder
    - forward

## DPRQuestionEncoder

[[autodoc]] DPRQuestionEncoder
    - forward

## DPRReader

[[autodoc]] DPRReader
    - forward

