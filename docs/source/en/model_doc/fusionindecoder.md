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

# FusionInDecoder

## Overview

The FusionInDecoder model was proposed in [Leveraging Passage Retrieval with Generative Models
for Open Domain Question Answering
](https://arxiv.org/pdf/2007.01282.pdf) by Facebook AI Research.

The abstract from the paper is the following:

Generative models for open domain question
answering have proven to be competitive, without resorting to external knowledge. While
promising, this approach requires to use models with billions of parameters, which are expensive to train and query. In this paper, we
investigate how much these models can benefit from retrieving text passages, potentially
containing evidence. We obtain state-of-theart results on the Natural Questions and TriviaQA open benchmarks. Interestingly, we observe that the performance of this method significantly improves when increasing the number of retrieved passages. This is evidence that
sequence-to-sequence models offers a flexible
framework to efficiently aggregate and combine evidence from multiple passages.

Tips:

This model is designed to address the issue of excessive computational demand in self-attention when concatenating multiple passages.
Except for bundling multiple passages into ```batch_size * passage ```and unbundling them, it operates identically to T5.

While the input doesn't necessarily have to be three-dimensional, it's typically expected to have the shape ```(batch_size, n_passages, passage_length)``` by default.

```python
question = "question"
contexts = ["context-1", "context-2", "context-3", "context-4"]
prefix = "\n"

inputs = [question + prefix + context for context in contexts]

input_ids = tokenizer(inputs, return_tensors="pt").input_ids
```


This model was contributed by [ohgnues](https://huggingface.co/ohgnues).
The original code can be found [here](https://github.com/facebookresearch/FiD).


## T5Config

[[autodoc]] T5Config

## FusionInDecoderModel

[[autodoc]] FusionInDecoderModel
    - forward

## FusionInDecoderForConditionalGeneration

[[autodoc]] FusionInDecoderForConditionalGeneration
    - forward

## FusionInDecoderEncoderModel

[[autodoc]] FusionInDecoderEncoderModel
    - forward

## FusionInDecoderForSequenceClassification

[[autodoc]] FusionInDecoderForSequenceClassification
    - forward

## FusionInDecoderForTokenClassification

[[autodoc]] FusionInDecoderForTokenClassification
    - forward

## FusionInDecoderForQuestionAnswering

[[autodoc]] FusionInDecoderForQuestionAnswering
    - forward

</pt>
<tf>
