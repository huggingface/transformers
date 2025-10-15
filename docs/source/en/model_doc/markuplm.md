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
*This model was released on 2021-10-16 and added to Hugging Face Transformers on 2022-09-30 and contributed by [nielsr](https://huggingface.co/nielsr).*

# MarkupLM

[MarkupLM](https://huggingface.co/papers/2110.08518) addresses Visually-rich Document Understanding (VrDU) for digital documents with dynamic layouts, such as HTML/XML-based web pages. By jointly pre-training on text and markup information, MarkupLM enhances document understanding tasks. It achieves state-of-the-art results on WebSRC and SWDE benchmarks, demonstrating superior performance compared to existing models.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, AutoModelForQuestionAnswering

processor = AutoProcessor.from_pretrained("microsoft/markuplm-base-finetuned-websrc")
model = AutoModelForQuestionAnswering.from_pretrained("microsoft/markuplm-base-finetuned-websrc", dtype="auto")

html_string = "<html> <head> <title>My name is Niels</title> </head> </html>"
question = "What's his name?"

encoding = processor(html_string, questions=question, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encoding)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
processor.decode(predict_answer_tokens).strip()
```

</hfoption>
</hfoptions>

## Usage tips

- In addition to `input_ids`, [`~MarkupLMModel.forward`] expects 2 additional inputs: `xpath_tags_seq` and `xpath_subs_seq`. These are the XPATH tags and subscripts respectively for each token in the input sequence.
- Use [`MarkupLMProcessor`] to prepare all data for the model. Refer to the usage guide for more information.

## MarkupLMConfig

[[autodoc]] MarkupLMConfig
    - all

## MarkupLMFeatureExtractor

[[autodoc]] MarkupLMFeatureExtractor
    - __call__

## MarkupLMTokenizer

[[autodoc]] MarkupLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## MarkupLMTokenizerFast

[[autodoc]] MarkupLMTokenizerFast
    - all

## MarkupLMProcessor

[[autodoc]] MarkupLMProcessor
    - __call__

## MarkupLMModel

[[autodoc]] MarkupLMModel
    - forward

## MarkupLMForSequenceClassification

[[autodoc]] MarkupLMForSequenceClassification
    - forward

## MarkupLMForTokenClassification

[[autodoc]] MarkupLMForTokenClassification
    - forward

## MarkupLMForQuestionAnswering

[[autodoc]] MarkupLMForQuestionAnswering
    - forward

