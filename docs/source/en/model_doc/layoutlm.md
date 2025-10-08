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
*This model was released on 2019-12-31 and added to Hugging Face Transformers on 2020-11-16.*

# LayoutLM

[LayoutLM](https://huggingface.co/papers/1912.13318) is a pre-trained model designed to jointly learn text, layout, and visual information from scanned document images. Unlike traditional NLP models that focus solely on text, LayoutLM integrates spatial layout data (like word positions) and visual features extracted from document images to better understand structured documents. This multimodal approach allows the model to capture both semantic and formatting cues essential for document understanding tasks. It achieves state-of-the-art performance on benchmarks such as form understanding, receipt parsing, and document classification, significantly improving accuracy across all evaluated tasks.

<hfoptions id="usage">
<hfoption id="LayoutLMForQuestionAnswering">

```py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LayoutLMForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)
model = LayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", dtype="auto")

dataset = load_dataset("nielsr/funsd", split="train")
example = dataset[0]
question = "what's his name?"
words = example["words"]
boxes = example["bboxes"]

encoding = tokenizer(
    question.split(),
    words,
    is_split_into_words=True,
    return_token_type_ids=True,
    return_tensors="pt"
)
bbox = []
for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
    if s == 1:
        bbox.append(boxes[w])
    elif i == tokenizer.sep_token_id:
        bbox.append([1000] * 4)
    else:
        bbox.append([0] * 4)
encoding["bbox"] = torch.tensor([bbox])

word_ids = encoding.word_ids(0)
outputs = model(**encoding)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
start, end = word_ids[start_scores.argmax(-1)], word_ids[end_scores.argmax(-1)]
print(" ".join(words[start : end + 1]))
```

</hfoption>
</hfoptions>

## LayoutLMConfig

[[autodoc]] LayoutLMConfig

## LayoutLMTokenizer

[[autodoc]] LayoutLMTokenizer

## LayoutLMTokenizerFast

[[autodoc]] LayoutLMTokenizerFast

## LayoutLMModel

[[autodoc]] LayoutLMModel

## LayoutLMForMaskedLM

[[autodoc]] LayoutLMForMaskedLM

## LayoutLMForSequenceClassification

[[autodoc]] LayoutLMForSequenceClassification

## LayoutLMForTokenClassification

[[autodoc]] LayoutLMForTokenClassification

## LayoutLMForQuestionAnswering

[[autodoc]] LayoutLMForQuestionAnswering

