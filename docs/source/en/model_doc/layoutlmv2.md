<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-12-29 and added to Hugging Face Transformers on 2021-08-30.*

# LayoutLMV2

[LayoutLMv2](https://huggingface.co/papers/2012.14740) enhances LayoutLM by pre-training text, layout, and image in a multi-modal framework, incorporating masked visual-language modeling, text-image alignment, and text-image matching tasks. It also integrates a spatial-aware self-attention mechanism into the Transformer architecture to better understand relative positional relationships among text blocks. This results in state-of-the-art performance on various document image understanding tasks, including information extraction from scanned documents, document image classification, and document visual question answering. Improvements are shown across datasets such as FUNSD, CORD, SROIE, Kleister-NDA, RVL-CDIP, and DocVQA.

<hfoptions id="usage">
<hfoption id="LayoutLMv2ForQuestionAnswering">

```py
import torch
from transformers import AutoProcessor, LayoutLMv2ForQuestionAnswering
from PIL import Image
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased", dtype="auto")

dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
image = dataset["test"][0]["image"]
question = "When is coffee break?"
encoding = processor(image, question, return_tensors="pt")

outputs = model(**encoding)
predicted_start_idx = outputs.start_logits.argmax(-1).item()
predicted_end_idx = outputs.end_logits.argmax(-1).item()
predicted_start_idx, predicted_end_idx

predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
print(processor.tokenizer.decode(predicted_answer_tokens))
```

</hfoption>
</hfoptions>

## LayoutLMv2Config

[[autodoc]] LayoutLMv2Config

## LayoutLMv2FeatureExtractor

[[autodoc]] LayoutLMv2FeatureExtractor
    - __call__

## LayoutLMv2ImageProcessor

[[autodoc]] LayoutLMv2ImageProcessor
    - preprocess

## LayoutLMv2ImageProcessorFast

[[autodoc]] LayoutLMv2ImageProcessorFast
    - preprocess

## LayoutLMv2Tokenizer

[[autodoc]] LayoutLMv2Tokenizer
    - __call__
    - save_vocabulary

## LayoutLMv2TokenizerFast

[[autodoc]] LayoutLMv2TokenizerFast
    - __call__

## LayoutLMv2Processor

[[autodoc]] LayoutLMv2Processor
    - __call__

## LayoutLMv2Model

[[autodoc]] LayoutLMv2Model
    - forward

## LayoutLMv2ForSequenceClassification

[[autodoc]] LayoutLMv2ForSequenceClassification

## LayoutLMv2ForTokenClassification

[[autodoc]] LayoutLMv2ForTokenClassification

## LayoutLMv2ForQuestionAnswering

[[autodoc]] LayoutLMv2ForQuestionAnswering

