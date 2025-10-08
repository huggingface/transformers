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
*This model was released on 2022-04-18 and added to Hugging Face Transformers on 2022-05-24 and contributed by [nielsr](https://huggingface.co/nielsr).*

# LayoutLMv3

[LayoutLMv3](https://huggingface.co/papers/2204.08387) simplifies LayoutLMv2 by using patch embeddings instead of a CNN backbone. It pre-trains on three objectives: masked language modeling, masked image modeling, and word-patch alignment. This unified approach facilitates multimodal representation learning and enhances performance across both text-centric and image-centric Document AI tasks, achieving state-of-the-art results in form understanding, receipt understanding, document visual question answering, document image classification, and document layout analysis.

<hfoptions id="usage">
<hfoption id="LayoutLMv3ForQuestionAnswering">

```py
import torch
from transformers import AutoProcessor, LayoutLMv3ForQuestionAnswering
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base", dtype="auto")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
question = "what's his name?"
words = example["tokens"]
boxes = example["bboxes"]

encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])

outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)
start_scores = outputs.start_logits
end_scores = outputs.end_logits

tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
for i, (token, start_score, end_score) in enumerate(zip(tokens, start_scores[0], end_scores[0])):
    print(f"Token {i}: '{token}' - Start: {start_score:.4f}, End: {end_score:.4f}")

predicted_start_idx = start_scores.argmax(-1).item()
predicted_end_idx = end_scores.argmax(-1).item()
predicted_answer = processor.tokenizer.decode(encoding.input_ids[0][predicted_start_idx:predicted_end_idx + 1])
print(f"\nPredicted answer: '{predicted_answer}' (tokens {predicted_start_idx}-{predicted_end_idx})")
```

</hfoption>
</hfoptions>

## LayoutLMv3Config

[[autodoc]] LayoutLMv3Config

## LayoutLMv3FeatureExtractor

[[autodoc]] LayoutLMv3FeatureExtractor
    - __call__

## LayoutLMv3ImageProcessor

[[autodoc]] LayoutLMv3ImageProcessor
    - preprocess

## LayoutLMv3ImageProcessorFast

[[autodoc]] LayoutLMv3ImageProcessorFast
    - preprocess

## LayoutLMv3Tokenizer

[[autodoc]] LayoutLMv3Tokenizer
    - __call__
    - save_vocabulary

## LayoutLMv3TokenizerFast

[[autodoc]] LayoutLMv3TokenizerFast
    - __call__

## LayoutLMv3Processor

[[autodoc]] LayoutLMv3Processor
    - __call__

## LayoutLMv3Model

[[autodoc]] LayoutLMv3Model
    - forward

## LayoutLMv3ForSequenceClassification

[[autodoc]] LayoutLMv3ForSequenceClassification
    - forward

## LayoutLMv3ForTokenClassification

[[autodoc]] LayoutLMv3ForTokenClassification
    - forward

## LayoutLMv3ForQuestionAnswering

[[autodoc]] LayoutLMv3ForQuestionAnswering
    - forward

