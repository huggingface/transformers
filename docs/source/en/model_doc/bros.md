<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2021-08-10 and added to Hugging Face Transformers on 2023-09-15 and contributed by [jinho8345](https://huggingface.co/jinho8345).*

# BROS

[BROS](https://huggingface.co/papers/2108.04539) is a pre-trained language model designed for key information extraction (KIE) from document images by focusing on the spatial relationships of text rather than visual features. It encodes the relative 2D positions of text elements and uses an area-masking pre-training strategy to learn spatial-textual dependencies from unlabeled documents. Unlike vision-text models, BROS effectively integrates text and layout information alone, achieving competitive or superior results on major KIE benchmarks (FUNSD, SROIE*, CORD, SciTSR). The model also addresses two key challenges in KIE—handling incorrect text order and learning efficiently with limited labeled data.

<hfoptions id="usage">
<hfoption id="BrosForTokenClassification">

```py
import torch
from transformers import AutoProcessor, AutoModelForTokenClassification

processor = AutoProcessor.from_pretrained("jinho8345/bros-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("jinho8345/bros-base-uncased", dtype="auto")

text = "Plants create energy through a process known as photosynthesis."
encoding = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")
bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
encoding["bbox"] = bbox

outputs = model(**encoding)
predictions = torch.argmax(outputs.logits, dim=-1)
tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

print("Token predictions:")
for token, pred in zip(tokens, predictions[0]):
    print(f"'{token}' -> Class {pred.item()}")
```

</hfoption>
</hfoptions>

## Usage tips

- [`BrosModel.forward`] requires `input_ids` and `bbox` (bounding box). Each bounding box uses `(x0, y0, x1, y1)` format (top-left corner, bottom-right corner). Get bounding boxes from external OCR systems. Normalize x coordinates by document image width and y coordinates by document image height.

- [`BrosForTokenClassification.forward`], [`BrosSpadeEEForTokenClassification.forward`], and [`BrosSpadeELForTokenClassification.forward`] require `input_ids`, `bbox`, and `box_first_token_mask` for loss calculation. This mask filters out non-first tokens of each box. Save start token indices of bounding boxes when creating `input_ids` from words to obtain this mask.

## BrosConfig

[[autodoc]] BrosConfig

## BrosProcessor

[[autodoc]] BrosProcessor
    - __call__

## BrosModel

[[autodoc]] BrosModel
    - forward

## BrosForTokenClassification

[[autodoc]] BrosForTokenClassification
    - forward

## BrosSpadeEEForTokenClassification

[[autodoc]] BrosSpadeEEForTokenClassification
    - forward

## BrosSpadeELForTokenClassification

[[autodoc]] BrosSpadeELForTokenClassification
    - forward

