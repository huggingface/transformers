<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Bros

## Overview

The Bros model was proposed in [BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents](https://arxiv.org/abs/2108.04539)  by Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park. BROS is a document understanding model pre-trained with the area-masking strategy. It obtains comparable or better result on KIE benchmarks (FUNSD, SROIE, CORD and SciTSR).

The abstract from the paper is the following:

*Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space. Many recent studies try to solve the task by developing pre-trained language models focusing on combining visual features from document images with texts and their layout. On the other hand, this paper tackles the problem by going back to the basic: effective combination of text and layout. Specifically, we propose a pre-trained language model, named BROS (BERT Relying On Spatiality), that encodes relative positions of texts in 2D space and learns from unlabeled documents with area-masking strategy. With this optimized training scheme for understanding texts in 2D space, BROS shows comparable or better performance compared to previous methods on four KIE benchmarks (FUNSD, SROIE*, CORD, and SciTSR) without relying on visual features. This paper also reveals two real-world challenges in KIE tasks-(1) minimizing the error from incorrect text ordering and (2) efficient learning from fewer downstream examples-and demonstrates the superiority of BROS over previous methods.*

Tips:

- [`~transformers.BrosModel.forward`] requires *input_ids* and `bbox` (bounding box). Each bounding box should be in (x0, y0, x1, y0, x1, y1, x0, y1) format represented by four clockwise points starting from top-left corner. Obtaining of Bounding boxes depends on external OCR system. The `x` coordinate should be normalized by document image width, and the `y` coordinate should be normalized by document image height. Since most OCR systems output bounding boxes with two points (x0, y0, x1, y1), you can expand and normalize bboxes with following code,

```python
def expand_and_normalize_bbox(bboxes, doc_width, doc_height):
    # here, bboxes are numpy array

    # Expand bbox from 2 points to 4 points
    bboxes = bboxes[:, [0, 1, 2, 1, 2, 3, 0, 3]]

    # Normalize bbox -> 0 ~ 1
    bboxes[:, [0, 2, 4, 6]] = bboxes[:, [0, 2, 4, 6]] / doc_width
    bboxes[:, [1, 3, 5, 7]] = bboxes[:, [1, 3, 5, 7]] / doc_height
```

- [`~transformers.BrosForTokenClassification.forward`, `~transformers.BrosSpadeEEForTokenClassification.forward`, `~transformers.BrosSpadeEEForTokenClassification.forward`] require not only *input_ids* and *bbox* but also `box_first_token_mask` for loss calculation. It is a mask to filter out non-first tokens of each box. You can obtain this mask by saving start token indices of bounding boxes when creating *input_ids* from words. Detailed instructions for this process can be found in Demo scripts or notebooks provided below.

- Demo scripts can be found [here](https://github.com/clovaai/bros).

This model was contributed by [jinho8345](https://huggingface.co/jinho8345). The original code can be found [here](https://github.com/clovaai/bros).

## BrosConfig

[[autodoc]] BrosConfig


## BrosTokenizer

[[autodoc]] BrosTokenizer
    - __call__
    - save_vocabulary


## BrosTokenizerFast

[[autodoc]] BrosTokenizerFast
    - __call__


## BrosModel

[[autodoc]] BrosModel
    - forward


## BrosForTokenClassification

[[autodoc]] transformers.BrosForTokenClassification
    - forward


## BrosSpadeEEForTokenClassification

[[autodoc]] transformers.BrosSpadeEEForTokenClassification
    - forward


## BrosSpadeELForTokenClassification

[[autodoc]] transformers.BrosSpadeELForTokenClassification
    - forward
