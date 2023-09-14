<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# BROS

## Overview

The BROS model was proposed in [BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents](https://arxiv.org/abs/2108.04539) by Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park.

BROS stands for *BERT Relying On Spatiality*. It is an encoder-only Transformer model that takes a sequence of tokens and their bounding boxes as inputs and outputs a sequence of hidden states. BROS encode relative spatial information instead of using absolute spatial information.

It is pre-trained with two objectives: a token-masked language modeling objective (TMLM) used in BERT, and a novel area-masked language modeling objective (AMLM)
In TMLM, tokens are randomly masked, and the model predicts the masked tokens using spatial information and other unmasked tokens.
AMLM is a 2D version of TMLM. It randomly masks text tokens and predicts with the same information as TMLM, but it masks text blocks (areas).

`BrosForTokenClassification` has a simple linear layer on top of BrosModel. It predicts the label of each token.
`BrosSpadeEEForTokenClassification` has an `initial_token_classifier` and `subsequent_token_classifier` on top of BrosModel. `initial_token_classifier` is used to predict the first token of each entity, and `subsequent_token_classifier` is used to predict the next token of within entity. `BrosSpadeELForTokenClassification` has an `entity_linker` on top of BrosModel. `entity_linker` is used to predict the relation between two entities.

`BrosForTokenClassification` and `BrosSpadeEEForTokenClassification` essentially perform the same job. However, `BrosForTokenClassification` assumes input tokens are perfectly serialized (which is very challenging task since they exist in a 2D space), while `BrosSpadeEEForTokenClassification` allows for more flexibility in handling serialization errors as it predicts next connection tokens from one token.

`BrosSpadeELForTokenClassification` perform the intra-entity linking task. It predicts relation from one token (of one entity) to another token (of another entity) if these two entities share some relation.

BROS achieves comparable or better result on Key Information Extraction (KIE) benchmarks such as FUNSD, SROIE, CORD and SciTSR, without relying on explicit visual features.


The abstract from the paper is the following:

*Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space. Many recent studies try to solve the task by developing pre-trained language models focusing on combining visual features from document images with texts and their layout. On the other hand, this paper tackles the problem by going back to the basic: effective combination of text and layout. Specifically, we propose a pre-trained language model, named BROS (BERT Relying On Spatiality), that encodes relative positions of texts in 2D space and learns from unlabeled documents with area-masking strategy. With this optimized training scheme for understanding texts in 2D space, BROS shows comparable or better performance compared to previous methods on four KIE benchmarks (FUNSD, SROIE*, CORD, and SciTSR) without relying on visual features. This paper also reveals two real-world challenges in KIE tasks-(1) minimizing the error from incorrect text ordering and (2) efficient learning from fewer downstream examples-and demonstrates the superiority of BROS over previous methods.*

Tips:

- [`~transformers.BrosModel.forward`] requires `input_ids` and `bbox` (bounding box). Each bounding box should be in (x0, y0, x1, y1) format (top-left corner, bottom-right corner). Obtaining of Bounding boxes depends on external OCR system. The `x` coordinate should be normalized by document image width, and the `y` coordinate should be normalized by document image height.

```python
def expand_and_normalize_bbox(bboxes, doc_width, doc_height):
    # here, bboxes are numpy array

    # Normalize bbox -> 0 ~ 1
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / height
```

- [`~transformers.BrosForTokenClassification.forward`, `~transformers.BrosSpadeEEForTokenClassification.forward`, `~transformers.BrosSpadeEEForTokenClassification.forward`] require not only `input_ids` and `bbox` but also `box_first_token_mask` for loss calculation. It is a mask to filter out non-first tokens of each box. You can obtain this mask by saving start token indices of bounding boxes when creating `input_ids` from words. You can make `box_first_token_mask` with following code,


```python
def make_box_first_token_mask(bboxes, words, tokenizer, max_seq_length=512):

    box_first_token_mask = np.zeros(max_seq_length, dtype=np.bool_)

    # encode(tokenize) each word from words (List[str])
    input_ids_list: List[List[int]] = [tokenizer.encode(e, add_special_tokens=False) for e in words]

    # get the length of each box
    tokens_length_list: List[int] = [len(l) for l in input_ids_list]

    box_end_token_indices = np.array(list(itertools.accumulate(tokens_length_list)))
    box_start_token_indices = box_end_token_indices - np.array(tokens_length_list)

    # filter out the indices that are out of max_seq_length
    box_end_token_indices = box_end_token_indices[box_end_token_indices < max_seq_length - 1]
    if len(box_start_token_indices) > len(box_end_token_indices):
        box_start_token_indices = box_start_token_indices[: len(box_end_token_indices)]

    # set box_start_token_indices to True
    box_first_token_mask[box_start_token_indices] = True

    return box_first_token_mask

```

- Demo scripts can be found [here](https://github.com/clovaai/bros).

This model was contributed by [jinho8345](https://huggingface.co/jinho8345). The original code can be found [here](https://github.com/clovaai/bros).

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
