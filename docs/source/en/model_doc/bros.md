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

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](<https://huggingface.co/<INSERT YOUR HF USERNAME HERE>). The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

**Use case: token classification

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
