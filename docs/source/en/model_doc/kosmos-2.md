<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# KOSMOS-2

## Overview

The KOSMOS-2 model was proposed in [Kosmos-2: Grounding Multimodal Large Language Models to the World]
(https://arxiv.org/abs/2306.14824) by Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei

KOSMOS-2 is a Transformer-based causal language model and is trained using the next-word prediction task on a web-scale
dataset of grounded image-text pairs [GRIT](https://huggingface.co/datasets/zzliang/GRIT). The spatial coordinates of the
bounding boxes are converted to a sequence of location tokens, which are appended to their respective text spans. The
data format is similar to “hyperlinks” that connect the object regions in an image to their text span in the
corresponding caption.

The abstract from the paper is the following:

*We introduce Kosmos-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent refer expressions as links in Markdown, i.e., ``[text span](bounding boxes)'', where object descriptions are sequences of location tokens. Together with multimodal corpora, we construct large-scale data of grounded image-text pairs (called GrIT) to train the model. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), Kosmos-2 integrates the grounding capability into downstream applications. We evaluate Kosmos-2 on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension, and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This work lays out the foundation for the development of Embodiment AI and sheds light on the big convergence of language, multimodal perception, action, and world modeling, which is a key step toward artificial general intelligence. Code and pretrained models are available at https://aka.ms/kosmos-2.*

This model was contributed by [Yih-Dar SHIEH](https://huggingface.co/ydshieh). The original code can be found [here](https://github.com/microsoft/unilm/tree/master/kosmos-2).


<Tip warning={true}>

Kosmos-2 modeling code in Transformers is for finetuning and inferencing the pre-trained Kosmos-2 models.

</Tip>


## Kosmos2Config

[[autodoc]] Kosmos2Config

## Kosmos2ImageProcessor

[[autodoc]] Kosmos2ImageProcessor
    - preprocess

## Kosmos2Processor

[[autodoc]] Kosmos2Processor
    - __call__

## Kosmos2Tokenizer

[[autodoc]] Kosmos2Tokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## Kosmos2TokenizerFast

[[autodoc]] Kosmos2TokenizerFast
    - build_inputs_with_special_tokens

## Kosmos2Model

[[autodoc]] Kosmos2Model
    - forward

## Kosmos2ForConditionalGeneration

[[autodoc]] Kosmos2ForConditionalGeneration
    - forward
