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

# LXMERT

## Overview

The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan & Mohit Bansal. It is a series of bidirectional transformer encoders
(one for the vision modality, one for the language modality, and then one to fuse both modalities) pretrained using a
combination of masked language modeling, visual-language text alignment, ROI-feature regression, masked
visual-attribute modeling, masked visual-object modeling, and visual-question answering objectives. The pretraining
consists of multiple multi-modal datasets: MSCOCO, Visual-Genome + Visual-Genome Question Answering, VQA 2.0, and GQA.

The abstract from the paper is the following:

*Vision-and-language reasoning requires an understanding of visual concepts, language semantics, and, most importantly,
the alignment and relationships between these two modalities. We thus propose the LXMERT (Learning Cross-Modality
Encoder Representations from Transformers) framework to learn these vision-and-language connections. In LXMERT, we
build a large-scale Transformer model that consists of three encoders: an object relationship encoder, a language
encoder, and a cross-modality encoder. Next, to endow our model with the capability of connecting vision and language
semantics, we pre-train the model with large amounts of image-and-sentence pairs, via five diverse representative
pretraining tasks: masked language modeling, masked object prediction (feature regression and label classification),
cross-modality matching, and image question answering. These tasks help in learning both intra-modality and
cross-modality relationships. After fine-tuning from our pretrained parameters, our model achieves the state-of-the-art
results on two visual question answering datasets (i.e., VQA and GQA). We also show the generalizability of our
pretrained cross-modality model by adapting it to a challenging visual-reasoning task, NLVR, and improve the previous
best result by 22% absolute (54% to 76%). Lastly, we demonstrate detailed ablation studies to prove that both our novel
model components and pretraining strategies significantly contribute to our strong results; and also present several
attention visualizations for the different encoders*

Tips:

- Bounding boxes are not necessary to be used in the visual feature embeddings, any kind of visual-spacial features
  will work.
- Both the language hidden states and the visual hidden states that LXMERT outputs are passed through the
  cross-modality layer, so they contain information from both modalities. To access a modality that only attends to
  itself, select the vision/language hidden states from the first input in the tuple.
- The bidirectional cross-modality encoder attention only returns attention values when the language modality is used
  as the input and the vision modality is used as the context vector. Further, while the cross-modality encoder
  contains self-attention for each respective modality and cross-attention, only the cross attention is returned and
  both self attention outputs are disregarded.

This model was contributed by [eltoto1219](https://huggingface.co/eltoto1219). The original code can be found [here](https://github.com/airsplay/lxmert).

## Documentation resources

- [Question answering task guide](../tasks/question_answering)

## LxmertConfig

[[autodoc]] LxmertConfig

## LxmertTokenizer

[[autodoc]] LxmertTokenizer

## LxmertTokenizerFast

[[autodoc]] LxmertTokenizerFast

## Lxmert specific outputs

[[autodoc]] models.lxmert.modeling_lxmert.LxmertModelOutput

[[autodoc]] models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput

[[autodoc]] models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput

[[autodoc]] models.lxmert.modeling_tf_lxmert.TFLxmertModelOutput

[[autodoc]] models.lxmert.modeling_tf_lxmert.TFLxmertForPreTrainingOutput

## LxmertModel

[[autodoc]] LxmertModel
    - forward

## LxmertForPreTraining

[[autodoc]] LxmertForPreTraining
    - forward

## LxmertForQuestionAnswering

[[autodoc]] LxmertForQuestionAnswering
    - forward

## TFLxmertModel

[[autodoc]] TFLxmertModel
    - call

## TFLxmertForPreTraining

[[autodoc]] TFLxmertForPreTraining
    - call
