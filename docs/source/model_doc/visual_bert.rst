.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

VisualBERT
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The VisualBERT model was proposed in `VisualBERT: A Simple and Performant Baseline for Vision and Language
<https://arxiv.org/pdf/1908.03557>`__ by Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang.

The model is a multi-modal (vision-and-language) pre-trainded model trained with Masked Language Modeling on textual
part, and Sentence-Image prediction task to predict whether two captions are matching for an image or not.

The abstract from the paper is the following:

*We propose VisualBERT, a simple and flexible framework for modeling a broad range of vision-and-language tasks.
VisualBERT consists of a stack of Transformer layers that implicitly align elements of an input text and regions in an
associated input image with self-attention. We further propose two visually-grounded language model objectives for
pre-training VisualBERT on image caption data. Experiments on four vision-and-language tasks including VQA, VCR, NLVR2,
and Flickr30K show that VisualBERT outperforms or rivals with state-of-the-art models while being significantly
simpler. Further analysis demonstrates that VisualBERT can ground elements of language to image regions without any
explicit supervision and is even sensitive to syntactic relationships, tracking, for example, associations between
verbs and image regions corresponding to their arguments.*

Tips:

1. All the checkpoints are named in a way to depict whether these checkpoints are the `pretrained` checkpoints. The
   visual embedding dimensions differ in each case. Here is a description of the configurations:
   - visualbert-vqa-coco-pre: autoclass:: transformers.VisualBertForPreTraining pre-trained on CoCo dataset with masked
     language modeling and sentence-image prediction tasks.
   - visualbert-vqa-pre: autoclass:: transformers.VisualBertForPreTraining pre-trained on VQA dataset with masked
     language modeling and sentence-image prediction tasks, after pre-training on CoCo dataset.
   - visualbert-vqa: autoclass:: transformers.VisualBertForQuestionAnswering fine-tuned on VQA task, after pre-training
     on CoCo and VQA dataset.
   - visualbert-nlvr2-coco-pre: autoclass:: transformers.VisualBertForPreTraining pre-trained on CoCo dataset with
     masked language modeling and sentence-image prediction tasks.
   - visualbert-nlvr2-pre: autoclass:: transformers.VisualBertForPreTraining pre-trained on NLVR2 dataset with masked
     language modeling and sentence-image prediction tasks, after pre-training on CoCo dataset.
   - visualbert-nlvr2: autoclass:: transformers.VisualBertForVisualReasoning fine-tuned on NLVR2 task, after
     pre-training on CoCo and NLVR2 dataset.
   - visualbert-vcr-coco-pre: autoclass:: transformers.VisualBertForPreTraining pre-trained on CoCo dataset with masked
     language modeling and sentence-image prediction tasks.
   - visualbert-vcr-pre: autoclass:: transformers.VisualBertForPreTraining pre-trained on VCR dataset with masked
     language modeling and sentence-image prediction tasks, after pre-training on CoCo dataset.
   - visualbert-vcr: autoclass:: transformers.VisualBertForMultipleChoice fine-tuned on VCR task, after pre-training on
     CoCo and VCR dataset.

2. Most of the checkpoints provided work with the `VisualBertForPreTraining` configuration. Other checkpoints provided
   are for down-stream tasks - VQA, VCR, NLVR2. Hence, if you are not working on these downstream tasks, you should
   create your own model and pre-train it or use one of these if it fits your needs.

3. For the VCR task, the authors use a fine-tuned ResNet detector for generating visual embeddings, for all the
   checkpoints. We do not provide the detector and its weights as a part of the package, but it will be available in
   the research projects, and the states can be loaded directly into the detector provided.
   
4. For tokenization of text, you can use any of the `BertTokenizer`s, although the authors used `bert-base-uncased`.

Note: More tips will be added, and a demo notebook on how to use a detector to generate your own visual embeddings, and
use the VisualBERT model for fine-tuning on your task.

VisualBertConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisualBertConfig
    :members:

VisualBertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisualBertModel
    :members: forward


VisualBertForPreTraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisualBertForPreTraining
    :members: forward


VisualBertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisualBertForQuestionAnswering
    :members: forward


VisualBertForQuestionAnsweringAdvanced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisualBertForQuestionAnsweringAdvanced
    :members: forward


VisualBertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisualBertForMultipleChoice
    :members: forward


VisualBertForVisualReasoning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisualBertForVisualReasoning
    :members: forward


VisualBertForRegionToPhraseAlignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisualBertForRegionToPhraseAlignment
    :members: forward
