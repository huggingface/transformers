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

# VisualBERT

## Overview

The VisualBERT model was proposed in [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557) by Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang.
VisualBERT is a neural network trained on a variety of (image, text) pairs.

The abstract from the paper is the following:

*We propose VisualBERT, a simple and flexible framework for modeling a broad range of vision-and-language tasks.
VisualBERT consists of a stack of Transformer layers that implicitly align elements of an input text and regions in an
associated input image with self-attention. We further propose two visually-grounded language model objectives for
pre-training VisualBERT on image caption data. Experiments on four vision-and-language tasks including VQA, VCR, NLVR2,
and Flickr30K show that VisualBERT outperforms or rivals with state-of-the-art models while being significantly
simpler. Further analysis demonstrates that VisualBERT can ground elements of language to image regions without any
explicit supervision and is even sensitive to syntactic relationships, tracking, for example, associations between
verbs and image regions corresponding to their arguments.*

This model was contributed by [gchhablani](https://huggingface.co/gchhablani). The original code can be found [here](https://github.com/uclanlp/visualbert).

## Usage tips

1. Most of the checkpoints provided work with the [`VisualBertForPreTraining`] configuration. Other
   checkpoints provided are the fine-tuned checkpoints for down-stream tasks - VQA ('visualbert-vqa'), VCR
   ('visualbert-vcr'), NLVR2 ('visualbert-nlvr2'). Hence, if you are not working on these downstream tasks, it is
   recommended that you use the pretrained checkpoints.

2. For the VCR task, the authors use a fine-tuned detector for generating visual embeddings, for all the checkpoints.
   We do not provide the detector and its weights as a part of the package, but it will be available in the research
   projects, and the states can be loaded directly into the detector provided.

VisualBERT is a multi-modal vision and language model. It can be used for visual question answering, multiple choice,
visual reasoning and region-to-phrase correspondence tasks. VisualBERT uses a BERT-like transformer to prepare
embeddings for image-text pairs. Both the text and visual features are then projected to a latent space with identical
dimension.

To feed images to the model, each image is passed through a pre-trained object detector and the regions and the
bounding boxes are extracted. The authors use the features generated after passing these regions through a pre-trained
CNN like ResNet as visual embeddings. They also add absolute position embeddings, and feed the resulting sequence of
vectors to a standard BERT model. The text input is concatenated in the front of the visual embeddings in the embedding
layer, and is expected to be bound by [CLS] and a [SEP] tokens, as in BERT. The segment IDs must also be set
appropriately for the textual and visual parts.

The [`BertTokenizer`] is used to encode the text. A custom detector/image processor must be used
to get the visual embeddings. The following example notebooks show how to use VisualBERT with Detectron-like models:

- [VisualBERT VQA demo notebook](https://github.com/huggingface/transformers/tree/main/examples/research_projects/visual_bert) : This notebook
  contains an example on VisualBERT VQA.

- [Generate Embeddings for VisualBERT (Colab Notebook)](https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing) : This notebook contains
  an example on how to generate visual embeddings.

The following example shows how to get the last hidden state using [`VisualBertModel`]:

```python
>>> import torch
>>> from transformers import BertTokenizer, VisualBertModel

>>> model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("What is the man eating?", return_tensors="pt")
>>> # this is a custom function that returns the visual embeddings given the image path
>>> visual_embeds = get_visual_embeddings(image_path)

>>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
>>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
>>> inputs.update(
...     {
...         "visual_embeds": visual_embeds,
...         "visual_token_type_ids": visual_token_type_ids,
...         "visual_attention_mask": visual_attention_mask,
...     }
... )
>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
```

## VisualBertConfig

[[autodoc]] VisualBertConfig

## VisualBertModel

[[autodoc]] VisualBertModel
    - forward

## VisualBertForPreTraining

[[autodoc]] VisualBertForPreTraining
    - forward

## VisualBertForQuestionAnswering

[[autodoc]] VisualBertForQuestionAnswering
    - forward

## VisualBertForMultipleChoice

[[autodoc]] VisualBertForMultipleChoice
    - forward

## VisualBertForVisualReasoning

[[autodoc]] VisualBertForVisualReasoning
    - forward

## VisualBertForRegionToPhraseAlignment

[[autodoc]] VisualBertForRegionToPhraseAlignment
    - forward
