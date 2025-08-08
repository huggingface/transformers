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

# LayoutLM

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
</div>

<a id='Overview'></a>

## Overview

The LayoutLM model was proposed in the paper [LayoutLM: Pre-training of Text and Layout for Document Image
Understanding](https://huggingface.co/papers/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and
Ming Zhou. It's a simple but effective pretraining method of text and layout for document image understanding and
information extraction tasks, such as form understanding and receipt understanding. It obtains state-of-the-art results
on several downstream tasks:

- form understanding: the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset (a collection of 199 annotated
  forms comprising more than 30,000 words).
- receipt understanding: the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset (a collection of 626 receipts for
  training and 347 receipts for testing).
- document image classification: the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset (a collection of
  400,000 images belonging to one of 16 classes).

The abstract from the paper is the following:

*Pre-training techniques have been verified successfully in a variety of NLP tasks in recent years. Despite the
widespread use of pretraining models for NLP applications, they almost exclusively focus on text-level manipulation,
while neglecting layout and style information that is vital for document image understanding. In this paper, we propose
the LayoutLM to jointly model interactions between text and layout information across scanned document images, which is
beneficial for a great number of real-world document image understanding tasks such as information extraction from
scanned documents. Furthermore, we also leverage image features to incorporate words' visual information into LayoutLM.
To the best of our knowledge, this is the first time that text and layout are jointly learned in a single framework for
document-level pretraining. It achieves new state-of-the-art results in several downstream tasks, including form
understanding (from 70.72 to 79.27), receipt understanding (from 94.02 to 95.24) and document image classification
(from 93.07 to 94.42).*

## Usage tips

- In addition to *input_ids*, [`~transformers.LayoutLMModel.forward`] also expects the input `bbox`, which are
  the bounding boxes (i.e. 2D-positions) of the input tokens. These can be obtained using an external OCR engine such
  as Google's [Tesseract](https://github.com/tesseract-ocr/tesseract) (there's a [Python wrapper](https://pypi.org/project/pytesseract/) available). Each bounding box should be in (x0, y0, x1, y1) format, where
  (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the
  position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on a 0-1000
  scale. To normalize, you can use the following function:

```python
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

Here, `width` and `height` correspond to the width and height of the original document in which the token
occurs. Those can be obtained using the Python Image Library (PIL) library for example, as follows:

```python
from PIL import Image

# Document can be a png, jpg, etc. PDFs must be converted to images.
image = Image.open(name_of_your_document).convert("RGB")

width, height = image.size
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LayoutLM. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.


<PipelineTag pipeline="document-question-answering" />

- A blog post on [fine-tuning
  LayoutLM for document-understanding using Keras & Hugging Face
  Transformers](https://www.philschmid.de/fine-tuning-layoutlm-keras).

- A blog post on how to [fine-tune LayoutLM for document-understanding using only Hugging Face Transformers](https://www.philschmid.de/fine-tuning-layoutlm).

- A notebook on how to [fine-tune LayoutLM on the FUNSD dataset with image embeddings](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb).

- See also: [Document question answering task guide](../tasks/document_question_answering)

<PipelineTag pipeline="text-classification" />

- A notebook on how to [fine-tune LayoutLM for sequence classification on the RVL-CDIP dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForSequenceClassification_on_RVL_CDIP.ipynb).
- [Text classification task guide](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification" />

- A notebook on how to [ fine-tune LayoutLM for token classification on the FUNSD dataset](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb).
- [Token classification task guide](../tasks/token_classification)

**Other resources**
- [Masked language modeling task guide](../tasks/masked_language_modeling)

🚀 Deploy

- A blog post on how to [Deploy LayoutLM with Hugging Face Inference Endpoints](https://www.philschmid.de/inference-endpoints-layoutlm).

## LayoutLMConfig

[[autodoc]] LayoutLMConfig

## LayoutLMTokenizer

[[autodoc]] LayoutLMTokenizer

## LayoutLMTokenizerFast

[[autodoc]] LayoutLMTokenizerFast

<frameworkcontent>
<pt>

## LayoutLMModel

[[autodoc]] LayoutLMModel

## LayoutLMForMaskedLM

[[autodoc]] LayoutLMForMaskedLM

## LayoutLMForSequenceClassification

[[autodoc]] LayoutLMForSequenceClassification

## LayoutLMForTokenClassification

[[autodoc]] LayoutLMForTokenClassification

## LayoutLMForQuestionAnswering

[[autodoc]] LayoutLMForQuestionAnswering

</pt>
<tf>

## TFLayoutLMModel

[[autodoc]] TFLayoutLMModel

## TFLayoutLMForMaskedLM

[[autodoc]] TFLayoutLMForMaskedLM

## TFLayoutLMForSequenceClassification

[[autodoc]] TFLayoutLMForSequenceClassification

## TFLayoutLMForTokenClassification

[[autodoc]] TFLayoutLMForTokenClassification

## TFLayoutLMForQuestionAnswering

[[autodoc]] TFLayoutLMForQuestionAnswering

</tf>
</frameworkcontent>


