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

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-blue.svg)](https://pytorch.org/get-started/locally/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/install)
[![Flax](https://img.shields.io/badge/Flax-0.6+-yellow.svg)](https://flax.readthedocs.io/en/latest/installation.html)
[![Safetensors](https://img.shields.io/badge/Safetensors-0.3+-green.svg)](https://huggingface.co/docs/safetensors/installation)
[![Flash Attention](https://img.shields.io/badge/Flash%20Attention-2.0+-blue.svg)](https://github.com/Dao-AILab/flash-attention)
[![SDPA](https://img.shields.io/badge/SDPA-2.0+-blue.svg)](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

# LayoutLMv3

LayoutLMv3 is a powerful multimodal transformer model designed specifically for Document AI tasks. What makes it unique is its unified approach to handling both text and images in documents, using a simple yet effective architecture that combines patch embeddings with transformer layers. Unlike its predecessor LayoutLMv2, it uses a more streamlined approach with patch embeddings (similar to ViT) instead of a CNN backbone.

The model is pre-trained on three key objectives:
1. Masked Language Modeling (MLM) for text understanding
2. Masked Image Modeling (MIM) for visual understanding
3. Word-Patch Alignment (WPA) for learning cross-modal relationships

This unified architecture and training approach makes LayoutLMv3 particularly effective for both text-centric tasks (like form understanding and receipt analysis) and image-centric tasks (like document classification and layout analysis).

[Paper](https://arxiv.org/abs/2204.08387) | [Official Checkpoints](https://huggingface.co/microsoft/layoutlmv3-base)

<Tip>
Click on the right sidebar for more examples of how to use the model for different tasks!
</Tip>

## Quick Start

Here's a quick example of how to use LayoutLMv3 for document understanding:

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from PIL import Image
import torch

# Load model and processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

# Load and preprocess your document image
image = Image.open("document.jpg").convert("RGB")
encoding = processor(image, return_tensors="pt")

# Get predictions
outputs = model(**encoding)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

## Using the Pipeline

The easiest way to use LayoutLMv3 is through the pipeline API:

```python
from transformers import pipeline

# For document classification
classifier = pipeline("document-classification", model="microsoft/layoutlmv3-base")
result = classifier("document.jpg")

# For token classification (e.g., form understanding)
token_classifier = pipeline("token-classification", model="microsoft/layoutlmv3-base")
result = token_classifier("form.jpg")

# For question answering
qa = pipeline("document-question-answering", model="microsoft/layoutlmv3-base")
result = qa(question="What is the total amount?", image="receipt.jpg")
```

## Using AutoModel

You can also use the AutoModel classes for more flexibility:

```python
from transformers import AutoModelForDocumentQuestionAnswering, AutoProcessor

# Load model and processor
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
model = AutoModelForDocumentQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

# Process inputs
image = Image.open("document.jpg").convert("RGB")
encoding = processor(image, return_tensors="pt")

# Get predictions
outputs = model(**encoding)
```

## Using transformers-cli

For quick inference from the command line:

```bash
transformers-cli document-classification "document.jpg" --model microsoft/layoutlmv3-base
```

## Quantization

For large models, you can use quantization to reduce memory usage:

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import torch

# Load model with 8-bit quantization
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    load_in_8bit=True,
    device_map="auto"
)

# Or with 4-bit quantization
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    load_in_4bit=True,
    device_map="auto"
)
```

## Attention Visualization

You can visualize the attention patterns using the AttentionMaskVisualizer:

```python
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from transformers.utils.visualization import AttentionMaskVisualizer
from PIL import Image

# Load model and processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")

# Process image
image = Image.open("document.jpg").convert("RGB")
encoding = processor(image, return_tensors="pt")

# Get attention weights
outputs = model(**encoding, output_attentions=True)

# Visualize attention
visualizer = AttentionMaskVisualizer()
visualizer.visualize_attention(
    image,
    outputs.attentions[-1][0],  # Last layer attention
    processor.tokenizer
)
```

## Usage tips

- In terms of data processing, LayoutLMv3 is identical to its predecessor [LayoutLMv2](layoutlmv2), except that:
    - images need to be resized and normalized with channels in regular RGB format. LayoutLMv2 on the other hand normalizes the images internally and expects the channels in BGR format.
    - text is tokenized using byte-pair encoding (BPE), as opposed to WordPiece.
  Due to these differences in data preprocessing, one can use [`LayoutLMv3Processor`] which internally combines a [`LayoutLMv3ImageProcessor`] (for the image modality) and a [`LayoutLMv3Tokenizer`]/[`LayoutLMv3TokenizerFast`] (for the text modality) to prepare all data for the model.
- Regarding usage of [`LayoutLMv3Processor`], we refer to the [usage guide](layoutlmv2#usage-layoutlmv2processor) of its predecessor.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LayoutLMv3. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<Tip>

LayoutLMv3 is nearly identical to LayoutLMv2, so we've also included LayoutLMv2 resources you can adapt for LayoutLMv3 tasks. For these notebooks, take care to use [`LayoutLMv2Processor`] instead when preparing data for the model!

</Tip>

- Demo notebooks for LayoutLMv3 can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3).
- Demo scripts can be found [here](https://github.com/huggingface/transformers-research-projects/tree/main/layoutlmv3).

<PipelineTag pipeline="text-classification"/>

- [`LayoutLMv2ForSequenceClassification`] is supported by this [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb).
- [Text classification task guide](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- [`LayoutLMv3ForTokenClassification`] is supported by this [example script](https://github.com/huggingface/transformers-research-projects/tree/main/layoutlmv3) and [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb).
- A [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Inference_with_LayoutLMv2ForTokenClassification.ipynb) for how to perform inference with [`LayoutLMv2ForTokenClassification`] and a [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/True_inference_with_LayoutLMv2ForTokenClassification_%2B_Gradio_demo.ipynb) for how to perform inference when no labels are available with [`LayoutLMv2ForTokenClassification`].
- A [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb) for how to finetune [`LayoutLMv2ForTokenClassification`] with the 🤗 Trainer.
- [Token classification task guide](../tasks/token_classification)

<PipelineTag pipeline="question-answering"/>

- [`LayoutLMv2ForQuestionAnswering`] is supported by this [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb).
- [Question answering task guide](../tasks/question_answering)

**Document question answering**
- [Document question answering task guide](../tasks/document_question_answering)

## Model Details

### LayoutLMv3Config

[[autodoc]] LayoutLMv3Config

### LayoutLMv3FeatureExtractor

[[autodoc]] LayoutLMv3FeatureExtractor
    - __call__

### LayoutLMv3Tokenizer

[[autodoc]] LayoutLMv3Tokenizer
    - __call__

### LayoutLMv3ImageProcessor

[[autodoc]] LayoutLMv3ImageProcessor
    - preprocess

### LayoutLMv3Tokenizer

[[autodoc]] LayoutLMv3Tokenizer
    - __call__
    - save_vocabulary

### LayoutLMv3TokenizerFast

[[autodoc]] LayoutLMv3TokenizerFast
    - __call__

### LayoutLMv3Processor

[[autodoc]] LayoutLMv3Processor
    - __call__

### LayoutLMv3Model

[[autodoc]] LayoutLMv3Model

### LayoutLMv3ForSequenceClassification

[[autodoc]] LayoutLMv3ForSequenceClassification

### LayoutLMv3ForTokenClassification

[[autodoc]] LayoutLMv3ForTokenClassification

### LayoutLMv3ForQuestionAnswering

[[autodoc]] LayoutLMv3ForQuestionAnswering

### TFLayoutLMv3Model

[[autodoc]] TFLayoutLMv3Model

### TFLayoutLMv3ForSequenceClassification

[[autodoc]] TFLayoutLMv3ForSequenceClassification

### TFLayoutLMv3ForTokenClassification

[[autodoc]] TFLayoutLMv3ForTokenClassification

### TFLayoutLMv3ForQuestionAnswering

[[autodoc]] TFLayoutLMv3ForQuestionAnswering
