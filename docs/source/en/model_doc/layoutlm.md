<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-12-31 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>



# LayoutLM

[LayoutLM](https://huggingface.co/papers/1912.13318) jointly learns text and the document layout rather than focusing only on text. It incorporates positional layout information and visual features of words from the document images.

E.g., if you're looking at a receipt, you know the price is next to the "Total" label because of its position. LayoutLM learns this exact kind of spatial relationship. It takes a standard model like [BERT](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/bert) for text embedding, but adds two more things: it feeds the model not just the text, but also 1. the location of every single word in the document (2-D position embedding via bounding boxes for each word) and 2. the whole document as an image (image embedding of scanned document images). 

It's also pre-trained using two unique approaches:

- **Masked Visual-Language Model (MVLM)**: It's like a fill-in-the-blanks test. The model sees a document with some words blacked out, and it has to guess what's missing by looking at the other words and their positions on the page.

- **Multi-label Document Classification (MDC)**: It also learns to categorize entire documents (like "this is a tax form" or "this is a resume"). This helps it get a better overall understanding of the document's structure and purpose.

So, in a nutshell, it's a model that understands both the words and the layout, making it a great tool for reading and understanding documents.

You can find all the original LayoutLM checkpoints under the [LayoutLM](https://huggingface.co/collections/microsoft/layoutlm-6564539601de72cb631d0902) collection.

> [!TIP]
> Click on the LayoutLM models in the right sidebar for more examples of how to apply LayoutLM to different vision and language tasks.

The example below demonstrates question answering with the [`AutoModel`] class. 


<hfoptions id="usage">
<hfoption id="Pipeline">

Note, that the original LayoutLM version cannot be readily used with the [`Pipeline`] class unlike later versions such as [**LayoutLMv3**](https://huggingface.co/docs/transformers/en/model_doc/layoutlmv3). 

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset

# Load a sample dataset to get an example
dataset = load_dataset("nielsr/funsd")
example = dataset["train"][0]
words = example["words"] # provides already pre-processed inputs for demonstration purposes
boxes = example["bboxes"] # provides already pre-processed inputs for demonstration purposes

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased", use_fast=False)
model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

# Process inputs
encoding = tokenizer(words, boxes=boxes, is_split_into_words=True, return_tensors="pt")

# Get predictions
outputs = model(**encoding)

print(outputs.logits.shape) # (1, 148, 7) - batch_size, sequence_length, num_labels
```

</hfoption>
<hfoption id="transformers-cli">

For the original LayoutLM version, `transformers-cli` usage is not readily applicable unlike later versions such as [**LayoutLMv3**](https://huggingface.co/docs/transformers/en/model_doc/layoutlmv3). 

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. 

The example below uses `TorchAoConfig` to only quantize the weights to `int8`.

```python
# pip install torchao accelerate
import torch
from transformers import TorchAoConfig, LayoutLMForTokenClassification, AutoTokenizer
from datasets import load_dataset

# Define the quantization configuration
quantization_config = TorchAoConfig("int8_weight_only", group_size=128)

# Load the model with 8-bit quantization
model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased", use_fast=False)

# Example input
dataset = load_dataset("nielsr/funsd")
example = dataset["train"][0]
encoding = tokenizer(example["words"], boxes=example["bboxes"], is_split_into_words=True, return_tensors="pt")
inputs = encoding

# Perform inference
outputs = model(**inputs)
print(outputs)
```

## Notes

- The original LayoutLM was not designed with a unified processing workflow. Instead, it expects preprocessed text (`words`) and bounding boxes (`boxes`) from an external OCR engine (like [Pytesseract](https://pypi.org/project/pytesseract/)) and provide them as additional inputs to the tokenizer. 

- The [`~LayoutLM.forward`] method expects the input `bbox` (bounding boxes of the input tokens). Each bounding box should be in the format `(x0, y0, x1, y1)`.  `(x0, y0)` corresponds to the upper left corner of the bounding box and `{x1, y1)` corresponds to the lower right corner. The bounding boxes need to be normalized on a 0-1000 scale as shown below.

```python
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

- `width` and `height` correspond to the width and height of the original document in which the token occurs. These values can be obtained as shown below.

```python
from PIL import Image

# Document can be a png, jpg, etc. PDFs must be converted to images.
image = Image.open(name_of_your_document).convert("RGB")

width, height = image.size
```


- [**AttentionMaskVisualizer**](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) is not directly compatible with LayoutLM's multi-modal output (text and layout) and attention structure (as opposed to text-based models like BERT with standard attention mechanisms).  


### LayoutLMConfig

[[autodoc]] LayoutLMConfig

### LayoutLMTokenizer

[[autodoc]] LayoutLMTokenizer
    - __call__

### LayoutLMTokenizerFast

[[autodoc]] LayoutLMTokenizerFast
    - __call__

<frameworkcontent>
<pt>

### LayoutLMModel

[[autodoc]] LayoutLMModel

### LayoutLMForMaskedLM

[[autodoc]] LayoutLMForMaskedLM

### LayoutLMForSequenceClassification

[[autodoc]] LayoutLMForSequenceClassification

### LayoutLMForTokenClassification

[[autodoc]] LayoutLMForTokenClassification

### LayoutLMForQuestionAnswering

[[autodoc]] LayoutLMForQuestionAnswering

</pt>
<tf>

### TFLayoutLMModel

[[autodoc]] TFLayoutLMModel

### TFLayoutLMForMaskedLM

[[autodoc]] TFLayoutLMForMaskedLM

### TFLayoutLMForSequenceClassification

[[autodoc]] TFLayoutLMForSequenceClassification

### TFLayoutLMForTokenClassification

[[autodoc]] TFLayoutLMForTokenClassification

### TFLayoutLMForQuestionAnswering

[[autodoc]] TFLayoutLMForQuestionAnswering

</tf>
</frameworkcontent>

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

- [The official GitHub repository for LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm)

🚀 Deploy

- A blog post on how to [Deploy LayoutLM with Hugging Face Inference Endpoints](https://www.philschmid.de/inference-endpoints-layoutlm).