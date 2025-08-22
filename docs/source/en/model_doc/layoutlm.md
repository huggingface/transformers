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
    </div>
</div>

# LayoutLM

[LayoutLM](https://huggingface.co/papers/1912.13318) jointly learns text and the document layout rather than focusing only on text. It incorporates positional layout information and visual features of words from the document images.

You can find all the original LayoutLM checkpoints under the [LayoutLM](https://huggingface.co/collections/microsoft/layoutlm-6564539601de72cb631d0902) collection.

> [!TIP]
> Click on the LayoutLM models in the right sidebar for more examples of how to apply LayoutLM to different vision and language tasks.

The example below demonstrates question answering with the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LayoutLMForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)
model = LayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", dtype=torch.float16)

dataset = load_dataset("nielsr/funsd", split="train")
example = dataset[0]
question = "what's his name?"
words = example["words"]
boxes = example["bboxes"]

encoding = tokenizer(
    question.split(),
    words,
    is_split_into_words=True,
    return_token_type_ids=True,
    return_tensors="pt"
)
bbox = []
for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
    if s == 1:
        bbox.append(boxes[w])
    elif i == tokenizer.sep_token_id:
        bbox.append([1000] * 4)
    else:
        bbox.append([0] * 4)
encoding["bbox"] = torch.tensor([bbox])

word_ids = encoding.word_ids(0)
outputs = model(**encoding)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
start, end = word_ids[start_scores.argmax(-1)], word_ids[end_scores.argmax(-1)]
print(" ".join(words[start : end + 1]))
```

</hfoption>
</hfoptions>

## Notes

- The original LayoutLM was not designed with a unified processing workflow. Instead, it expects preprocessed text (`words`) and bounding boxes (`boxes`) from an external OCR engine (like [Pytesseract](https://pypi.org/project/pytesseract/)) and provide them as additional inputs to the tokenizer.

- The [`~LayoutLMModel.forward`] method expects the input `bbox` (bounding boxes of the input tokens). Each bounding box should be in the format `(x0, y0, x1, y1)`.  `(x0, y0)` corresponds to the upper left corner of the bounding box and `{x1, y1)` corresponds to the lower right corner. The bounding boxes need to be normalized on a 0-1000 scale as shown below.

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

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LayoutLM. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- Read [fine-tuning LayoutLM for document-understanding using Keras & Hugging Face Transformers](https://www.philschmid.de/fine-tuning-layoutlm-keras) to learn more.
- Read [fine-tune LayoutLM for document-understanding using only Hugging Face Transformers](https://www.philschmid.de/fine-tuning-layoutlm) for more information.
- Refer to this [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb) for a practical example of how to fine-tune LayoutLM.
- Refer to this [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForSequenceClassification_on_RVL_CDIP.ipynb) for an example of how to fine-tune LayoutLM for sequence classification.
- Refer to this [notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb) for an example of how to fine-tune LayoutLM for token classification.
- Read [Deploy LayoutLM with Hugging Face Inference Endpoints](https://www.philschmid.de/inference-endpoints-layoutlm) to learn how to deploy LayoutLM.


## LayoutLMConfig

[[autodoc]] LayoutLMConfig

## LayoutLMTokenizer

[[autodoc]] LayoutLMTokenizer
    - __call__

## LayoutLMTokenizerFast

[[autodoc]] LayoutLMTokenizerFast
    - __call__

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
