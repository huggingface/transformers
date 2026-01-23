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
*This model was released on 2022-04-18 and added to Hugging Face Transformers on 2022-05-24.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# LayoutLMv3

[LayoutLMv3](https://huggingface.co/papers/2204.08387) is a multimodal transformer model designed specifically for Document AI tasks. It unites the pretraining objective for text and images, masked language and masked image modeling, and also includes a word-patch alignment objective for even stronger text and image alignment. The model architecture is also unified and uses a more streamlined approach with patch embeddings (similar to [ViT](./vit)) instead of a CNN backbone.

The model is pre-trained on three key objectives:
1. Masked Language Modeling (MLM) for text understanding
2. Masked Image Modeling (MIM) for visual understanding
3. Word-Patch Alignment (WPA) for learning cross-modal relationships
The LayoutLMv3 model was proposed in [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://huggingface.co/papers/2204.08387) by Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei.
LayoutLMv3 simplifies [LayoutLMv2](layoutlmv2) by using patch embeddings (as in [ViT](vit)) instead of leveraging a CNN backbone, and pre-trains the model on 3 objectives: masked language modeling (MLM), masked image modeling (MIM)
and word-patch alignment (WPA).

This unified architecture and training approach makes LayoutLMv3 particularly effective for both text-centric tasks (like form understanding and receipt analysis) and image-centric tasks (like document classification and layout analysis).

You can find all the original LayoutLMv3 checkpoints under the [LayoutLM](https://huggingface.co/collections/microsoft/layoutlm-6564539601de72cb631d0902) collection.  


> [!TIP]
> Click on the LayoutLMv3 models in the right sidebar for more examples of how to apply LayoutLMv3 to different vision and language tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.
<small> LayoutLMv3 architecture. Taken from the <a href="https://huggingface.co/papers/2204.08387">original paper</a>. </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/microsoft/unilm/tree/master/layoutlmv3).

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

# For document classification
classifier = pipeline("document-classification", model="microsoft/layoutlmv3-base")
result = classifier("document.jpg")

# For token classification (e.g., form understanding)
token_classifier = pipeline("token-classification", model="microsoft/layoutlmv3-base")
result = token_classifier("form.jpg")

# For question answering
qa = pipeline(task="document-question-answering", model="microsoft/layoutlmv3-base", torch_dtype=torch.bfloat16, device=0)  
result = qa(question="What is the total amount?", image="receipt.jpg")
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForDocumentQuestionAnswering, AutoProcessor

# Load model and processor
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
model = AutoModelForDocumentQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

# Process inputs
image = Image.open("document.jpg").convert("RGB")
encoding = processor(image, return_tensors="pt")
- In terms of data processing, LayoutLMv3 is identical to its predecessor [LayoutLMv2](layoutlmv2), except that:
  - images need to be resized and normalized with channels in regular RGB format. LayoutLMv2 on the other hand normalizes the images internally and expects the channels in BGR format.
  - text is tokenized using byte-pair encoding (BPE), as opposed to WordPiece.
  Due to these differences in data preprocessing, one can use [`LayoutLMv3Processor`] which internally combines a [`LayoutLMv3ImageProcessor`] (for the image modality) and a [`LayoutLMv3Tokenizer`]/[`LayoutLMv3TokenizerFast`] (for the text modality) to prepare all data for the model.
- Regarding usage of [`LayoutLMv3Processor`], we refer to the [usage guide](layoutlmv2#usage-layoutlmv2processor) of its predecessor.

# Get predictions
outputs = model(**encoding)
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli document-classification "document.jpg" --model microsoft/layoutlmv3-base
```

</hfoption>
</hfoptions>

For large models, you can use quantization to reduce memory usage. The example below demonstrates how to quantize the weights to 8-bit precision using the `TorchAoConfig` configuration.

```python
# pip install torchao
import torch
from transformers import TorchAoConfig, LayoutLMv3ForSequenceClassification, AutoProcessor

# Define the quantization configuration
quantization_config = TorchAoConfig("int8_weight_only", group_size=128)

# Load the model with 8-bit quantization
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

# Load the processor
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")

# Example input
image_path = "document.jpg"
inputs = processor(image_path, return_tensors="pt").to("cuda")

# Perform inference
outputs = model(**inputs)
print(outputs)
```

## Notes

- In terms of data processing, LayoutLMv3 is identical to its predecessor [LayoutLMv2](layoutlmv2), except that:
    - images need to be resized and normalized with channels in regular RGB format. LayoutLMv2 on the other hand normalizes the images internally and expects the channels in BGR format.
    - text is tokenized using byte-pair encoding (BPE), as opposed to WordPiece.
  Due to these differences in data preprocessing, one can use [`LayoutLMv3Processor`] which internally combines a [`LayoutLMv3ImageProcessor`] (for the image modality) and a [`LayoutLMv3Tokenizer`]/[`LayoutLMv3TokenizerFast`] (for the text modality) to prepare all data for the model.
- Regarding usage of [`LayoutLMv3Processor`], we refer to the [usage guide](layoutlmv2#usage-layoutlmv2processor) of its predecessor.

**Document question answering**

- [Document question answering task guide](../tasks/document_question_answering)

### LayoutLMv3Config

[[autodoc]] LayoutLMv3Config

### LayoutLMv3FeatureExtractor

[[autodoc]] LayoutLMv3FeatureExtractor
    - __call__

### LayoutLMv3Tokenizer

[[autodoc]] LayoutLMv3Tokenizer
    - __call__

### LayoutLMv3ImageProcessor
## LayoutLMv3ImageProcessor

[[autodoc]] LayoutLMv3ImageProcessor
    - preprocess

### LayoutLMv3Tokenizer
## LayoutLMv3ImageProcessorFast

[[autodoc]] LayoutLMv3ImageProcessorFast
    - preprocess

## LayoutLMv3Tokenizer

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
## LayoutLMv3Model

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
    - forward
