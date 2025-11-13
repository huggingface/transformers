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
*This model was released on 2020-12-29 and added to Hugging Face Transformers on 2021-08-30.*

# LayoutLMV2

[LayoutLMv2](https://huggingface.co/papers/2012.14740) enhances LayoutLM by pre-training text, layout, and image in a multi-modal framework, incorporating masked visual-language modeling, text-image alignment, and text-image matching tasks. It also integrates a spatial-aware self-attention mechanism into the Transformer architecture to better understand relative positional relationships among text blocks. This results in state-of-the-art performance on various document image understanding tasks, including information extraction from scanned documents, document image classification, and document visual question answering. Improvements are shown across datasets such as FUNSD, CORD, SROIE, Kleister-NDA, RVL-CDIP, and DocVQA.

<hfoptions id="usage">
<hfoption id="LayoutLMv2ForQuestionAnswering">

```py
import torch
from transformers import AutoProcessor, LayoutLMv2ForQuestionAnswering
from PIL import Image
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased", dtype="auto")

dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
image = dataset["test"][0]["image"]
question = "When is coffee break?"
encoding = processor(image, question, return_tensors="pt")

outputs = model(**encoding)
predicted_start_idx = outputs.start_logits.argmax(-1).item()
predicted_end_idx = outputs.end_logits.argmax(-1).item()
predicted_start_idx, predicted_end_idx

predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
print(processor.tokenizer.decode(predicted_answer_tokens))
```

</hfoption>
</hfoptions>

## Usage tips

- LayoutLMv2 incorporates visual embeddings during pre-training, unlike LayoutLMv1 which only adds them during fine-tuning.
- LayoutLMv2 adds relative 1D attention bias and spatial 2D attention bias to self-attention layers. Details are on page 5 of the paper.
- Demo notebooks for RVL-CDIP, FUNSD, DocVQA, and CORD are available [here](https://github.com/microsoft/unilm/tree/master/layoutlmv2).
- LayoutLMv2 uses Facebook AI's Detectron2 package for its visual backbone. See [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
- The [`~LayoutLMv2Model.forward`] method expects `image` and `bbox` inputs in addition to `input_ids`. The image input corresponds to the original document image where text tokens occur. Each document image must be 224×224 pixels. For batches, use tensor shape `(batch_size, 3, 224, 224)`. This can be either a `torch.Tensor` or `Detectron2.structures.ImageList`. The model handles channel normalization automatically. The visual backbone expects BGR channels instead of RGB, as Detectron2 models are pre-trained using BGR format.
- The `bbox` input contains bounding boxes (2D positions) of input text tokens, identical to LayoutLMModel. Get these from external OCR engines like Google's Tesseract. Use `(x0, y0, x1, y1)` format where `(x0, y0)` is the upper left corner and `(x1, y1)` is the lower right corner. Normalize bounding boxes to a 0-1000 scale.
- [`LayoutLMv2Processor`] prepares data for the model directly, including OCR processing. More information is in the "Usage" section below.
- Internally, [`LayoutLMv2Model`] sends image input through its visual backbone to get a lower-resolution feature map. The feature map shape equals the `image_feature_pool_shape` attribute of [`LayoutLMv2Config`]. This feature map flattens to get image tokens. With default 7×7 feature map size, you get 49 image tokens. These concatenate with text tokens and send through the Transformer encoder. Last hidden states have length 512 + 49 = 561 if you pad text tokens to max length. More generally, last hidden states have shape `seq_length + image_feature_pool_shape[0] * config.image_feature_pool_shape[1]`.
- [`from_pretrained`] prints warnings about uninitialized parameters. This isn't a problem, as these are batch normalization statistics that get values during fine-tuning on custom datasets.
- For distributed training, call `synchronize_batch_norm` on the model to properly synchronize batch normalization layers of the visual backbone.
- Use [`LayoutLMv2Processor`] to prepare data for the model. It combines [`LayoutLMv2ImageProcessor`] and [`LayoutLMv2Tokenizer`] or [`LayoutLMv2TokenizerFast`]. The image processor handles image modality, while the tokenizer handles text modality. Use both separately if you only want to handle one modality.
- Provide a document image (and optional additional data) to [`LayoutLMv2Processor`], and it creates inputs expected by the model. Internally, the processor uses [`LayoutLMv2ImageProcessor`] to apply OCR on the image, getting words and normalized bounding boxes, and resizing the image to get the image input. Words and normalized bounding boxes go to [`LayoutLMv2Tokenizer`] or [`LayoutLMv2TokenizerFast`], which converts them to token-level `input_ids`, `attention_mask`, `token_type_ids`, and `bbox`. Optionally provide word labels to the processor, which turn into token-level labels.
- [`LayoutLMv2Processor`] uses PyTesseract, a Python wrapper around Google's Tesseract OCR engine. Use your own OCR engine by providing words and normalized boxes yourself. Initialize [`LayoutLMv2ImageProcessor`] with `apply_ocr=False`.

## LayoutLMv2Config

[[autodoc]] LayoutLMv2Config

## LayoutLMv2ImageProcessor

[[autodoc]] LayoutLMv2ImageProcessor
    - preprocess

## LayoutLMv2ImageProcessorFast

[[autodoc]] LayoutLMv2ImageProcessorFast
    - preprocess

## LayoutLMv2Tokenizer

[[autodoc]] LayoutLMv2Tokenizer
    - __call__
    - save_vocabulary

## LayoutLMv2TokenizerFast

[[autodoc]] LayoutLMv2TokenizerFast
    - __call__

## LayoutLMv2Processor

[[autodoc]] LayoutLMv2Processor
    - __call__

## LayoutLMv2Model

[[autodoc]] LayoutLMv2Model
    - forward

## LayoutLMv2ForSequenceClassification

[[autodoc]] LayoutLMv2ForSequenceClassification

## LayoutLMv2ForTokenClassification

[[autodoc]] LayoutLMv2ForTokenClassification

## LayoutLMv2ForQuestionAnswering

[[autodoc]] LayoutLMv2ForQuestionAnswering

