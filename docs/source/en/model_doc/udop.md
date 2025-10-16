<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2022-12-05 and added to Hugging Face Transformers on 2024-03-04 and contributed by [nielsr](https://huggingface.co/nielsr).*

# UDOP

[Unifying Vision, Text, and Layout for Universal Document Processing](https://huggingface.co/papers/2212.02623) is a foundation Document AI model that integrates text, image, and layout information into a single representation using a Vision-Text-Layout Transformer. It supports multiple document tasks, including understanding, question answering, and generation, through a unified prompt-based sequence generation framework. The model is pretrained on large-scale unlabeled documents with self-supervised objectives and also leverages labeled data, enabling it to generate document images from text and layout via masked image reconstruction. UDOP achieves state-of-the-art performance across eight Document AI tasks and leads the Document Understanding Benchmark.

<hfoptions id="usage">
<hfoption id="UdopForConditionalGeneration">

```py
from transformers import AutoProcessor, UdopForConditionalGeneration
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
model = UdopForConditionalGeneration.from_pretrained("microsoft/udop-large", dtype="auto")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]

question = "Question answering. What is the date on the form?"
encoding = processor(image, question, text_pair=words, boxes=boxes, return_tensors="pt")

predicted_ids = model.generate(**encoding)
print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## Usage tips

- [`UdopForConditionalGeneration`] expects `input_ids` and `bbox` (bounding boxes/2D positions of input tokens). Get these from external OCR engines like Google's Tesseract (Python wrapper available). Each bounding box uses `(x0, y0, x1, y1)` format where `(x0, y0)` is the upper left corner and `(x1, y1)` is the lower right corner. Normalize bounding boxes to a 0-1000 scale first.
- Use [`UdopProcessor`] to prepare images and text for the model. This handles everything automatically. By default, this class uses the Tesseract engine to extract words and boxes (coordinates) from documents. Its functionality matches [`LayoutLMv3Processor`], so it supports `apply_ocr=False` for your own OCR engine or `apply_ocr=True` for the default OCR engine. See the LayoutLMv2 usage guide for all possible use cases (functionality is identical).
- For custom OCR engines, Azure's Read API works well and supports line segments. Segment position embeddings typically improve performance.
- Use the [`generate`] method at inference time to autoregressively generate text from document images.
- The model pre-trains on both self-supervised and supervised objectives. Use various task prefixes (prompts) from pre-training to test out-of-the-box capabilities. For example, prompt with "Question answering. What is the date?" since "Question answering." is the task prefix used during pre-training for DocVQA. See the paper (Table 1) for all task prefixes.
- Fine-tune [`UdopEncoderModel`], the encoder-only part of UDOP (similar to LayoutLMv3-like Transformer encoder). For discriminative tasks, add a linear classifier on top and fine-tune on labeled datasets.

## UdopConfig

[[autodoc]] UdopConfig

## UdopTokenizer

[[autodoc]] UdopTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## UdopTokenizerFast

[[autodoc]] UdopTokenizerFast

## UdopProcessor

[[autodoc]] UdopProcessor
    - __call__

## UdopModel

[[autodoc]] UdopModel
    - forward

## UdopForConditionalGeneration

[[autodoc]] UdopForConditionalGeneration
    - forward

## UdopEncoderModel

[[autodoc]] UdopEncoderModel
    - forward