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