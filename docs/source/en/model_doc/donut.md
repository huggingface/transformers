<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->
*This model was released on 2021-11-30 and added to Hugging Face Transformers on 2022-08-12 and contributed by [nielsr](https://huggingface.co/nielsr).*

# Donut

[Donut](https://huggingface.co/papers/2111.15664) is an OCR-free Document Understanding Transformer that combines an image Transformer encoder with an autoregressive text Transformer decoder. It addresses the challenges of OCR-based approaches by eliminating the need for OCR, thus reducing computational costs, enhancing flexibility across languages and document types, and preventing OCR errors. Donut achieves state-of-the-art performance in document image classification, form understanding, and visual question answering, demonstrating both speed and accuracy. Additionally, a synthetic data generator is provided to support flexible pre-training across various languages and domains.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline
from PIL import Image

pipeline = pipeline(task="document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa", dtype="auto")
dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]

pipeline(image=image, question="What time is the coffee break?")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = AutoModelForVision2Seq.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", dtype="auto")

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
question = "What time is the coffee break?"
task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
inputs = processor(image, task_prompt, return_tensors="pt")

outputs = model.generate(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values, max_length=512)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## DonutSwinConfig

[[autodoc]] DonutSwinConfig

## DonutImageProcessor

[[autodoc]] DonutImageProcessor
    - preprocess

## DonutImageProcessorFast

[[autodoc]] DonutImageProcessorFast
    - preprocess

## DonutProcessor

[[autodoc]] DonutProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## DonutSwinModel

[[autodoc]] DonutSwinModel
    - forward

## DonutSwinForImageClassification

[[autodoc]] DonutSwinForImageClassification
    - forward

