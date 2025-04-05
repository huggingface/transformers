<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Donut

[Donut (Document Understanding Transformer)](https://huggingface.co/papers2111.15664) is a visual document understanding model that doesn't require an Optical Character Recognition (OCR) engine. Unlike traditional approaches that extract text using OCR before processing, Donut employs an end-to-end Transformer-based architecture to directly analyze document images. This eliminates OCR-related inefficiencies making it more accurate and adaptable to diverse languages and formats. 

Donut features vision encoder ([Swin](./swin)) and a text decoder ([BART](./bart)). Swin converts document images into embeddings and BART processes them into meaningful text sequences.

You can find all the original DONUT checkpoints under the [DONUT](https://huggingface.co/models?other=donut) collection in Modelhub.

> [!TIP]
> Click on the DONUT models in the right sidebar for more examples of how to apply DONUT to different language and vision tasks.

The examples below demonstrate how to perform document understanding tasks using Donut with [`Pipeline`] and [`AutoModel`]

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline
from PIL import Image

pipeline = pipeline(
    task="document-question-answering",
    model="naver-clova-ix/donut-base-finetuned-docvqa",
    device=0,
    torch_dtype=torch.float16
)

pipeline(
    image=Image.open("path/to/document.png"),
    question="What is the purchase amount?"
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import DonutProcessor, AutoModelForVision2Seq
from PIL import Image

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = AutoModelForVision2Seq.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image = Image.open("path/to/document.png")
question = "What is the purchase amount?"

task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
encoding = processor(image, task_prompt, return_tensors="pt").to(device)

outputs = model.generate(
    input_ids=encoding.input_ids,
    pixel_values=encoding.pixel_values,
    max_length=512
)

answer = processor.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](https://huggingface.co/docs/transformers/main/en/quantization/overview) overview for more available quantization backends.

The example below uses [torchao](https://huggingface.co/docs/transformers/main/en/quantization/torchao) to only quantize the weights to int4.

```py
#pip install torchao
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, TorchAoConfig
from PIL import Image

model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
processor = DonutProcessor.from_pretrained(model_name)
quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = VisionEncoderDecoderModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image = Image.open("path/to/document.png")
question = "What is the purchase amount?"

task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
inputs = processor(image, task_prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_length=512)
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## Notes

- The quickest way to get started with Donut is by checking the [tutorial notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut), which show how to use the model at inference time as well as fine-tuning on custom data.
- Donut is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework.
- This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found
[here](https://github.com/clovaai/donut).

## DonutSwinConfig

[[autodoc]] DonutSwinConfig

## DonutImageProcessor

[[autodoc]] DonutImageProcessor
    - preprocess

## DonutFeatureExtractor

[[autodoc]] DonutFeatureExtractor
    - __call__

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
