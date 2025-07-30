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

You can find all the original Donut checkpoints under the [Naver Clova Information Extraction](https://huggingface.co/naver-clova-ix) organization.

> [!TIP]
> Click on the Donut models in the right sidebar for more examples of how to apply Donut to different language and vision tasks.

The examples below demonstrate how to perform document understanding tasks using Donut with [`Pipeline`] and [`AutoModel`]

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
# pip install datasets
import torch
from transformers import pipeline
from PIL import Image

pipeline = pipeline(
    task="document-question-answering",
    model="naver-clova-ix/donut-base-finetuned-docvqa",
    device=0,
    dtype=torch.float16
)
dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]

pipeline(image=image, question="What time is the coffee break?")
```

</hfoption>
<hfoption id="AutoModel">

```py
# pip install datasets
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = AutoModelForVision2Seq.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
question = "What time is the coffee break?"
task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
inputs = processor(image, task_prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs.input_ids,
    pixel_values=inputs.pixel_values,
    max_length=512
)
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```py
# pip install datasets torchao
import torch
from datasets import load_dataset
from transformers import TorchAoConfig, AutoProcessor, AutoModelForVision2Seq

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = AutoModelForVision2Seq.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", quantization_config=quantization_config)

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[0]["image"]
question = "What time is the coffee break?"
task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
inputs = processor(image, task_prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs.input_ids,
    pixel_values=inputs.pixel_values,
    max_length=512
)
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## Notes

- Use Donut for document image classification as shown below.

    ```py
    >>> import re
    >>> from transformers import DonutProcessor, VisionEncoderDecoderModel
    >>> from datasets import load_dataset
    >>> import torch

    >>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
    >>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> model.to(device)  # doctest: +IGNORE_RESULT

    >>> # load document image
    >>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
    >>> image = dataset[1]["image"]

    >>> # prepare decoder inputs
    >>> task_prompt = "<s_rvlcdip>"
    >>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    >>> pixel_values = processor(image, return_tensors="pt").pixel_values

    >>> outputs = model.generate(
    ...     pixel_values.to(device),
    ...     decoder_input_ids=decoder_input_ids.to(device),
    ...     max_length=model.decoder.config.max_position_embeddings,
    ...     pad_token_id=processor.tokenizer.pad_token_id,
    ...     eos_token_id=processor.tokenizer.eos_token_id,
    ...     use_cache=True,
    ...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
    ...     return_dict_in_generate=True,
    ... )

    >>> sequence = processor.batch_decode(outputs.sequences)[0]
    >>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    >>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    >>> print(processor.token2json(sequence))
    {'class': 'advertisement'}
    ```

- Use Donut for document parsing as shown below.

    ```py
    >>> import re
    >>> from transformers import DonutProcessor, VisionEncoderDecoderModel
    >>> from datasets import load_dataset
    >>> import torch

    >>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    >>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> model.to(device)  # doctest: +IGNORE_RESULT

    >>> # load document image
    >>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
    >>> image = dataset[2]["image"]

    >>> # prepare decoder inputs
    >>> task_prompt = "<s_cord-v2>"
    >>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    >>> pixel_values = processor(image, return_tensors="pt").pixel_values

    >>> outputs = model.generate(
    ...     pixel_values.to(device),
    ...     decoder_input_ids=decoder_input_ids.to(device),
    ...     max_length=model.decoder.config.max_position_embeddings,
    ...     pad_token_id=processor.tokenizer.pad_token_id,
    ...     eos_token_id=processor.tokenizer.eos_token_id,
    ...     use_cache=True,
    ...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
    ...     return_dict_in_generate=True,
    ... )

    >>> sequence = processor.batch_decode(outputs.sequences)[0]
    >>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    >>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    >>> print(processor.token2json(sequence))
    {'menu': {'nm': 'CINNAMON SUGAR', 'unitprice': '17,000', 'cnt': '1 x', 'price': '17,000'}, 'sub_total': {'subtotal_price': '17,000'}, 'total': 
    {'total_price': '17,000', 'cashprice': '20,000', 'changeprice': '3,000'}}
    ```

## DonutSwinConfig

[[autodoc]] DonutSwinConfig

## DonutImageProcessor

[[autodoc]] DonutImageProcessor
    - preprocess

## DonutImageProcessorFast

[[autodoc]] DonutImageProcessorFast
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

## DonutSwinForImageClassification

[[autodoc]] transformers.DonutSwinForImageClassification
    - forward