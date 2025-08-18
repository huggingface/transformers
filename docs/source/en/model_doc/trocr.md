<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->
*This model was released on 2021-09-21 and added to Hugging Face Transformers on 2021-10-13.*



<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

<!-- <div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div> -->

# TrOCR

The TrOCR model was proposed in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://huggingface.co/papers/2109.10282) by Minghao Li, Tengchao Lv, Lei Cui, and colleagues.

TrOCR, which stands for Transformer Optical Character Recognition, is an end-to-end model that's great at reading text from an image. It uses a Transformer-based architecture, combining a vision encoder to "see" the image and a text decoder to "write" out the characters it sees, making it highly effective for Optical Character Recognition (OCR).

You can find all the original TrOCR checkpoints under the [TrOCR collection](https://huggingface.co/models?other=trocr).


> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on the TrOCR models in the right sidebar for more examples of how to apply TrOCR to different image-to-text tasks.


The example below demonstrates how to perform Optical Character Recognition (OCR) with [`Pipeline`] or the [`VisionEncoderDecoderModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline
from PIL import Image
import requests

# Load an image from the IAM dataset
url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Initialize the OCR pipeline
ocr_pipeline = pipeline("image-to-text", model="microsoft/trocr-base-handwritten")

# Get the recognized text
generated_text = ocr_pipeline(image)
print(generated_text)
# [{'generated_text': 'industry, " Mr. Brown commented ,'}]
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# load image from the IAM dataset
url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

</hfoption>
<hfoption id="transformers-cli">
</hfoption>
</hfoptions>

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](https://huggingface.co/docs/transformers/main/en/quantization#bitsandbytes) to quantize the weights to 8-bit.

```python
# pip install bitsandbytes accelerate
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

# Use a large checkpoint for a more noticeable impact
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten", load_in_8bit=True)

# load image from the IAM dataset
url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

## Notes

- The quickest way to get started with TrOCR is by checking the [tutorial notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/TrOCR), which show how to use the model at inference time as well as fine-tuning on custom data.
- TrOCR is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework.

## Resources

- A blog post on [Accelerating Document AI](https://huggingface.co/blog/document-ai) with TrOCR.
- A blog post on how to [Document AI](https://github.com/philschmid/document-ai-transformers) with TrOCR.
- A notebook on how to [finetune TrOCR on IAM Handwriting Database using Seq2SeqTrainer](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb).
- An interactive-demo on [TrOCR handwritten character recognition](https://huggingface.co/spaces/nielsr/TrOCR-handwritten).


## TrOCRConfig

[[autodoc]] TrOCRConfig

## TrOCRProcessor

[[autodoc]] TrOCRProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## TrOCRForCausalLM

[[autodoc]] TrOCRForCausalLM
     - forward
