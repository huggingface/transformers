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

# TrOCR

[TrOCR](https://huggingface.co/papers/2109.10282) is a text recognition model for both image understanding and text generation. It doesn't require separate models for image processing or character generation. TrOCR is a simple single end-to-end system that uses a transformer to handle visual understanding and text generation.

You can find all the original TrOCR checkpoints under the [Microsoft](https://huggingface.co/microsoft/models?search=trocr) organization.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/trocr_architecture.jpg"
alt="drawing" width="600"/>
<small> TrOCR architecture. Taken from the <a href="https://huggingface.co/papers/2109.10282">original paper</a>. </small>


> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on the TrOCR models in the right sidebar for more examples of how to apply TrOCR to different image and text tasks.


The example below demonstrates how to perform optical character recognition (OCR) with the [`AutoModel`] class.

<hfoptions id="usage">
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
</hfoptions>

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 8-bits.

```python
# pip install bitsandbytes accelerate
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, BitsandBytesConfig
import requests
from PIL import Image

# Set up the quantization configuration
quantization_config = BitsandBytesConfig(load_in_8bit=True)

# Use a large checkpoint for a more noticeable impact
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten",
    quantization_config=quantization_config
)

# load image from the IAM dataset
url = "[https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg](https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg)"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

## Notes

- TrOCR wraps [`ViTImageProcessor`]/[`DeiTImageProcessor`] and [`RobertaTokenizer`]/[`XLMRobertaTokenizer`] into a single instance of [`TrOCRProcessor`] to handle images and text.
- TrOCR is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework.

## Resources

- A blog post on [Accelerating Document AI](https://huggingface.co/blog/document-ai) with TrOCR.
- A blog post on how to [Document AI](https://github.com/philschmid/document-ai-transformers) with TrOCR.
- A notebook on how to [finetune TrOCR on IAM Handwriting Database using Seq2SeqTrainer](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb).
- An interactive-demo on [TrOCR handwritten character recognition](https://huggingface.co/spaces/nielsr/TrOCR-handwritten).
- A notebook on [inference with TrOCR](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference_with_TrOCR_%2B_Gradio_demo.ipynb) and Gradio demo.
- A notebook on [evaluating TrOCR on the IAM test set](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Evaluating_TrOCR_base_handwritten_on_the_IAM_test_set.ipynb).


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
