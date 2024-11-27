<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->

# TrOCR

## Overview

The TrOCR model was proposed in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained
Models](https://arxiv.org/abs/2109.10282) by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang,
Zhoujun Li, Furu Wei. TrOCR consists of an image Transformer encoder and an autoregressive text Transformer decoder to
perform [optical character recognition (OCR)](https://en.wikipedia.org/wiki/Optical_character_recognition).

The abstract from the paper is the following:

*Text recognition is a long-standing research problem for document digitalization. Existing approaches for text recognition
are usually built based on CNN for image understanding and RNN for char-level text generation. In addition, another language
model is usually needed to improve the overall accuracy as a post-processing step. In this paper, we propose an end-to-end
text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR, which leverages the
Transformer architecture for both image understanding and wordpiece-level text generation. The TrOCR model is simple but
effective, and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. Experiments
show that the TrOCR model outperforms the current state-of-the-art models on both printed and handwritten text recognition
tasks.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/trocr_architecture.jpg"
alt="drawing" width="600"/>

<small> TrOCR architecture. Taken from the <a href="https://arxiv.org/abs/2109.10282">original paper</a>. </small>

Please refer to the [`VisionEncoderDecoder`] class on how to use this model.

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found
[here](https://github.com/microsoft/unilm/tree/6f60612e7cc86a2a1ae85c47231507a587ab4e01/trocr).

## Usage tips

- The quickest way to get started with TrOCR is by checking the [tutorial
  notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/TrOCR), which show how to use the model
  at inference time as well as fine-tuning on custom data.
- TrOCR is pre-trained in 2 stages before being fine-tuned on downstream datasets. It achieves state-of-the-art results
  on both printed (e.g. the [SROIE dataset](https://paperswithcode.com/dataset/sroie) and handwritten (e.g. the [IAM
  Handwriting dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database>) text recognition tasks. For more
  information, see the [official models](https://huggingface.co/models?other=trocr>).
- TrOCR is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with TrOCR. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-classification"/>

- A blog post on [Accelerating Document AI](https://huggingface.co/blog/document-ai) with TrOCR.
- A blog post on how to [Document AI](https://github.com/philschmid/document-ai-transformers) with TrOCR.
- A notebook on how to [finetune TrOCR on IAM Handwriting Database using Seq2SeqTrainer](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb).
- A notebook on [inference with TrOCR](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference_with_TrOCR_%2B_Gradio_demo.ipynb) and Gradio demo.
- A notebook on [finetune TrOCR on the IAM Handwriting Database](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb) using native PyTorch.
- A notebook on [evaluating TrOCR on the IAM test set](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Evaluating_TrOCR_base_handwritten_on_the_IAM_test_set.ipynb).

<PipelineTag pipeline="text-generation"/>

- [Casual language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) task guide.

⚡️ Inference

- An interactive-demo on [TrOCR handwritten character recognition](https://huggingface.co/spaces/nielsr/TrOCR-handwritten).

## Inference

TrOCR's [`VisionEncoderDecoder`] model accepts images as input and makes use of
[`~generation.GenerationMixin.generate`] to autoregressively generate text given the input image.

The [`ViTImageProcessor`/`DeiTImageProcessor`] class is responsible for preprocessing the input image and
[`RobertaTokenizer`/`XLMRobertaTokenizer`] decodes the generated target tokens to the target string. The
[`TrOCRProcessor`] wraps [`ViTImageProcessor`/`DeiTImageProcessor`] and [`RobertaTokenizer`/`XLMRobertaTokenizer`]
into a single instance to both extract the input features and decode the predicted token ids.

- Step-by-step Optical Character Recognition (OCR)

``` py
>>> from transformers import TrOCRProcessor, VisionEncoderDecoderModel
>>> import requests
>>> from PIL import Image

>>> processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
>>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

>>> # load image from the IAM dataset
>>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> pixel_values = processor(image, return_tensors="pt").pixel_values
>>> generated_ids = model.generate(pixel_values)

>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

See the [model hub](https://huggingface.co/models?filter=trocr) to look for TrOCR checkpoints.

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
