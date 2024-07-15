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

# CLIP

## Overview

The CLIP model was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. CLIP
(Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be
instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing
for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

The abstract from the paper is the following:

*State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This
restricted form of supervision limits their generality and usability since additional labeled data is needed to specify
any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a
much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes
with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400
million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference
learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study
the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks
such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The
model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need
for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot
without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained
model weights at this https URL.*

This model was contributed by [valhalla](https://huggingface.co/valhalla). The original code can be found [here](https://github.com/openai/CLIP).

## Usage tips and example

CLIP is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image
classification. CLIP uses a ViT like transformer to get visual features and a causal language model to get the text
features. Both the text and visual features are then projected to a latent space with identical dimension. The dot
product between the projected image and text features is then used as a similar score.

To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches,
which are then linearly embedded. A [CLS] token is added to serve as representation of an entire image. The authors
also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.
The [`CLIPImageProcessor`] can be used to resize (or rescale) and normalize images for the model.

The [`CLIPTokenizer`] is used to encode the text. The [`CLIPProcessor`] wraps
[`CLIPImageProcessor`] and [`CLIPTokenizer`] into a single instance to both
encode the text and prepare the images. The following example shows how to get the image-text similarity scores using
[`CLIPProcessor`] and [`CLIPModel`].


```python
>>> from PIL import Image
>>> import requests

>>> from transformers import CLIPProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```


### Combining CLIP and Flash Attention 2

First, make sure to install the latest version of Flash Attention 2.

```bash
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of flash-attn repository. Make also sure to load your model in half-precision (e.g. `torch.float16`)

<Tip warning={true}>

For small batch sizes, you might notice a slowdown in your model when using flash attention. Refer to the section [Expected speedups with Flash Attention and SDPA](#Expected-speedups-with-Flash-Attention-and-SDPA) below and select an appropriate attention implementation.

</Tip>

To load and run a model using Flash Attention 2, refer to the snippet below:

```python
>>> import torch
>>> import requests
>>> from PIL import Image

>>> from transformers import CLIPProcessor, CLIPModel

>>> device = "cuda"
>>> torch_dtype = torch.float16

>>> model = CLIPModel.from_pretrained(
...     "openai/clip-vit-base-patch32",
...     attn_implementation="flash_attention_2",
...     device_map=device,
...     torch_dtype=torch_dtype,
... )
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
>>> inputs.to(device)

>>> with torch.no_grad():
...     with torch.autocast(device):
...         outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
>>> print(probs)
tensor([[0.9946, 0.0052]], device='cuda:0', dtype=torch.float16)
```


### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```python
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16, attn_implementation="sdpa")
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

### Expected speedups with Flash Attention and SDPA

On a local benchmark (NVIDIA A10G, PyTorch 2.3.1+cu121) with `float16`, we saw the following speedups during inference for `"openai/clip-vit-large-patch14"` checkpoint ([code](https://gist.github.com/qubvel/ac691a54e54f9fae8144275f866a7ff8)):

#### CLIPTextModel

|   Num text labels |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                 4 |            0.007 |          0.011 |         0.677 |           0.006 |          1.175 |
|                16 |            0.007 |          0.013 |         0.577 |           0.007 |          1.056 |
|                64 |            0.029 |          0.03  |         0.966 |           0.026 |          1.094 |
|               128 |            0.052 |          0.049 |         1.069 |           0.047 |          1.108 |
|               256 |            0.103 |          0.092 |         1.115 |           0.092 |          1.109 |

![clip_text_model_viz_2](https://github.com/user-attachments/assets/8b6b4d87-4e2b-48ab-924d-e23c417fa48a)

#### CLIPVisionModel

|   Image batch size |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|-------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                  1 |            0.013 |          0.011 |         1.208 |           0.01  |          1.325 |
|                  4 |            0.021 |          0.018 |         1.134 |           0.018 |          1.138 |
|                 16 |            0.076 |          0.065 |         1.166 |           0.065 |          1.17  |
|                 32 |            0.149 |          0.128 |         1.165 |           0.127 |          1.168 |

![clip_image_model_viz_2](https://github.com/user-attachments/assets/20b7551b-21a5-4277-a3b2-0eae5a25521e)

#### CLIPModel

|   Image batch size |   Num text labels |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|-------------------:|------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                  1 |                 4 |            0.021 |          0.022 |         0.942 |           0.017 |          1.262 |
|                  1 |                16 |            0.021 |          0.024 |         0.877 |           0.017 |          1.261 |
|                  1 |                64 |            0.035 |          0.04  |         0.876 |           0.032 |          1.107 |
|                  4 |                 4 |            0.023 |          0.029 |         0.78  |           0.021 |          1.117 |
|                  4 |                16 |            0.028 |          0.031 |         0.895 |           0.025 |          1.105 |
|                  4 |                64 |            0.049 |          0.048 |         1.025 |           0.044 |          1.111 |
|                 16 |                 4 |            0.079 |          0.077 |         1.027 |           0.067 |          1.166 |
|                 16 |                16 |            0.084 |          0.079 |         1.063 |           0.072 |          1.156 |
|                 16 |                64 |            0.105 |          0.095 |         1.099 |           0.091 |          1.148 |
|                 32 |                 4 |            0.151 |          0.139 |         1.088 |           0.13  |          1.167 |
|                 32 |                16 |            0.156 |          0.141 |         1.108 |           0.134 |          1.162 |
|                 32 |                64 |            0.177 |          0.158 |         1.125 |           0.153 |          1.157 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CLIP.

- [Fine tuning CLIP with Remote Sensing (Satellite) images and captions](https://huggingface.co/blog/fine-tune-clip-rsicd), a blog post about how to fine-tune CLIP with [RSICD dataset](https://github.com/201528014227051/RSICD_optimal) and comparison of performance changes due to data augmentation.
- This [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text) shows how to train a CLIP-like vision-text dual encoder model using a pre-trained vision and text encoder using [COCO dataset](https://cocodataset.org/#home).

<PipelineTag pipeline="image-to-text"/>

- A [notebook](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing) on how to use a pretrained CLIP for inference with beam search for image captioning. 🌎

**Image retrieval**

- A [notebook](https://colab.research.google.com/drive/1bLVwVKpAndpEDHqjzxVPr_9nGrSbuOQd?usp=sharing) on image retrieval using pretrained CLIP and computing MRR(Mean Reciprocal Rank) score. 🌎
- A [notebook](https://colab.research.google.com/github/deep-diver/image_search_with_natural_language/blob/main/notebooks/Image_Search_CLIP.ipynb) on image retrieval and showing the similarity score. 🌎
- A [notebook](https://colab.research.google.com/drive/1xO-wC_m_GNzgjIBQ4a4znvQkvDoZJvH4?usp=sharing) on how to map images and texts to the same vector space using Multilingual CLIP. 🌎 
- A [notebook](https://colab.research.google.com/github/vivien000/clip-demo/blob/master/clip.ipynb#scrollTo=uzdFhRGqiWkR) on how to run CLIP on semantic image search using [Unsplash](https://unsplash.com) and [TMDB](https://www.themoviedb.org/) datasets. 🌎

**Explainability**

- A [notebook](https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb) on how to visualize similarity between input token and image segment. 🌎

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it.
The resource should ideally demonstrate something new instead of duplicating an existing resource.

## CLIPConfig

[[autodoc]] CLIPConfig
    - from_text_vision_configs

## CLIPTextConfig

[[autodoc]] CLIPTextConfig

## CLIPVisionConfig

[[autodoc]] CLIPVisionConfig

## CLIPTokenizer

[[autodoc]] CLIPTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CLIPTokenizerFast

[[autodoc]] CLIPTokenizerFast

## CLIPImageProcessor

[[autodoc]] CLIPImageProcessor
    - preprocess

## CLIPFeatureExtractor

[[autodoc]] CLIPFeatureExtractor

## CLIPProcessor

[[autodoc]] CLIPProcessor

<frameworkcontent>
<pt>

## CLIPModel

[[autodoc]] CLIPModel
    - forward
    - get_text_features
    - get_image_features

## CLIPTextModel

[[autodoc]] CLIPTextModel
    - forward

## CLIPTextModelWithProjection

[[autodoc]] CLIPTextModelWithProjection
    - forward

## CLIPVisionModelWithProjection

[[autodoc]] CLIPVisionModelWithProjection
    - forward

## CLIPVisionModel

[[autodoc]] CLIPVisionModel
    - forward

## CLIPForImageClassification

[[autodoc]] CLIPForImageClassification
    - forward

</pt>
<tf>

## TFCLIPModel

[[autodoc]] TFCLIPModel
    - call
    - get_text_features
    - get_image_features

## TFCLIPTextModel

[[autodoc]] TFCLIPTextModel
    - call

## TFCLIPVisionModel

[[autodoc]] TFCLIPVisionModel
    - call

</tf>
<jax>

## FlaxCLIPModel

[[autodoc]] FlaxCLIPModel
    - __call__
    - get_text_features
    - get_image_features

## FlaxCLIPTextModel

[[autodoc]] FlaxCLIPTextModel
    - __call__

## FlaxCLIPTextModelWithProjection

[[autodoc]] FlaxCLIPTextModelWithProjection
    - __call__

## FlaxCLIPVisionModel

[[autodoc]] FlaxCLIPVisionModel
    - __call__

</jax>
</frameworkcontent>
