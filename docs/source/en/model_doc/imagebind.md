<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ImageBind

## Overview

The ImageBind model was proposed in [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665) by Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra.
ImageBind is a multimodal joint embedding model for image/video, text, audio, depth, IMU, and thermal images.
For any input from these six modalities, it outputs the same-sized embedding that can be used for cross-modal and multimodal tasks.

The abstract from the paper is the following:

*We present ImageBind, an approach to learn a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. We show that all combinations of paired data are not necessary to train such a joint embedding, and only image-paired data is sufficient to bind the modalities together. ImageBind can leverage recent large scale vision-language models, and extends their zero-shot capabilities to new modalities just by using their natural pairing with images. It enables novel emergent applications 'out-of-the-box' including cross-modal retrieval, composing modalities with arithmetic, cross-modal detection and generation. The emergent capabilities improve with the strength of the image encoder and we set a new state-of-the-art on emergent zero-shot recognition tasks across modalities, outperforming specialist supervised models. Finally, we show strong few-shot recognition results outperforming prior work, and that ImageBind serves as a new way to evaluate vision models for visual and non-visual tasks.*

This model was contributed by [EduardoPacheco](https://huggingface.co/EduardoPacheco) and [dg845](https://huggingface.co/dg845) and [shehan97](https://huggingface.co/shehan97).
The original code can be found [here](https://github.com/facebookresearch/ImageBind).

## Usage tips

- ImageBind can be used for multi-modality similarity and zero-shot tasks.
- Currently only Vision (image and video), Audio and Text are supported.
- One can use [`ImageBindProcessor`] to prepare all or pairs of the available modalities.
- [`ImageBindModel`] `forward` expects only one pair of modalities where one of those MUST be vision modality.
- If interest only on the modalities embeddings one can use [`ImageBindModel`] `get_xxx_features` method or the appropriate `ImageBindXxxModelWithProjection`
- As ImageBind vision and text encoders were frozen during training and are initialized with OpenCLIP ViT-H if one has an application using this model the addition of other modalities by including other encoders would be possible.

Here's one example of how to get the embeddings for images, text and audios (this example requires `torchaudio`!)

```python
import torch
import torchaudio
from datasets import load_dataset
from transformers import ImageBindModel, ImageBindProcessor

ds = load_dataset("EduardoPacheco/imagebind-example-data", split="train")
images = ds["image"]
text = ds["text"]
audios = ds["audio"] # It's a dict with keys -> array and sampling_rate
audios = [
    torchaudio.functional.resample(
        torch.from_numpy(audio["array"]), 
        orig_freq=audio["sampling_rate"], 
        new_freq=16000
    ).numpy() 
    for audio in audios
]

model = ImageBindModel.from_pretrained("EduardoPacheco/imagebind-huge")
processor = ImageBindProcessor.from_pretrained("EduardoPacheco/imagebind-huge")

inputs = processor(text=text, images=images, audios=audios, padding=True, return_tensors="pt")

with torch.no_grad():
    audio_embeds = model.get_audio_features(input_features=inputs.input_features)
    image_embeds = model.get_image_features(pixel_values=inputs.pixel_values)
    text_embeds = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

# we can compute probs to use for retrieval or zero-shot workflows.
probs_image_text = (image_embeds @ text_embeds.T).softmax(dim=-1)
probs_text_audio = (text_embeds @ audio_embeds.T).softmax(dim=-1)
probs_image_audio = (image_embeds @ audio_embeds.T).softmax(dim=-1)
```

## ImageBindConfig

[[autodoc]] ImageBindConfig
    - from_text_vision_configs

## ImageBindTextConfig

[[autodoc]] ImageBindTextConfig

## ImageBindVisionConfig

[[autodoc]] ImageBindVisionConfig

## ImageBindAudioConfig

[[autodoc]] ImageBindAudioConfig

## ImageBindImageProcessor

[[autodoc]] ImageBindImageProcessor
    - preprocess

## ImageBindFeatureExtractor

[[autodoc]] ImageBindFeatureExtractor

## ImageBindProcessor

[[autodoc]] ImageBindProcessor

## ImageBindModel

[[autodoc]] ImageBindModel
    - forward
    - get_text_features
    - get_image_features
    - get_audio_features

## ImageBindTextModel

[[autodoc]] ImageBindTextModel
    - forward

## ImageBindTextModelWithProjection

[[autodoc]] ImageBindTextModelWithProjection
    - forward

## ImageBindVisionModel

[[autodoc]] ImageBindVisionModel
    - forward


## ImageBindVisionModelWithProjection

[[autodoc]] ImageBindVisionModelWithProjection
    - forward

## ImageBindAudioModel

[[autodoc]] ImageBindAudioModel
    - forward

## ImageBindAudioModelWithProjection

[[autodoc]] ImageBindAudioModelWithProjection
    - forward