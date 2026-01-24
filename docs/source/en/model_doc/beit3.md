<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# BEiT-3

## Overview

The BEiT-3 model was proposed in [Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language
Tasks](https://arxiv.org/abs/2208.10442) by Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu,
Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei.

BEiT-3 is a general-purpose multimodal foundation model that excels in both vision and vision-language tasks. It
utilizes [Multiway transformers](https://arxiv.org/abs/2208.10442) for deep fusion and modality-specific encoding,
and unifies masked modeling on images, texts, and image-text pairs, achieving top performance on multiple benchmarks.

The abstract from the paper is the following:

*A big convergence of language, vision, and multimodal pretraining is emerging. In this work, we introduce a
general-purpose multimodal foundation model BEiT-3, which achieves state-of-the-art transfer performance on both vision
and vision-language tasks. Specifically, we advance the big convergence from three aspects: backbone architecture,
pretraining task, and model scaling up. We introduce Multiway Transformers for general-purpose modeling, where the
modular architecture enables both deep fusion and modality-specific encoding. Based on the shared backbone, we perform
masked "language" modeling on images (Imglish), texts (English), and image-text pairs ("parallel sentences") in a
unified manner. Experimental results show that BEiT-3 obtains state-of-the-art performance on object detection (COCO),
image classification (ImageNet), visual reasoning (NLVR2), visual question answering
(VQAv2), image captioning (COCO), and cross-modal retrieval (Flickr30K, COCO).*

This model was contributed by [Raghavan](https://huggingface.co/Raghavan).
The original code can be found [here](https://github.com/microsoft/unilm/tree/master/beit3).

## Usage example

Here is a sample of Beit3 model for ImageClassification

```python
>>> from transformers import Beit3Processor, Beit3ForImageClassification
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg" # Image of a couch with remotes and cats
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")
>>> model = Beit3ForImageClassification.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")

>>> inputs = processor(images=image, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**inputs)

>>> predicted_class_idx = outputs.logits.argmax(-1).item()
>>> predicted_class = model.config.id2label[predicted_class_idx]
>>> print("Predicted class:", predicted_class)
Predicted class: remote control, remote
```

## Beit3Config

[[autodoc]] Beit3Config

## Beit3Processor

[[autodoc]] Beit3Processor

## Beit3Model

[[autodoc]] Beit3Model
- forward

## Beit3ForCaptioning

[[autodoc]] Beit3ForCaptioning
- forward

## Beit3ForImageClassification

[[autodoc]] Beit3ForImageClassification
- forward

## Beit3ForImageTextRetrieval

[[autodoc]] Beit3ForImageTextRetrieval
- forward

## Beit3ForQuestionAnswering

[[autodoc]] Beit3ForQuestionAnswering
- forward

## Beit3ForImagesAndTextClassification

[[autodoc]] Beit3ForImagesAndTextClassification
- forward
