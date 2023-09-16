<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Zero-shot image classification

[[open-in-colab]]

Zero-shot image classification is a task that involves classifying images into different categories using a model that was
not explicitly trained on data containing labeled examples from those specific categories.

Traditionally, image classification requires training a model on a specific set of labeled images, and this model learns to
"map" certain image features to labels. When there's a need to use such model for a classification task that introduces a
new set of labels, fine-tuning is required to "recalibrate" the model.

In contrast, zero-shot or open vocabulary image classification models are typically multi-modal models that have been trained on a large
dataset of images and associated descriptions. These models learn aligned vision-language representations that can be used for many downstream tasks including zero-shot image classification.

This is a more flexible approach to image classification that allows models to generalize to new and unseen categories
without the need for additional training data and enables users to query images with free-form text descriptions of their target objects .

In this guide you'll learn how to:

* create a zero-shot image classification pipeline
* run zero-shot image classification inference by hand

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install -q transformers
```

## Zero-shot image classification pipeline

The simplest way to try out inference with a model supporting zero-shot image classification is to use the corresponding [`pipeline`].
Instantiate a pipeline from a [checkpoint on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads):

```python
>>> from transformers import pipeline

>>> checkpoint = "openai/clip-vit-large-patch14"
>>> detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
```

Next, choose an image you'd like to classify.

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/owl.jpg" alt="Photo of an owl"/>
</div>

Pass the image and the candidate object labels to the pipeline. Here we pass the image directly; other suitable options
include a local path to an image or an image url.
The candidate labels can be simple words like in this example, or more descriptive.

```py
>>> predictions = classifier(image, candidate_labels=["fox", "bear", "seagull", "owl"])
>>> predictions
[{'score': 0.9996670484542847, 'label': 'owl'},
 {'score': 0.000199399160919711, 'label': 'seagull'},
 {'score': 7.392891711788252e-05, 'label': 'fox'},
 {'score': 5.96074532950297e-05, 'label': 'bear'}]
```

## Zero-shot image classification by hand

Now that you've seen how to use the zero-shot image classification pipeline, let's take a look how you can run zero-shot
image classification manually.

Start by loading the model and associated processor from a [checkpoint on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads).
Here we'll use the same checkpoint as before:

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

>>> model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

Let's take a different image to switch things up.

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" alt="Photo of a car"/>
</div>

Use the processor to prepare the inputs for the model. The processor combines an image processor that prepares the
image for the model by resizing and normalizing it, and a tokenizer that takes care of the text inputs.

```py
>>> candidate_labels = ["tree", "car", "bike", "cat"]
>>> inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
```

Pass the inputs through the model, and post-process the results:

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits_per_image[0]
>>> probs = logits.softmax(dim=-1).numpy()
>>> scores = probs.tolist()

>>> result = [
...     {"score": score, "label": candidate_label}
...     for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
... ]

>>> result
[{'score': 0.998572, 'label': 'car'},
 {'score': 0.0010570387, 'label': 'bike'},
 {'score': 0.0003393686, 'label': 'tree'},
 {'score': 3.1572064e-05, 'label': 'cat'}]
```