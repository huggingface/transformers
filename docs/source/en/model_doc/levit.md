<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-04-02 and added to Hugging Face Transformers on 2022-06-01 and contributed by [anugunj](https://huggingface.co/anugunj).*

# LeViT

[LeViT: Introducing Convolutions to Vision Transformers](https://huggingface.co/papers/2104.01136) optimizes the trade-off between accuracy and efficiency in image classification by integrating principles from convolutional neural networks into transformers. Key innovations include activation maps with decreasing resolutions and the introduction of an attention bias to better integrate positional information. LeViT achieves significant improvements over existing convnets and vision transformers, offering faster inference times while maintaining high accuracy. For instance, at 80% ImageNet top-1 accuracy, LeViT is 5 times faster than EfficientNet on CPU.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/levit-128S", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/levit-128S")
model = AutoModelForImageClassification.from_pretrained("facebook/levit-128S", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- Fine-tune distilled models in two ways: (1) Classic fine-tuning places a prediction head only on the final hidden state without using the distillation head, or (2) Fine-tuning with distillation places both prediction head and distillation head on the final hidden state.
- For fine-tuning with distillation, the prediction head trains using regular cross-entropy between the prediction and ground-truth label. The distillation prediction head trains using hard distillation (cross-entropy between the distillation head prediction and teacher's predicted label). At inference, take the average prediction between both heads as the final prediction.
- Fine-tuning with distillation relies on a teacher already fine-tuned on the downstream dataset. Use [`LevitForImageClassification`] for classic fine-tuning and [`LevitForImageClassificationWithTeacher`] for fine-tuning with distillation.
- All released checkpoints were pre-trained and fine-tuned on ImageNet-1k (ILSVRC 2012, 1.3 million images and 1,000 classes) only. No external data was used. This contrasts with the original ViT model, which used external data like JFT-300M dataset/ImageNet-21k for pre-training.
- LeViT authors released 5 trained LeViT models that plug directly into [`LevitModel`] or [`LevitForImageClassification`]. Techniques like data augmentation, optimization, and regularization simulate training on a much larger dataset while only using ImageNet-1k for pre-training.
- [`LevitForImageClassificationWithTeacher`] currently supports only inference, not training or fine-tuning.
- Check out demo [notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) for inference and fine-tuning on custom data. Replace [`ViTImageProcessor`] with [`LevitImageProcessor`] and [`ViTForImageClassification`] with [`LevitForImageClassification`] or [`LevitForImageClassificationWithTeacher`].

## LevitConfig

[[autodoc]] LevitConfig

## LevitImageProcessor

  [[autodoc]] LevitImageProcessor
    - preprocess

## LevitImageProcessorFast

[[autodoc]] LevitImageProcessorFast
    - preprocess

## LevitModel

[[autodoc]] LevitModel
    - forward

## LevitForImageClassification

[[autodoc]] LevitForImageClassification
    - forward

## LevitForImageClassificationWithTeacher

[[autodoc]] LevitForImageClassificationWithTeacher
    - forward

