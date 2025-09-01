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

# ConvNeXt V2

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

[ConvNeXt V2](https://huggingface.co/papers/2301.00808) is a pure convolutional model (ConvNet) that combines masked autoencoders with architectural improvements. It's like the original ConvNeXt but with better self-supervised learning capabilities and a new Global Response Normalization layer that makes the model more competitive.

You can find all the original [ConvNeXt V2](https://huggingface.co/models?other=convnextv2) checkpoints under the ConvNeXt V2 collection.

> [!TIP]
> This model was contributed by [adirik](https://huggingface.co/adirik).
>
> Click on the **ConvNeXt V2** models in the right sidebar for more examples of how to apply ConvNeXt V2 to different **image-classification** tasks.

The example below demonstrates how to perform image classification with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">

<hfoption id="Pipeline">

```python
from transformers import pipeline
from PIL import Image

clf = pipeline("image-classification", model="facebook/convnextv2-tiny-1k-224")
img = Image.open("cat.jpg")
print(clf(img)[:3])  # top-3 predictions
```

</hfoption>

<hfoption id="AutoModel">

```python
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification, ConvNextV2Model
from PIL import Image
import torch

model_id = "facebook/convnextv2-tiny-1k-224"

# Classification head
processor = AutoImageProcessor.from_pretrained(model_id)
model = ConvNextV2ForImageClassification.from_pretrained(model_id)

img = Image.open("cat.jpg")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
pred = logits.argmax(-1).item()
print(model.config.id2label[pred])

# Backbone features
backbone = ConvNextV2Model.from_pretrained(model_id)
with torch.no_grad():
    feats = backbone(**inputs).last_hidden_state  # (B, C, H, W)
print(feats.shape)
```

</hfoption>

<hfoption id="transformers-cli">

```bash
# Image classification with transformers-cli
transformers-cli run --model facebook/convnextv2-tiny-1k-224 --task image-classification --input cat.jpg
```

</hfoption>

</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [BitsAndBytesConfig](https://huggingface.co/docs/transformers/quantization#bitsandbytes) to quantize the weights to 8-bit.

```python
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification, BitsAndBytesConfig

model_id = "facebook/convnextv2-tiny-1k-224"
bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)  # or load_in_4bit=True

processor = AutoImageProcessor.from_pretrained(model_id)
model = ConvNextV2ForImageClassification.from_pretrained(
    model_id, quantization_config=bnb_cfg, device_map="auto"
)
```

## Overview

The ConvNeXt V2 model was proposed in [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://huggingface.co/papers/2301.00808) by Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
ConvNeXt V2 is a pure convolutional model (ConvNet), inspired by the design of Vision Transformers, and a successor of [ConvNeXT](convnext).

The abstract from the paper is the following:

*Driven by improved architectures and better representation learning frameworks, the field of visual recognition has enjoyed rapid modernization and performance boost in the early 2020s. For example, modern ConvNets, represented by ConvNeXt, have demonstrated strong performance in various scenarios. While these models were originally designed for supervised learning with ImageNet labels, they can also potentially benefit from self-supervised learning techniques such as masked autoencoders (MAE). However, we found that simply combining these two approaches leads to subpar performance. In this paper, we propose a fully convolutional masked autoencoder framework and a new Global Response Normalization (GRN) layer that can be added to the ConvNeXt architecture to enhance inter-channel feature competition. This co-design of self-supervised learning techniques and architectural improvement results in a new model family called ConvNeXt V2, which significantly improves the performance of pure ConvNets on various recognition benchmarks, including ImageNet classification, COCO detection, and ADE20K segmentation. We also provide pre-trained ConvNeXt V2 models of various sizes, ranging from an efficient 3.7M-parameter Atto model with 76.7% top-1 accuracy on ImageNet, to a 650M Huge model that achieves a state-of-the-art 88.9% accuracy using only public training data.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnextv2_architecture.png"
alt="drawing" width="600"/>

<small> ConvNeXt V2 architecture. Taken from the <a href="https://huggingface.co/papers/2301.00808">original paper</a>.</small>

This model was contributed by [adirik](https://huggingface.co/adirik). The original code can be found [here](https://github.com/facebookresearch/ConvNeXt-V2).

## Intended uses & limitations

**Use for**
- Image classification out of the box (ImageNet-1k/22k fine-tuned checkpoints).
- As a **backbone** to extract multi-scale feature maps for detection/segmentation tasks.

**Limitations / caveats**
- Most layers are `Conv2d`. Quantization methods that only target linear layers (e.g. 8/4-bit with bitsandbytes) will primarily affect the classification head and yield modest memory savings compared to transformer LLMs.
- Accuracy is sensitive to input resolution and preprocessing. Match your evaluation transforms to the checkpoint's training recipe (e.g., 224 vs 384).

## Preprocessing

Use `AutoImageProcessor` (which selects the correct `ConvNextImageProcessor`) to apply the expected resizing/cropping and mean/std normalization that match the pretrained checkpoint's recipe. Keep the input resolution consistent with the checkpoint (e.g., 224 or 384).

## Model variants (examples)

Common checkpoints on the Hub follow this pattern: `facebook/convnextv2-<size>-<1k|22k>-<224|384>`

* `facebook/convnextv2-atto-1k-224`, `facebook/convnextv2-femto-1k-224`, `facebook/convnextv2-pico-1k-224`
* `facebook/convnextv2-nano-1k-224`, `facebook/convnextv2-tiny-1k-224`, `facebook/convnextv2-small-1k-224`
* `facebook/convnextv2-base-22k-224`, `facebook/convnextv2-base-22k-384`
* Larger variants (e.g., large/huge) may require more memory.

## Training & evaluation notes

* Align augmentation and resolution with the chosen checkpoint (224 or 384).
* Mixed precision (`float16`/`bfloat16`) is recommended on GPU.
* When using as a backbone, adjust `out_indices` to expose multi-scale features for downstream heads.

## Notes

- Since ConvNeXt V2 is mostly composed of `Conv2d` layers, quantization methods that only target linear layers will primarily affect the classification head and yield modest memory savings.

   ```python
   # Example: Using quantization for memory efficiency
   from transformers import BitsAndBytesConfig
   
   quantization_config = BitsAndBytesConfig(
       load_in_8bit=True,
       llm_int8_threshold=6.0
   )
   ```

- When using as a backbone, adjust `out_indices` to expose multi-scale features for downstream heads.

   ```python
   # Example: Using ConvNeXt V2 as a backbone
   backbone = ConvNextV2Model.from_pretrained(
       "facebook/convnextv2-tiny-1k-224",
       out_indices=[0, 1, 2, 3]  # Get features from all stages
   )
   ```

## Resources

* [`ConvNextV2ForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
* [ConvNeXt V2 paper](https://huggingface.co/papers/2301.00808)
* [Original implementation](https://github.com/facebookresearch/ConvNeXt-V2)

## Citation

If you use ConvNeXt V2, please cite:

```bibtex
@article{woo2023convnextv2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and Liu, Zhuang and Kweon, In So and Xie, Saining},
  journal={arXiv preprint arXiv:2301.00808},
  year={2023}
}
```

## ConvNextV2Config

[[autodoc]] ConvNextV2Config

## ConvNextV2Model

[[autodoc]] ConvNextV2Model
- forward

## ConvNextV2ForImageClassification

[[autodoc]] ConvNextV2ForImageClassification
- forward