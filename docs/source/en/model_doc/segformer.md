<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

⚠️ Note that this file is in Markdown but contains specific syntax
for our doc-builder (similar to MDX) that may not render properly
in your Markdown viewer.
-->
*This model was released on 2021-05-31 and added to Hugging Face Transformers on 2021-10-28.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SegFormer

[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://huggingface.co/papers/2105.15203) is a semantic segmentation model that combines a hierarchical Transformer encoder (Mix Transformer, MiT) with a lightweight all-MLP decoder. It avoids positional encodings and complex decoders and achieves state-of-the-art performance on benchmarks like ADE20K and Cityscapes. This simple and lightweight design is more efficient and scalable.

You can find all the original SegFormer checkpoints under the [NVIDIA](https://huggingface.co/nvidia/models?search=segformer) organization.

> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on the SegFormer models in the right sidebar for more examples of how to apply SegFormer to different vision tasks.

The example below demonstrates semantic segmentation with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

segmenter = pipeline("semantic-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")
image = "your_image.png"
outputs = segmenter(image)
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # shape [batch, num_labels, height, width]
```

</hfoption>

</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) to quantize SegFormer weights to 8-bit.

```python
from transformers import BitsAndBytesConfig, AutoModelForSemanticSegmentation

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    quantization_config=quantization_config
)
```

## Notes

- SegFormer works with **any input size**, padding inputs to be divisible by `config.patch_sizes`.
- The most important preprocessing step is to randomly crop and pad all images to the same size (such as 512x512 or 640x640) and normalize afterwards.
- When preprocessing, be mindful of `do_reduce_labels`:
  - Some datasets (like ADE20K) don’t include background in the labels → set `do_reduce_labels=True`.
  - Other datasets do include background → set `do_reduce_labels=False`.
- Model variants differ in size and accuracy (MiT-B0 to MiT-B5). Example:
  - **SegFormer-B4**: 50.3% mIoU on ADE20K with 64M parameters, 5x smaller and better than prior methods.
  - **SegFormer-B5**: 84.0% mIoU on Cityscapes validation set, strong zero-shot robustness.

```python
from transformers import SegformerImageProcessor
processor = SegformerImageProcessor(do_reduce_labels=True)
```

## Resources

- [Original SegFormer code (NVlabs)](https://github.com/NVlabs/SegFormer)  
- [Fine-tuning blog post](https://huggingface.co/blog/fine-tune-segformer)  
- [Tutorial notebooks (Niels Rogge)](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SegFormer)  
- [Hugging Face demo space](https://huggingface.co/spaces/chansung/segformer-tf-transformers)  

## SegformerConfig

[[autodoc]] SegformerConfig

## SegformerFeatureExtractor

[[autodoc]] SegformerFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## SegformerImageProcessor

[[autodoc]] SegformerImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## SegformerImageProcessorFast

[[autodoc]] SegformerImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation

## SegformerModel

[[autodoc]] SegformerModel
    - forward

## SegformerDecodeHead

[[autodoc]] SegformerDecodeHead
    - forward

## SegformerForImageClassification

[[autodoc]] SegformerForImageClassification
    - forward

## SegformerForSemanticSegmentation

[[autodoc]] SegformerForSemanticSegmentation
    - forward
