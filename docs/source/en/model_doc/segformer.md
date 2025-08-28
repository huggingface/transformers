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

The figure below illustrates the architecture of SegFormer.

<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/segformer_architecture.png"/>

You can find all the original SegFormer checkpoints under the [NVIDIA](https://huggingface.co/nvidia/models?search=segformer) organization.

> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on the SegFormer models in the right sidebar for more examples of how to apply SegFormer to different vision tasks.

The example below demonstrates semantic segmentation with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(task="image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512", torch_dtype=torch.float16)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForSemanticSegmentation

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits # shape [batch, num_labels, height, width]
```

</hfoption>

</hfoptions>



## Notes

- SegFormer works with **any input size**, padding inputs to be divisible by `config.patch_sizes`.
- The most important preprocessing step is to randomly crop and pad all images to the same size (such as 512x512 or 640x640) and normalize afterwards.
- Some datasets (ADE20k) uses the `0` index in the annotated segmentation as the background, but doesn't include the "background" class in its labels. The `do_reduce_labels` argument in [`SegformerForImageProcessor`] is used to reduce all labels by `1`. To make sure no loss is computed for the background class, it replaces `0` in the annotated maps by `255`, which is the `ignore_index` of the loss function.

   Other datasets may include a background class and label though, in which case, `do_reduce_labels` should be `False`.

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
