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

# ViTMSN

## Overview

The ViTMSN model was proposed in [Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141) by Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes,
Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas. The paper presents a joint-embedding architecture to match the prototypes
of masked patches with that of the unmasked patches. With this setup, their method yields excellent performance in the low-shot and extreme low-shot
regimes.

The abstract from the paper is the following:

*We propose Masked Siamese Networks (MSN), a self-supervised learning framework for learning image representations. Our
approach matches the representation of an image view containing randomly masked patches to the representation of the original
unmasked image. This self-supervised pre-training strategy is particularly scalable when applied to Vision Transformers since only the
unmasked patches are processed by the network. As a result, MSNs improve the scalability of joint-embedding architectures,
while producing representations of a high semantic level that perform competitively on low-shot image classification. For instance,
on ImageNet-1K, with only 5,000 annotated images, our base MSN model achieves 72.4% top-1 accuracy,
and with 1% of ImageNet-1K labels, we achieve 75.7% top-1 accuracy, setting a new state-of-the-art for self-supervised learning on this benchmark.*

<img src="https://i.ibb.co/W6PQMdC/Screenshot-2022-09-13-at-9-08-40-AM.png" alt="drawing" width="600"/> 

<small> MSN architecture. Taken from the <a href="https://arxiv.org/abs/2204.07141">original paper.</a> </small>

This model was contributed by [sayakpaul](https://huggingface.co/sayakpaul). The original code can be found [here](https://github.com/facebookresearch/msn). 

## Usage tips

- MSN (masked siamese networks) is a method for self-supervised pre-training of Vision Transformers (ViTs). The pre-training
objective is to match the prototypes assigned to the unmasked views of the images to that of the masked views of the same images.
- The authors have only released pre-trained weights of the backbone (ImageNet-1k pre-training). So, to use that on your own image classification dataset,
use the [`ViTMSNForImageClassification`] class which is initialized from [`ViTMSNModel`]. Follow
[this notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb) for a detailed tutorial on fine-tuning.
- MSN is particularly useful in the low-shot and extreme low-shot regimes. Notably, it achieves 75.7% top-1 accuracy with only 1% of ImageNet-1K
labels when fine-tuned.

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```
from transformers import ViTMSNForImageClassification
model = ViTMSNForImageClassification.from_pretrained("facebook/vit-msn-base", attn_implementation="sdpa", torch_dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `facebook/vit-msn-base` model, we saw the following speedups during inference.

|   Batch size |   Average inference time (ms), eager mode |   Average inference time (ms), sdpa model |   Speed up, Sdpa / Eager (x) |
|--------------|-------------------------------------------|-------------------------------------------|------------------------------|
|            1 |                                         7 |                                         6 |                      1.17 |
|            2 |                                         8 |                                         6 |                      1.33 |
|            4 |                                         8 |                                         6 |                      1.33 |
|            8 |                                         8 |                                         6 |                      1.33 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViT MSN.

<PipelineTag pipeline="image-classification"/>

- [`ViTMSNForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ViTMSNConfig

[[autodoc]] ViTMSNConfig

## ViTMSNModel

[[autodoc]] ViTMSNModel
    - forward

## ViTMSNForImageClassification

[[autodoc]] ViTMSNForImageClassification
    - forward
