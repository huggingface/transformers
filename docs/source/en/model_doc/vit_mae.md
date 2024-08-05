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

# ViTMAE

## Overview

The ViTMAE model was proposed in [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377v2) by Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li,
Piotr Dollár, Ross Girshick. The paper shows that, by pre-training a Vision Transformer (ViT) to reconstruct pixel values for masked patches, one can get results after
fine-tuning that outperform supervised pre-training.

The abstract from the paper is the following:

*This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the
input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates
only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask
tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs
enables us to train large models efficiently and effectively: we accelerate training (by 3x or more) and improve accuracy. Our scalable approach allows for learning high-capacity
models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream
tasks outperforms supervised pre-training and shows promising scaling behavior.*

<img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png"
alt="drawing" width="600"/> 

<small> MAE architecture. Taken from the <a href="https://arxiv.org/abs/2111.06377">original paper.</a> </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr). TensorFlow version of the model was contributed by [sayakpaul](https://github.com/sayakpaul) and 
[ariG23498](https://github.com/ariG23498) (equal contribution). The original code can be found [here](https://github.com/facebookresearch/mae). 

## Usage tips

- MAE (masked auto encoding) is a method for self-supervised pre-training of Vision Transformers (ViTs). The pre-training objective is relatively simple:
by masking a large portion (75%) of the image patches, the model must reconstruct raw pixel values. One can use [`ViTMAEForPreTraining`] for this purpose.
- After pre-training, one "throws away" the decoder used to reconstruct pixels, and one uses the encoder for fine-tuning/linear probing. This means that after
fine-tuning, one can directly plug in the weights into a [`ViTForImageClassification`].
- One can use [`ViTImageProcessor`] to prepare images for the model. See the code examples for more info.
- Note that the encoder of MAE is only used to encode the visual patches. The encoded patches are then concatenated with mask tokens, which the decoder (which also
consists of Transformer blocks) takes as input. Each mask token is a shared, learned vector that indicates the presence of a missing patch to be predicted. Fixed
sin/cos position embeddings are added both to the input of the encoder and the decoder.
- For a visual understanding of how MAEs work you can check out this [post](https://keras.io/examples/vision/masked_image_modeling/).

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```
from transformers import ViTMAEModel
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base", attn_implementation="sdpa", torch_dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `facebook/vit-mae-base` model, we saw the following speedups during inference.

|   Batch size |   Average inference time (ms), eager mode |   Average inference time (ms), sdpa model |   Speed up, Sdpa / Eager (x) |
|--------------|-------------------------------------------|-------------------------------------------|------------------------------|
|            1 |                                        11 |                                         6 |                      1.83 |
|            2 |                                         8 |                                         6 |                      1.33 |
|            4 |                                         8 |                                         6 |                      1.33 |
|            8 |                                         8 |                                         6 |                      1.33 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViTMAE.

- [`ViTMAEForPreTraining`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining), allowing you to pre-train the model from scratch/further pre-train the model on custom data.
- A notebook that illustrates how to visualize reconstructed pixel values with [`ViTMAEForPreTraining`] can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb).

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ViTMAEConfig

[[autodoc]] ViTMAEConfig

<frameworkcontent>
<pt>

## ViTMAEModel

[[autodoc]] ViTMAEModel
    - forward

## ViTMAEForPreTraining

[[autodoc]] transformers.ViTMAEForPreTraining
    - forward

</pt>
<tf>

## TFViTMAEModel

[[autodoc]] TFViTMAEModel
    - call

## TFViTMAEForPreTraining

[[autodoc]] transformers.TFViTMAEForPreTraining
    - call

</tf>
</frameworkcontent>
