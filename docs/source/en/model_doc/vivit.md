<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Video Vision Transformer (ViViT)

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Vivit model was proposed in [ViViT: A Video Vision Transformer](https://huggingface.co/papers/2103.15691) by Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid.
The paper proposes one of the first successful pure-transformer based set of models for video understanding.

The abstract from the paper is the following:

*We present pure-transformer based models for video classification, drawing upon the recent success of such models in image classification. Our model extracts spatio-temporal tokens from the input video, which are then encoded by a series of transformer layers. In order to handle the long sequences of tokens encountered in video, we propose several, efficient variants of our model which factorise the spatial- and temporal-dimensions of the input. Although transformer-based models are known to only be effective when large training datasets are available, we show how we can effectively regularise the model during training and leverage pretrained image models to be able to train on comparatively small datasets. We conduct thorough ablation studies, and achieve state-of-the-art results on multiple video classification benchmarks including Kinetics 400 and 600, Epic Kitchens, Something-Something v2 and Moments in Time, outperforming prior methods based on deep 3D convolutional networks.*

This model was contributed by [jegormeister](https://huggingface.co/jegormeister). The original code (written in JAX) can be found [here](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit).

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```
from transformers import VivitModel
model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", attn_implementation="sdpa", dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `google/vivit-b-16x2-kinetics400` model, we saw the following speedups during inference.

### Training
|   num_training_steps |   batch_size |   is cuda |   Speedup (%) |   Eager peak mem (MB) |   sdpa peak mem (MB) |   Mem saving (%) |
|---------------------:|-------------:|----------:|--------------:|----------------------:|---------------------:|-----------------:|
|                  100 |            1 |      True |         7.122 |               2575.28 |              5932.54 |           130.364 |



### Inference
|   num_batches |   batch_size |   is cuda |   is half |   Speedup (%) |   Mem eager (MB) |   Mem BT (MB) |   Mem saved (%) |
|---------------|--------------|-----------|-----------|---------------|------------------|---------------|-----------------|
|            20 |             1 |   True    |   False   |      15.422   |     715.807      |    317.079    |      125.75     |
|            20 |             2 |   True    |   False   |      17.146   |    1234.75       |    447.175    |      176.122    |
|            20 |             4 |   True    |   False   |      18.093   |    2275.82       |    709.864    |      220.6      |
|            20 |             8 |   True    |   False   |      19.284   |    4358.19       |   1233.24     |      253.393    |
           

## VivitConfig

[[autodoc]] VivitConfig

## VivitImageProcessor

[[autodoc]] VivitImageProcessor
    - preprocess

## VivitModel

[[autodoc]] VivitModel
    - forward

## VivitForVideoClassification

[[autodoc]] transformers.VivitForVideoClassification
    - forward
