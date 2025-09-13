<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2021-03-29 and added to Hugging Face Transformers on 2023-07-11.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>


# Video Vision Transformer (ViViT)

[ViViT](https://huggingface.co/papers/2103.15691) is one of the first successful attempts to bring pure Transformer architectures to the video understanding domain. Inspired by the success of Vision Transformers (ViTs) in image classification, ViViT applies similar principles to video inputs by extracting spatio-temporal tokens from sequences of frames. What makes ViViT unique is its modular and scalable design: instead of applying heavy 3D convolutions, it factorizes the spatial and temporal dimensions and processes them separately using Transformer layers. This not only reduces computational cost, but also allows ViViT to leverage pretrained image models effectively.

You can find all the original ViViT checkpoints under the [Google](https://huggingface.co/google/models?search=vivit) organization.

> [!TIP]
> This model was contributed by [jegormeister](https://huggingface.co/jegormeister).
>
> Click on the ViViT models in the right sidebar for more examples of how to apply ViViT to different vision tasks.

The example below demonstrates how to run video classification with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline>

```python
from transformers import pipeline, VivitImageProcessor
from huggingface_hub import list_repo_files, hf_hub_download

image_processor = VivitImageProcessor.from_pretrained(model_name)

video_cls_pipeline = pipeline(
    task="video-classification",
    model="google/vivit-b-16x2-kinetics400",
    image_processor=image_processor,
)

files = list_repo_files("nateraw/kinetics-mini", repo_type="dataset")
videos = [f for f in files if f.endswith(".mp4")]
video_path = hf_hub_download("nateraw/kinetics-mini", repo_type="dataset", filename=videos[0])

preds_pipeline = video_cls_pipeline(video_path)
print(preds_pipeline)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from huggingface_hub import list_repo_files, hf_hub_download
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForVideoClassification, VivitImageProcessor 
import numpy as np 

files = list_repo_files("nateraw/kinetics-mini", repo_type="dataset")
videos = [f for f in files if f.endswith(".mp4")]
video_path = hf_hub_download("nateraw/kinetics-mini", repo_type="dataset", filename=videos[0])
video, _, _ = read_video(video_path, pts_unit="sec")

processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = AutoModelForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400").eval()

num_frames = 32
target_size = (224, 224) 

T = video.shape[0]
indices = np.linspace(0, T - 1, num_frames, dtype=int)
frames = [to_pil_image(video[i].permute(2, 0, 1)).resize(target_size) for i in indices] 

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = processor(frames, return_tensors="pt").to(device)
model.to(device)

with torch.no_grad():
    logits = model(pixel_values=inputs['pixel_values']).logits

probs = logits.softmax(-1)[0]
topk = probs.topk(5)

id2label = model.config.id2label
print([{ "label": id2label[i.item()], "score": probs[i].item() } for i in topk.indices])
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [BitsAndBytes](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) to quantize the weights to 8-bit precision:

```python
from transformers import AutoModelForVideoClassification

model = AutoModelForVideoClassification.from_pretrained(
    "google/vivit-b-16x2-kinetics400",
    load_in_8bit=True,
    device_map="auto",
).eval()
```

## Using Scaled Dot Product Attention (SDPA)

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

## Notes

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
