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
*This model was released on 2022-03-23 and added to Hugging Face Transformers on 2022-08-04.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# VideoMAE

[VideoMAE](https://huggingface.co/papers/2203.12602) is a self-supervised video representation learning model that extends [Masked Autoencoders(MAE)](vit_mae) to video inputs. It learns by randomly masking a large portion of video patches (typically 90%–95%) and reconstructing the missing parts, making it highly data-efficient. Without using any external data, VideoMAE achieves competitive performance across popular video classification benchmarks. Its simple design, strong results, and ability to work with limited labeled data make it a practical choice for video understanding tasks.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/videomae_architecture.jpeg"
alt="drawing" width="600"/>

You can find all the original VideoMAE checkpoints under the [MCG-NJU](https://huggingface.co/MCG-NJU/models) organization.

> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on the VideoMAE models in the right sidebar for more examples of how to apply VideoMAE to vision tasks.

The example below demonstrates how to perform video classification with [`pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline
from huggingface_hub import list_repo_files, hf_hub_download

pipeline = pipeline(
    task="video-classification",
    model="MCG-NJU/videomae-base-finetuned-kinetics"
)

files = list_repo_files("nateraw/kinetics-mini", repo_type="dataset")
videos = [f for f in files if f.endswith(".mp4")]
video_path = hf_hub_download("nateraw/kinetics-mini", repo_type="dataset", filename=videos[0])

preds = pipeline(video_path)
print(preds)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from huggingface_hub import list_repo_files, hf_hub_download
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModelForVideoClassification

files = list_repo_files("nateraw/kinetics-mini", repo_type="dataset")
videos = [f for f in files if f.endswith(".mp4")]
video_path = hf_hub_download("nateraw/kinetics-mini", repo_type="dataset", filename=videos[0])

video, _, _ = read_video(video_path, pts_unit="sec")
T = video.shape[0]
indices = torch.linspace(0, T - 1, steps=16).long().tolist()
frames = [to_pil_image(video[i].permute(2, 0, 1)) for i in indices]

processor = AutoProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = processor(frames, return_tensors="pt").to(device)
model.to(device)

with torch.no_grad():
    logits = model(**inputs).logits

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
    "MCG-NJU/videomae-base-finetuned-kinetics",
    load_in_8bit=True, 
    device_map="auto"
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
from transformers import VideoMAEForVideoClassification
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", attn_implementation="sdpa", dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `MCG-NJU/videomae-base-finetuned-kinetics` model, we saw the following speedups during inference.

|   Batch size |   Average inference time (ms), eager mode |   Average inference time (ms), sdpa model |   Speed up, Sdpa / Eager (x) |
|--------------|-------------------------------------------|-------------------------------------------|------------------------------|
|            1 |                                        37 |                                        10 |                      3.7  |
|            2 |                                        24 |                                        18 |                      1.33 |
|            4 |                                        43 |                                        32 |                      1.34 |
|            8 |                                        84 |                                        60 |                      1.4  |

## Notes

- Pre-training uses heavy masking (90–95%) on video patches.
- Fine-tuning checkpoints are available for datasets like Kinetics-400, UCF101, and Something-Something-V2.
- For custom datasets, you can follow the [Video classification](../tasks/video_classification) task guide.

   ```py
   # Example: feature extraction + forward pass
    from transformers import VideoMAEFeatureExtractor, VideoMAEModel

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    video = ...  # load frames
    inputs = feature_extractor(list(video), return_tensors="pt")
    outputs = model(**inputs)
    ``` 
   
## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with VideoMAE. If
you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll
review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

**Video classification**
- [A notebook](https://github.com/huggingface/notebooks/blob/main/examples/video_classification.ipynb) that shows how
to fine-tune a VideoMAE model on a custom dataset.
- [Video classification task guide](../tasks/video_classification)
- [A 🤗 Space](https://huggingface.co/spaces/sayakpaul/video-classification-ucf101-subset) showing how to perform inference with a video classification model.

## VideoMAEConfig

[[autodoc]] VideoMAEConfig

## VideoMAEFeatureExtractor

[[autodoc]] VideoMAEFeatureExtractor
    - __call__

## VideoMAEImageProcessor

[[autodoc]] VideoMAEImageProcessor
    - preprocess

## VideoMAEModel

[[autodoc]] VideoMAEModel
    - forward

## VideoMAEForPreTraining

`VideoMAEForPreTraining` includes the decoder on top for self-supervised pre-training.

[[autodoc]] transformers.VideoMAEForPreTraining
    - forward

## VideoMAEForVideoClassification

[[autodoc]] transformers.VideoMAEForVideoClassification
    - forward
