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
*This model was released on 2021-02-09 and added to Hugging Face Transformers on 2022-12-02.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# TimeSformer

[TimeSformer](https://huggingface.co/papers/2102.05095) is a convolution-free video transformer model designed to classify actions in video clips using only self-attention mechanisms over space and time. Inspired by the success of Transformers in NLP and vision, TimeSformer treats a video as a sequence of frame-level patches and applies attention across both spatial and temporal dimensions. The key innovation is the use of divided attention, where temporal attention and spatial attention are applied separately within each transformer block. This design leads to better performance on video understanding benchmarks by allowing the model to learn motion and appearance features more effectively.

You can find all the original TimeSformer checkpoints under the [Facebook](https://huggingface.co/facebook/models?search=timesformer) organization.

> [!TIP]
> This model was contributed by [fcakyon](https://huggingface.co/fcakyon).
>
> Click on the TimeSformer models in the right sidebar for more examples of how to apply TimeSformer to different vision tasks.

The example below demonstrates how to run video classification with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline>

```python
import torch
from transformers import pipeline
from huggingface_hub import list_repo_files, hf_hub_download

pipeline = pipeline(
    task="video-classification",
    model="facebook/timesformer-base-finetuned-k400",
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
import numpy as np

processor = AutoProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400").eval()

files = list_repo_files("nateraw/kinetics-mini", repo_type="dataset")
videos = [f for f in files if f.endswith(".mp4")]
video_path = hf_hub_download("nateraw/kinetics-mini", repo_type="dataset", filename=videos[0])
video, _, _ = read_video(video_path, pts_unit="sec")

num_frames = processor.num_frames if hasattr(processor, 'num_frames') else 8 
target_size = (processor.size["height"], processor.size["width"]) if hasattr(processor, 'size') and 'height' in processor.size and 'width' in processor.size else (224, 224)

T = video.shape[0]
indices = np.linspace(0, T - 1, num_frames, dtype=int)
frames = [to_pil_image(video[i].permute(2, 0, 1)).resize(target_size) for i in indices]

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
    "facebook/timesformer-base-finetuned-k400",
    load_in_8bit=True,
    device_map="auto",
).eval()
```

## Notes

There are many pretrained variants. Select your pretrained model based on the dataset it is trained on. Moreover, the number of input frames per clip changes based on the model size so you should consider this parameter while selecting your pretrained model.

## Resources

- [Video classification task guide](../tasks/video_classification)

## TimesformerConfig

[[autodoc]] TimesformerConfig

## TimesformerModel

[[autodoc]] TimesformerModel
    - forward

## TimesformerForVideoClassification

[[autodoc]] TimesformerForVideoClassification
    - forward