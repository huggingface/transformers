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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
            <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
            <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
            <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SigLIP

[SigLIP](https://huggingface.co/papers/2303.15343) is a multimodal image and text model similar to [CLIP](clip), It uses separate image and text encoders to generate representations for both modalities.

Unlike CLIP, SigLIP employs a pairwise sigmoid loss on image-text pairs during training. This training loss eliminates the need for a global view of all pairwise similarities between images and texts within a batch. Consequently, it enables more efficient scaling to larger batch sizes while also delivering superior performance with smaller batch sizes.

You can find all the original SigLIP checkpoints under the [SigLIP](https://huggingface.co/collections/google/siglip-659d5e62f0ae1a57ae0e83ba) collection.


> [!TIP]
> Click on the SigLIP models in the right sidebar for more examples of how to apply SigLIP to different image and text tasks.

The example below demonstrates how to generate similarity scores between texts and image(s) with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline
from PIL import Image
import requests

# load pipe
image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-224")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
candidate_labels = ["2 cats", "a plane", "a remote"]
outputs = image_classifier(image, candidate_labels=candidate_labels)
outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
print(outputs)
[{'score': 0.1979, 'label': '2 cats'}, {'score': 0.0, 'label': 'a remote'}, {'score': 0.0, 'label': 'a plane'}]
```

</hfoption>
<hfoption id="AutoModel">

```py
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

candidate_labels = ["2 cats", "2 dogs"]
# follows the pipeline prompt template to get same results
texts = [f'This is a photo of {label}.' for label in candidate_labels]
# important: we pass `padding=max_length` since the model was trained with this
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
19.8% that image 0 is '2 cats'
```

</hfoption>
</hfoptions>


## Notes

- Training is supported but does not use `torch.distributed` utilities which may limit the scalability of batch size. However, DDP and FDSP works on single-node multi-gpu setup.
- When using the standalone [`SiglipTokenizer`] or [`SiglipProcessor`], make sure to pass `padding="max_length"` as that's how the model was trained.
- To get the same results as the pipeline, a prompt template of "This is a photo of {label}." should be used.
- To explicitly use SDPA or Flash-Attention 2, set `attn_implementation="sdpa"` or `attn_implementation="flash_attention_2"` in the from_pretrained() method. Ensure that your PyTorch version is >=2.1.1.
```python
from transformers import SiglipModel

model = SiglipModel.from_pretrained(
    "google/siglip-so400m-patch14-384",
    attn_implementation="sdpa",
    torch_dtype=torch.float16,
    device_map=device,
)
```
- For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).


## SiglipConfig

[[autodoc]] SiglipConfig
    - from_text_vision_configs

## SiglipTextConfig

[[autodoc]] SiglipTextConfig

## SiglipVisionConfig

[[autodoc]] SiglipVisionConfig

## SiglipTokenizer

[[autodoc]] SiglipTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SiglipImageProcessor

[[autodoc]] SiglipImageProcessor
    - preprocess

## SiglipImageProcessorFast

[[autodoc]] SiglipImageProcessorFast
    - preprocess

## SiglipProcessor

[[autodoc]] SiglipProcessor

## SiglipModel

[[autodoc]] SiglipModel
    - forward
    - get_text_features
    - get_image_features

## SiglipTextModel

[[autodoc]] SiglipTextModel
    - forward

## SiglipVisionModel

[[autodoc]] SiglipVisionModel
    - forward


## SiglipForImageClassification

[[autodoc]] SiglipForImageClassification
    - forward
