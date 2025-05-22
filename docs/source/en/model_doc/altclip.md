<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ This file is Markdown that contains MDX-style syntax for the HF doc-builder; it may not render perfectly in a plain
viewer.
-->

# altCLIP Vision–Language Model

<div class="flex flex-wrap space-x-1">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**altCLIP** swaps CLIP’s text encoder for a multilingual XLM-R encoder and then aligns vision and language
representations with a teacher-then-contrastive schedule.  
It achieves near-SOTA zero-shot retrieval across languages while preserving CLIP’s English performance.

| Dataset | LAION-400M |
|---------|-----------:|
| Metrics | Image↔Text Retrieval (Recall @ 1 / 5 / 10) |
| Paper   | [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v2) |

---

## Ready-to-use code example (AutoModel)

```python
from PIL import Image
import requests
from transformers import AltCLIPProcessor, AltCLIPModel

# 1 – load model & processor
model_id  = "BAAI/AltCLIP"
processor = AltCLIPProcessor.from_pretrained(model_id)
model     = AltCLIPModel.from_pretrained(model_id)

# 2 – prepare inputs
url   = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat", "a photo of a dog"]

inputs  = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# 3 – similarity scores
logits_per_image = outputs.logits_per_image
probs            = logits_per_image.softmax(dim=1)
print(probs)                 # → tensor([[0.96, 0.04]], grad_fn=<SoftmaxBackward>)
```

<Tip>

`AltCLIPModel` is a subclass of `CLIPModel`, so you can drop it into any code that already expects CLIP.

</Tip>

---

## Quantization example (large model)

```bash
# Dynamic INT-8 quantization with transformers-cli
transformers-cli quantize BAAI/AltCLIP \
  --method dynamic \
  --dtype int8 \
  --output AltCLIP-int8
```

---

## Attention visualisation

```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

viz = AttentionMaskVisualizer("BAAI/AltCLIP")
viz("<img>Identify the main object in this image")
```

---

## API reference

[[autodoc]] AltCLIPConfig - from_text_vision_configs  
[[autodoc]] AltCLIPTextConfig  
[[autodoc]] AltCLIPVisionConfig  
[[autodoc]] AltCLIPProcessor  
[[autodoc]] AltCLIPModel - forward - get_text_features - get_image_features  
[[autodoc]] AltCLIPTextModel - forward  
[[autodoc]] AltCLIPVisionModel - forward  

---

## Citation

```bibtex
@inproceedings{chen2022altclip,
  title     = {AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities},
  author    = {Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong
               and Yang, Qinghong and Wu, Ledell},
  booktitle = {arXiv preprint arXiv:2211.06679v2},
  year      = {2022}
}
```
