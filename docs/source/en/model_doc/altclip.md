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

> ### Why not use `pipeline()`?
>
> The 🤗 `pipeline()` helper offers a one-liner for many architectures,  
> but it only works when a *task-specific head* is officially registered.  
> AltCLIP currently exposes **only its original contrastive head**
> (image × text similarity) and no `VisualQuestionAnswering` head, so the
> generic VQA pipeline rejects it with  
> “*Model \<name\> is not supported*.”
>
> **AutoModel + Processor** gives you full control:
>
> 1. Load the checkpoint (`AltCLIPModel` & `AltCLIPProcessor`).
> 2. Feed any image–text pair(s) in one forward pass.
> 3. Post-process `logits_per_image` however you like (softmax, ranking …).
>
> ```python
> from PIL import Image
> import requests
> from transformers import AltCLIPProcessor, AltCLIPModel
>
> model_id  = "BAAI/AltCLIP"
> processor = AltCLIPProcessor.from_pretrained(model_id)
> model     = AltCLIPModel.from_pretrained(model_id)
>
> # sample data
> url   = "http://images.cocodataset.org/val2017/000000039769.jpg"
> image = Image.open(requests.get(url, stream=True).raw)
> texts = ["a photo of a cat", "a photo of a dog"]
>
> inputs  = processor(text=texts, images=image, return_tensors="pt", padding=True)
> outputs = model(**inputs)
>
> probs = outputs.logits_per_image.softmax(dim=1)
> print(probs)      # tensor([[0.9996, 0.0004]])
> ```
>
> The first caption (“cat”) wins with > 99 % confidence, exactly what we expect
> for that COCO image.

<Tip>

`AltCLIPModel` is a subclass of `CLIPModel`, so it drops straight into any code that already uses CLIP.

</Tip>

---

## Quantization example (large model)

```bash
transformers-cli quantize BAAI/AltCLIP \
  --method dynamic \
  --dtype int8 \
  --output AltCLIP-int8
```

---

## Attention visualisation  

> `AttentionMaskVisualizer` does **not yet support encoder-only models**
> like AltCLIP (see [issue #25096](https://github.com/huggingface/transformers/issues/25096));
> it errors out with  
> `_ModelWrapper has no attribute _update_causal_mask`.
>
> Until that lands you can still **retrieve and plot raw ViT self-attention**
> in ~15 lines:

```python
from PIL import Image
import requests, torch, matplotlib.pyplot as plt
from transformers import AltCLIPProcessor, AltCLIPModel

# 1 – load model
model_id  = "BAAI/AltCLIP"
processor = AltCLIPProcessor.from_pretrained(model_id)
model     = AltCLIPModel.from_pretrained(model_id).eval()

# 2 – prepare a sample image + text
url   = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text="a photo of a cat", images=image,
                   return_tensors="pt", padding=True)

# 3 – forward pass with attentions
with torch.no_grad():
    out = model(**inputs, output_attentions=True)

att = out.vision_model_output.attentions      # list[num_layers] · (B,H,S,S)

# 4 – pick last layer, head 0 → attention from every patch to CLS token
layer, head = -1, 0
grid_len    = int((att[layer].size(-1) - 1) ** 0.5)    # 14 for ViT-B/16
heatmap     = att[layer][0, head, 1:, 0].reshape(grid_len, grid_len)

plt.imshow(heatmap.cpu(), interpolation="nearest")
plt.title("AltCLIP · CLS attention (last layer, head 0)")
plt.axis("off")
plt.show()
```

This produces a 14 × 14 heat-map (for ViT-B/16) showing which image regions
attend most to the global CLS token.

---

## API reference

[[autodoc]] AltCLIPConfig — from_text_vision_configs  
[[autodoc]] AltCLIPTextConfig  
[[autodoc]] AltCLIPVisionConfig  
[[autodoc]] AltCLIPProcessor  
[[autodoc]] AltCLIPModel — forward · get_text_features · get_image_features  
[[autodoc]] AltCLIPTextModel — forward  
[[autodoc]] AltCLIPVisionModel — forward  

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
