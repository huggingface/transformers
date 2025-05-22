<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ This file is Markdown that contains special MDX-style syntax for Hugging Face’s doc-builder; it may not render
correctly in a generic Markdown viewer.
-->

# altCLIP Vision–Language Model

<div class="flex flex-wrap space-x-1">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**altCLIP** extends OpenAI’s CLIP by swapping its text encoder for a pretrained multilingual encoder (XLM-R) and then
aligning vision and language representations via a two-stage (teacher + contrastive) training scheme.  
The result is near-SOTA zero-shot retrieval across languages and domains while preserving CLIP’s original strengths.

| Dataset used | LAION-400M |
|--------------|-----------:|
| Key metrics  | Image–Text Retrieval (Recall @ 1 / 5 / 10) |
| Paper        | [“AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities”](https://arxiv.org/abs/2211.06679v2) |

---

## Usage examples

> **Tip** – altCLIP works with either the built-in CLIP pipelines **or** the dedicated
> `AltCLIPProcessor` + `AltCLIPModel`.

```python
# Pipeline example (zero-shot VQA)
from transformers import pipeline

clip_vqa = pipeline(
    task="visual-question-answering",
    model="facebook/altclip-multilingual",
    feature_extractor="facebook/altclip-multilingual",
)

answer = clip_vqa({"image": image, "question": "What object is on the table?"})
print(answer)
```

```python
# AutoModel + Processor example (image–text similarity)
from transformers import AltCLIPProcessor, AltCLIPModel

processor = AltCLIPProcessor.from_pretrained("facebook/altclip-multilingual")
model      = AltCLIPModel.from_pretrained("facebook/altclip-multilingual")

inputs  = processor(text=["a cat on a sofa"], images=image,
                    return_tensors="pt", padding=True)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # similarity scores
probs            = logits_per_image.softmax(dim=1)
print(probs)
```

<Tip>

This model inherits from `CLIPModel`, so you can drop it into any workflow that already uses CLIP.

</Tip>

---

## Quantization example

Reduce memory and latency via dynamic INT-8 quantization:

```bash
transformers-cli quantize facebook/altclip-multilingual \
  --method dynamic \
  --dtype int8 \
  --output altclip-multilingual-quantized
```

---

## Attention visualization

Visualise cross-modal attention with `AttentionMaskVisualizer`:

```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

viz = AttentionMaskVisualizer("facebook/altclip-multilingual")
viz("<img>Identify the main object in this image")
```

---

## API reference

[[autodoc]] AltCLIPConfig  
- from_text_vision_configs

[[autodoc]] AltCLIPTextConfig

[[autodoc]] AltCLIPVisionConfig

[[autodoc]] AltCLIPProcessor

[[autodoc]] AltCLIPModel  
- forward  
- get_text_features  
- get_image_features

[[autodoc]] AltCLIPTextModel  
- forward

[[autodoc]] AltCLIPVisionModel  
- forward

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
