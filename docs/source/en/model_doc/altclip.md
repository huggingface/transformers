<!--
Copyright 2022 The HuggingFace Team
SPDX-License-Identifier: Apache-2.0
-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <!-- example badge -->
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# altCLIP

[altCLIP](https://arxiv.org/abs/2211.06679v2) swaps CLIP’s original English text encoder for a **multilingual XLM-R** encoder and then realigns image and text representations with a two-stage (teacher → contrastive) schedule.  
Result: you get CLIP-level zero-shot performance in English **plus** strong retrieval in many other languages — without retraining a separate model for each locale 🎉.

You can find all the original **altCLIP** checkpoints under the [altCLIP](https://huggingface.co/models?search=AltCLIP) collection.

> [!TIP]
> Click on the *altCLIP* models in the right sidebar for more examples of how to apply altCLIP to different **image-text retrieval** tasks.

The examples below show how to get similarity scores between an image and one or more captions with the [`AutoModel`] class (a pipeline is not yet registered for altCLIP).

<hfoptions id="usage">

<hfoption id="Pipeline">

`pipeline()` isn’t available because altCLIP currently exposes only its contrastive head (image × text similarity) and no dedicated *visual-question-answering* or *captioning* head.  
Use the AutoModel path shown below instead.

</hfoption>

<hfoption id="AutoModel">

```python
from PIL import Image
import requests
from transformers import AltCLIPProcessor, AltCLIPModel

model_id  = "BAAI/AltCLIP"
processor = AltCLIPProcessor.from_pretrained(model_id)
model     = AltCLIPModel.from_pretrained(model_id)

url   = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat", "a photo of a dog"]

inputs  = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)
print(probs)   # tensor([[0.9996, 0.0004]])
```

</hfoption>

<hfoption id="transformers-cli">

altCLIP does **not** require `transformers-cli` at inference time, but the tool is handy for quantisation (see next section).

</hfoption>

</hfoptions>

---

## Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision.  
Refer to the [Quantization](../quantization/overview) overview for more available back-ends.

The example below uses **[dynamic INT-8 quantization](../quantization/overview#dynamic-quantization)** to only
quantize the *Linear* and *Embedding* weights to **8-bit integers** while keeping activations in FP32.

```bash
# one-liner via transformers-cli
transformers-cli quantize BAAI/AltCLIP \
  --method dynamic \
  --dtype int8 \
  --modules Linear Embedding \
  --output AltCLIP-int8
```

Or do it programmatically in PyTorch:

```python
from transformers import AltCLIPModel
from torch.quantization import quantize_dynamic
import torch, psutil, os

def mb() -> float:                      # helper to report RAM usage
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Resident RAM before load: {mb():.0f} MB")

model_fp32 = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
print(f"After FP32 load: {mb():.0f} MB")

model_int8 = quantize_dynamic(
    model_fp32, {torch.nn.Linear, torch.nn.Embedding}, dtype=torch.qint8
)
print(f"After INT-8 quantization: {mb():.0f} MB")
```

Expect roughly a **4× reduction** in GPU/CPU memory for the quantized
checkpoint while inference accuracy remains virtually unchanged.


---

## Attention visualisation

[AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/attention_visualizer.py) is **NOT** compatible yet with encoder models. It may be used **once encoder support lands** (current release only handles decoder-style LLMs).

In the meantime you can manually retrieve ViT self-attention:

```py
outputs = model(**inputs, output_attentions=True)
attn = outputs.vision_model_output.attentions  # list[num_layers] of (B,H,S,S)
```


## Notes

- altCLIP’s ViT backbone uses **16 × 16 pixel patches**, so a 224 × 224 image becomes a 14 × 14 patch grid.  
  When you reshape attention scores remember to use that grid size.

```py
grid_len = int((attn[-1].size(-1) - 1) ** 0.5)  # 14 for ViT-B/16
heatmap  = attn[-1][0, 0, 1:, 0].reshape(grid_len, grid_len)
```
