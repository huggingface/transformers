<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MLCD

## Overview

The MLCD models were released by the DeepGlint-AI team in [unicom](https://github.com/deepglint/unicom), which focuses on building foundational visual models for large multimodal language models using large-scale datasets such as LAION400M and COYO700M, and employs sample-to-cluster contrastive learning to optimize performance. MLCD models are primarily used for multimodal visual large language models, such as LLaVA.

🔥**MLCD-ViT-bigG**🔥 series is the state-of-the-art vision transformer model enhanced with 2D Rotary Position Embedding (RoPE2D), achieving superior performance on document understanding and visual question answering tasks. Developed by DeepGlint AI, this model demonstrates exceptional capabilities in processing complex visual-language interactions.

Tips:

- We adopted the official [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and the official training dataset [LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) for evaluating the foundational visual models.

- The language model is [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). 

Result: 

| Vision Tower                                                                                  | RoPE2D | ChartQA   | DocVQA    | InfoVQA   | OCRBench   | MMMU      |
| :-------------------------------------------------------------------------------------------- | :----: | :-------- | :-------- | :-------- | :--------- | :-------- |
| CLIP (ViT-L-14-336px)                                                                         |   ×    | 66.52     | 75.21     | 38.88     | 525.00     | 44.20     |
| SigLIP (ViT-SO400M-384px)                                                                     |   ×    | 69.28     | 76.71     | 41.38     | 554.00     | 46.78     |
| DFN5B (ViT-H-14-378px)                                                                        |   ×    | 64.36     | 70.87     | 38.59     | 473.00     | **48.00** |
| **[MLCD (ViT-L-14-336px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336)**   |   ×    | 67.84     | 76.46     | 43.48     | 531.00     | 44.30     |
| **[MLCD (ViT-bigG-14-336px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-336)** |   √    | 71.07     | 79.63     | 44.38     | 572.00     | 46.78     |
| **[MLCD (ViT-bigG-14-448px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-448)** |   √    | **73.80** | **83.34** | **46.59** | **582.00** | 46.00     |


## Usage

```python
from transformers import AutoProcessor, MLCDVisionModel
from PIL import Image
import requests
import torch

# Load model and processor
model = MLCDVisionModel.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-336")
processor = AutoProcessor.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-336")

# Process single image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

# Get visual features
outputs = model(**inputs)
features = outputs.last_hidden_state

print(f"Extracted features shape: {features.shape}")
```

## MLCDVisionConfig

[[autodoc]] MLCDVisionConfig

## MLCDVisionModel

[[autodoc]] MLCDVisionModel
    - forward
