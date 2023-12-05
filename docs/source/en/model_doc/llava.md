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

# Llava

## Overview

The Llava model was proposed in [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/pdf/2310.03744) by Haotian Liu, Chunyuan Li, Yuheng Li and Yong Jae Lee.

The abstract from the paper is the following:

*Large multimodal models (LMM) have recently shown encouraging progress with visual instruction tuning. In this note, we show that the fully-connected vision-language cross-modal connector in LLaVA is surprisingly powerful and data-efficient. With simple modifications to LLaVA, namely, using CLIP-ViT-L-336px with an MLP projection and adding academic-task-oriented VQA data with simple response formatting prompts, we establish stronger baselines that achieve state-of-the-art across 11 benchmarks. Our final 13B checkpoint uses merely 1.2M publicly available data, and finishes full training in ∼1 day on a single 8-A100 node. We hope this can make state-of-the-art LMM research more accessible. Code and model will be publicly available*

Tips:

We have benchmarked our implementation against the original [`BakLlava`](https://github.com/SkunkworksAI/BakLLaVA) implementation that is derived from the original implementation and our implementation leads to important speedups in all scenarios

### Multiple prompts and fixed number of images

| implementation | batch size | Nb images per prompt | total time (s) |
|----------------|------------|----------------------|----------------|
| original       | 2          | 4                    | 4.73           |
| transformers   | 2          | 4                    | 1.72           |
| original       | 4          | 4                    | 8.74           |
| transformers   | 4          | 4                    | 3.63           |
| original       | 8          | 4                    | 17.65          |
| transformers   | 8          | 4                    | 4.35           |

### Multiple prompts and single image

| implementation | batch size | Nb images per prompt | total time (s) |
|----------------|------------|----------------------|----------------|
| original       | 2          | 1                    | 1.35           |
| transformers   | 2          | 1                    | 0.82           |
| original       | 4          | 1                    | 2.42           |
| transformers   | 4          | 1                    | 1.01           |
| original       | 8          | 1                    | 4.24           |
| transformers   | 8          | 1                    | 1.98           |

This model was contributed by [ybelkada](https://huggingface.co/ybelkada).
The original code can be found [here](https://github.com/haotian-liu/LLaVA/tree/main/llava).

### Using Flash Attention 2

Flash Attention 2 is an even faster, optimized version of the previous optimization.

#### Installation 

First, check whether your hardware is compatible with Flash Attention 2. The latest list of compatible hardware can be found in the [official documentation](https://github.com/Dao-AILab/flash-attention#installation-and-features). If your hardware is not compatible with Flash Attention 2, you can still benefit from attention kernel optimisations through Better Transformer support covered [above](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

Next, [install](https://github.com/Dao-AILab/flash-attention#installation-and-features) the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```


#### Usage

To load a model using Flash Attention 2, we can pass the `use_flash_attention_2` flag to [`.from_pretrained`](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). We'll also load the model in half-precision (e.g. `torch.float16`), since it results in almost no degradation to audio quality but significantly lower memory usage and faster inference:

```python
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/bakLlava-v1-hf", torch_dtype=torch.float16, use_flash_attention_2=True).to(device)
```

You can also use it with transformers `pipeline`:
```python
from transformers import pipeline
from PIL import Image    
import request

model_id = "llava-hf/bakLlava-v1-hf"
pipe = pipeline("image-to-text", model=model_id)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"

image = Image.open(requests.get(url, stream=True).raw)
prompt = "<image>\nUSER: What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud\nASSISTANT:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
>>> {"generated_text": "\nUSER: What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud\nASSISTANT: Lava"}
```

[[autodoc]] LlavaConfig

## LlavaProcessor

[[autodoc]] LlavaProcessor

## LlavaForConditionalGeneration

[[autodoc]] LlavaForConditionalGeneration
    - forward
