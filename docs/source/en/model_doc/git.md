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
*This model was released on 2022-05-27 and added to Hugging Face Transformers on 2023-01-03 and contributed by [nielsr](https://huggingface.co/nielsr).*

# GIT

[GIT](https://huggingface.co/papers/2205.14100) is a decoder-only Transformer that uses CLIP's vision encoder to condition on visual inputs alongside text. By simplifying the architecture to a single image encoder and text decoder under a unified language modeling task, and by scaling up pre-training data and model size, GIT achieves state-of-the-art results on 12 challenging vision-language benchmarks. Notably, it surpasses human performance on TextCaps. Additionally, GIT introduces a generation-based approach for image classification and scene text recognition, demonstrating strong performance on standard benchmarks.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = processor(images=image, return_tensors="pt").pixel_values
output_ids = model.generate(pixel_values=pixel_values, max_length=50)
output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(output)
```

</hfoption>
</hfoptions>

## Usage tips

- GIT implements GPT-2 architecture with additional conditioning on `pixel_values`.

## GitVisionConfig

[[autodoc]] GitVisionConfig

## GitVisionModel

[[autodoc]] GitVisionModel
    - forward

## GitConfig

[[autodoc]] GitConfig
    - all

## GitProcessor

[[autodoc]] GitProcessor
    - __call__

## GitModel

[[autodoc]] GitModel
    - forward

## GitForCausalLM

[[autodoc]] GitForCausalLM
    - forward

