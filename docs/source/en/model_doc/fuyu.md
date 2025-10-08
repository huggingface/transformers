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
*This model was released on {release_date} and added to Hugging Face Transformers on 2023-10-19 and contributed by [Molbap](https://huggingface.co/Molbap).*

# Fuyu

[Fuyu](https://www.adept.ai/blog/fuyu-8b) is a small, open-source multimodal model designed for AI agents that can handle both text and images. Unlike most multimodal models, it uses a decoder-only Transformer without a separate image encoder, projecting image patches directly into the transformer and supporting arbitrary image resolutions. This simplified architecture allows fast inference—under 100 milliseconds for large images—and streamlines training by removing multiple specialized stages. Despite its small size, Fuyu-8B achieves competitive performance on standard image understanding benchmarks like VQAv2, OKVQA, COCO Captions, and AI2D, outperforming larger models on several metrics while being easier to scale and deploy.

<hfoptions id="usage">
<hfoption id="Pipeline">

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("adept/fuyu-8b")
model = AutoModelForCausalLM.from_pretrained("adept/fuyu-8b", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "Generate a coco-style caption.\n"

inputs = processor(images=image, text=prompt, return_tensors="pt")
outputs = model(**inputs)
generated_ids = model.generate(**inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generated_ids[:, -7:], skip_special_tokens=True)
print(generation_text[0])
```

</hfoption>
</hfoptions>

## FuyuConfig

[[autodoc]] FuyuConfig

## FuyuForCausalLM

[[autodoc]] FuyuForCausalLM
    - forward

## FuyuModel

[[autodoc]] FuyuModel
    - forward

## FuyuImageProcessor

[[autodoc]] FuyuImageProcessor
    - __call__

## FuyuImageProcessor

[[autodoc]] FuyuImageProcessorFast
    - __call__

## FuyuProcessor

[[autodoc]] FuyuProcessor
    - __call__
