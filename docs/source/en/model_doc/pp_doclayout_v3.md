<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-01-28 and added to Hugging Face Transformers on 2026-01-28.*

# PP-DocLayoutV3

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

TBD.

## Usage

### Single input inference

The example below demonstrates how to generate text with PP-DocLayoutV3 using [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
from PIL import Image
from transformers import pipeline

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg", stream=True).raw)
layout_detector = pipeline("object-detection", model="PaddlePaddle/PP-DocLayoutV3_safetensors")
results = layout_detector(image)
for idx, res in enumerate(results):
    print(f"Order {idx + 1}: {res}")
```

</hfoption>

<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

model_path = "PaddlePaddle/PP-DocLayoutV3_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg", stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")

outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])
for result in results:
    for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"Order {idx + 1}: {model.config.id2label[label]}: {score:.2f} {box}")
```

</hfoption>
</hfoptions>

### Batched inference

PP-DocLayoutV3 also supports batched inference. Here is how you can do it with PP-DocLayoutV3 using [`Pipeline`] or the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
from PIL import Image
from transformers import pipeline

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg", stream=True).raw)
layout_detector = pipeline("object-detection", model="PaddlePaddle/PP-DocLayoutV3_safetensors")
results = layout_detector([image, image])
for result in results:
    print("result:")
    for idx, res in enumerate(result):
        print(f"Order {idx + 1}: {res}")
```

</hfoption>

<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

model_path = "PaddlePaddle/PP-DocLayoutV3_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg", stream=True).raw)
inputs = image_processor(images=[image, image], return_tensors="pt")
target_sizes = [image.size[::-1], image.size[::-1]]

outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes)
for result in results:
    print("result:")
    for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"Order {idx + 1}: {model.config.id2label[label]}: {score:.2f} {box}")
```

</hfoption>
</hfoptions>

## PPDocLayoutV3ForObjectDetection

[[autodoc]] PPDocLayoutV3ForObjectDetection
    - forward

## PPDocLayoutV3Model

[[autodoc]] PPDocLayoutV3Model
    
## PPDocLayoutV3Config

[[autodoc]] PPDocLayoutV3Config

## PPDocLayoutV3ImageProcessorFast

[[autodoc]] PPDocLayoutV3ImageProcessorFast
