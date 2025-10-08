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
*This model was released on 2021-09-30 and added to Hugging Face Transformers on 2022-10-18 and contributed by [nielsr](https://huggingface.co/nielsr).*

# Table Transformer

[Table Transformer](https://huggingface.co/papers/2110.00061) introduces PubTables-1M, a dataset with nearly one million tables from scientific articles, addressing ground truth inconsistency through a canonicalization procedure. The dataset supports multiple input modalities and detailed header and location information. Two DETR-based models, one for table detection and another for table structure recognition, achieve excellent results across detection, structure recognition, and functional analysis tasks without task-specific customization.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="microsoft/table-transformer-detection", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/table-transformer-example.png")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/table-transformer-example.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", dtype="auto")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
    0
]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

</hfoption>
</hfoptions>


## TableTransformerConfig

[[autodoc]] TableTransformerConfig

## TableTransformerModel

[[autodoc]] TableTransformerModel
    - forward

## TableTransformerForObjectDetection

[[autodoc]] TableTransformerForObjectDetection
    - forward

