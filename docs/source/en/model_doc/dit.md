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
*This model was released on 2022-03-04 and added to Hugging Face Transformers on 2022-03-10 and contributed by [nielsr](https://huggingface.co/nielsr).*

# DiT

[DiT: Self-supervised Pre-training for Document Image Transformer](https://huggingface.co/papers/2203.02378) applies self-supervised pre-training to 42 million document images, achieving state-of-the-art results in document image classification, document layout analysis, and table detection. Specifically, it improves performance on the RVL-CDIP dataset from 91.11% to 92.69%, on PubLayNet from 91.0% to 94.9%, and on ICDAR 2019 cTDaR from 94.23% to 96.55%. DiT leverages large-scale unlabeled text images to address the lack of human-labeled document images in Document AI tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="microsoft/dit-base-finetuned-rvlcdip", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dit-example.jpg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dit-example.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- Load pretrained DiT weights in a [`BEiT`] model with a modeling head to predict visual tokens.