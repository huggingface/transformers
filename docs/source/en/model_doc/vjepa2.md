<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# VJEPA 2

V-JEPA 2 is a self-supervised approach to training video encoders, using internet-scale video data, that attains state-of-the-art performance on motion understanding and human action anticpation tasks. V-JEPA 2-AC is a latent action-conditioned world model post-trained from V-JEPA 2 (using a small amount of robot trajectory interaction data) that solves robot manipulation tasks without environment-specific data collection or task-specific training or calibration.

You can find all original VJEPA2 checkpoints under the [VJEPA 2](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6) collection.

The snippet below shows how to load the VJEPA 2 model using `AutoModel` class.

```py
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
encoder_outputs = outputs.last_hidden_state
predictor_outputs = outputs.predictor_output.last_hidden_state
```

## VJEPA2Config

[[autodoc]] VJEPA2Config

<frameworkcontent>
<pt>

## VJEPA2Model

[[autodoc]] VJEPA2Model
    - forward

</frameworkcontent>

## VJEPA2VideoProcessor

[[autodoc]] VJEPA2VideoProcessor
