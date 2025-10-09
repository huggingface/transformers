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
*This model was released on 2022-09-08 and added to Hugging Face Transformers on 2023-03-13 and contributed by [yuekun](https://huggingface.co/yuekun).*

# MGP-STR

[MGP-STR](https://huggingface.co/papers/2209.03592) is a vision-based Scene Text Recognition (STR) model built on the Vision Transformer (ViT) architecture. It outperforms existing STR models by integrating linguistic knowledge through a Multi-Granularity Prediction (MGP) strategy. This strategy uses subword representations (BPE and WordPiece) alongside character-level representations without requiring a separate language model. MGP-STR achieves an average recognition accuracy of 93.35% on standard benchmarks and is trained on synthetic datasets MJSynth and SynthText.

<hfoptions id="usage">
<hfoption id="">

```py
import torch
import requests
from PIL import Image
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition

processor = MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base")
model = MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base", dtype="auto")

url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values
outputs = model(pixel_values)
print(processor.batch_decode(outputs.logits)['generated_text'])
```

</hfoption>
</hfoptions>

## MgpstrConfig

[[autodoc]] MgpstrConfig

## MgpstrTokenizer

[[autodoc]] MgpstrTokenizer
    - save_vocabulary

## MgpstrProcessor

[[autodoc]] MgpstrProcessor
    - __call__
    - batch_decode

## MgpstrModel

[[autodoc]] MgpstrModel
    - forward

## MgpstrForSceneTextRecognition

[[autodoc]] MgpstrForSceneTextRecognition
    - forward

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-to-text", model="alibaba-damo/mgp-str-base", dtype="auto")
pipeline("path/to/image.png")
```

