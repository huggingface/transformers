<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->
*This model was released on 2021-09-21 and added to Hugging Face Transformers on 2021-10-13 and contributed by [nielsr](https://huggingface.co/nielsr).*

# TrOCR

[TrOCR](https://huggingface.co/papers/2109.10282) is an end-to-end text recognition model that uses Transformers for both image understanding and text generation at the wordpiece level, replacing the traditional CNN-RNN pipeline. It can be pre-trained on large-scale synthetic data and fine-tuned with human-labeled datasets, simplifying the architecture by eliminating the need for a separate language model for post-processing. TrOCR achieves state-of-the-art performance across printed, handwritten, and scene text recognition tasks. The model and code are publicly available for use and experimentation.

<hfoptions id="usage">
<hfoption id="VisionEncoderDecoderModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten", dtype="auto")

url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

</hfoption>
</hfoptions>

## TrOCRConfig

[[autodoc]] TrOCRConfig

## TrOCRProcessor

[[autodoc]] TrOCRProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## TrOCRForCausalLM

[[autodoc]] TrOCRForCausalLM
     - forward

