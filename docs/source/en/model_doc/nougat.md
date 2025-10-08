<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->
*This model was released on 2023-08-25 and added to Hugging Face Transformers on 2023-09-26 and contributed by [nielsr](https://huggingface.co/nielsr).*

# Nougat

[Nougat: Neural Optical Understanding for Academic Documents](https://huggingface.co/papers/2308.13418) uses a Visual Transformer architecture similar to Donut, featuring an image Transformer encoder and an autoregressive text Transformer decoder. This model processes scientific documents, converting them into markdown format to enhance accessibility and preserve semantic information, especially for mathematical expressions.

<hfoptions id="usage">
<hfoption id="VisionEncoderDecoderModel">

```py
import torch
import re
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel, infer_device
from datasets import load_dataset

processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base", dtype="auto")

filepath = hf_hub_download(repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_paper.png", repo_type="dataset")
image = Image.open(filepath)
pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(
    pixel_values.to(device),
    min_length=1,
    max_new_tokens=30,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(processor.post_process_generation(sequence, fix_markdown=False))
```

</hfoption>
</hfoptions>

## NougatImageProcessor

[[autodoc]] NougatImageProcessor
    - preprocess

## NougatImageProcessorFast

[[autodoc]] NougatImageProcessorFast
    - preprocess

## NougatTokenizerFast

[[autodoc]] NougatTokenizerFast

## NougatProcessor

[[autodoc]] NougatProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode
    - post_process_generation

