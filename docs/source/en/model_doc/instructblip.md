<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2023-05-11 and added to Hugging Face Transformers on 2023-06-26 and contributed by [nielsr](https://huggingface.co/nielsr).*

# InstructBLIP

[InstructBLIP](https://huggingface.co/papers/2305.06500) leverages the BLIP-2 architecture for vision-language instruction tuning. It uses a wide variety of 26 publicly available datasets, transformed into an instruction tuning format, and categorized into held-in and held-out clusters. The model introduces instruction-aware visual feature extraction, enhancing its ability to extract relevant features based on given instructions. InstructBLIP achieves state-of-the-art zero-shot performance across 13 held-out datasets, outperforming BLIP-2 and Flamingo. It also excels in fine-tuned downstream tasks, such as achieving 90.7% accuracy on ScienceQA IMG, and demonstrates qualitative advantages over other multimodal models.

<hfoptions id="usage">
<hfoption id="InstructBlipForConditionalGeneration">

```py
import torch
import requests
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", dtype="auto")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")


url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
prompt = "What is the weather in this image?"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    do_sample=False,
    num_beams=5,
    max_length=256,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.5,
    length_penalty=1.0,
    temperature=1,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
```

</hfoption>
</hfoptions>

## Usage tips

- InstructBLIP uses the same architecture as BLIP-2 with one key difference: it feeds the text prompt (instruction) to the Q-Former.
- BLIP models after release v4.46 raise warnings about adding `processor.num_query_tokens = {{num_query_tokens}}` and expanding model embeddings to add special `<image>` tokens. Add these attributes to the processor if you own the model checkpoint, or open a PR if you don't.
- Adding these attributes means BLIP adds the required query tokens per image and expands text with `<image>` placeholders. Usually around 500 tokens per image, so ensure text isn't truncated to avoid embedding merge failures.
- Get attributes from `model.config.num_query_tokens` and expand model embeddings following this [link](https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042).

## InstructBlipConfig

[[autodoc]] InstructBlipConfig

## InstructBlipVisionConfig

[[autodoc]] InstructBlipVisionConfig

## InstructBlipQFormerConfig

[[autodoc]] InstructBlipQFormerConfig

## InstructBlipProcessor

[[autodoc]] InstructBlipProcessor

## InstructBlipVisionModel

[[autodoc]] InstructBlipVisionModel
    - forward

## InstructBlipQFormerModel

[[autodoc]] InstructBlipQFormerModel
    - forward

## InstructBlipForConditionalGeneration

[[autodoc]] InstructBlipForConditionalGeneration
    - forward
    - generate

## InstructBlipModel

[[autodoc]] InstructBlipModel
    - forward

