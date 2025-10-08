<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-05-31 and added to Hugging Face Transformers on 2025-08-18 and contributed by [thisisiron](https://huggingface.co/thisisiron).*

# Ovis2

[Ovis2](https://huggingface.co/papers/2405.20797) is a multimodal large language model (MLLM) that addresses the misalignment between textual embeddings and visual embeddings in traditional MLLMs. It introduces a learnable visual embedding table within the visual encoder, allowing each image patch to index the table multiple times and produce a probabilistic combination of embeddings that structurally mirrors textual embeddings. This alignment enables more seamless fusion of visual and textual information. Evaluations show that Ovis outperforms comparable open-source MLLMs and even surpasses the proprietary Qwen-VL-Plus, demonstrating the effectiveness of its structured visual representation.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained("thisisiron/Ovis2-2B-hf", dtype="auto")
processor = AutoProcessor.from_pretrained("thisisiron/Ovis2-2B-hf")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url},
            {"type": "text", "text": "Describe the image."},
        ],
    },
]
messages = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    images=[image],
    text=messages,
    return_tensors="pt",
)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(output_text)
```

</hfoption>
</hfoptions>

## Ovis2Config

[[autodoc]] Ovis2Config

## Ovis2VisionConfig

[[autodoc]] Ovis2VisionConfig

## Ovis2Model

[[autodoc]] Ovis2Model

## Ovis2ForConditionalGeneration

[[autodoc]] Ovis2ForConditionalGeneration
    - forward

## Ovis2ImageProcessor

[[autodoc]] Ovis2ImageProcessor

## Ovis2ImageProcessorFast

[[autodoc]] Ovis2ImageProcessorFast

## Ovis2Processor

[[autodoc]] Ovis2Processor
