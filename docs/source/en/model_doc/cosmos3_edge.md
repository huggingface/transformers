<!--Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not
be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-10.*

# Cosmos3 Edge

[Cosmos3 Edge](https://huggingface.co/nvidia/Cosmos3-Edge) is NVIDIA's multimodal reasoning model from the Cosmos3
family. Transformers integrates the **Reasoner** tower only; the checkpoint's diffusion Generator, VAE, scheduler,
and other generation components remain Diffusers components.

The reasoner uses a dense, Llama-compatible language tower with 28 decoder blocks, each containing attention and an
MLP. Its SigLIP2 vision encoder accepts packed variable-resolution patches, uses sequence boundaries to keep images
and video frames independent during vision attention, groups patches spatially in 2×2 blocks, and projects them into
the language model. Image and video inputs use multimodal rotary position IDs; video prompts are expanded into one
timestamped vision span per sampled frame.

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "nvidia/Cosmos3-Edge"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
print(processor.batch_decode(generated_ids, skip_special_tokens=True))
```

## Cosmos3EdgeConfig

[[autodoc]] Cosmos3EdgeConfig

## Cosmos3EdgeTextConfig

[[autodoc]] Cosmos3EdgeTextConfig

## Cosmos3EdgeVisionConfig

[[autodoc]] Cosmos3EdgeVisionConfig

## Cosmos3EdgeProjectorConfig

[[autodoc]] Cosmos3EdgeProjectorConfig

## Cosmos3EdgeProcessor

[[autodoc]] Cosmos3EdgeProcessor
    - __call__
    - apply_chat_template

## Cosmos3EdgeImageProcessor

[[autodoc]] Cosmos3EdgeImageProcessor
    - preprocess

## Cosmos3EdgeImageProcessorPil

[[autodoc]] Cosmos3EdgeImageProcessorPil
    - preprocess

## Cosmos3EdgeVideoProcessor

[[autodoc]] Cosmos3EdgeVideoProcessor
    - preprocess

## Cosmos3EdgeModel

[[autodoc]] Cosmos3EdgeModel
    - forward
    - get_image_features
    - get_video_features

## Cosmos3EdgeTextModel

[[autodoc]] Cosmos3EdgeTextModel
    - forward

## Cosmos3EdgeVisionModel

[[autodoc]] Cosmos3EdgeVisionModel
    - forward

## Cosmos3EdgeForConditionalGeneration

[[autodoc]] Cosmos3EdgeForConditionalGeneration
    - forward
    - get_image_features
    - get_video_features
