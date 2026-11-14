<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-11.*

# Inkling 2 Small

[Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8) is a multimodal model from [Thinking Machines Lab](https://huggingface.co/thinkingmachines) that accepts text and image inputs and generates text outputs. It is intended for use in English and other languages, and across multiple coding languages. It keeps the decoder of [Inkling](./inkling), with learned relative attention, short causal convolutions and mixture-of-experts layers, and adds block attention residuals, gated attention with a learned sink and dynamic key/value convolutions. The language backbone consists of 48 mixture-of-experts decoder layers, each routing tokens to 8 of 384 experts. The model has 594B total and 17B active parameters. Images are encoded by a vision transformer based on the Perception Encoder and projected into the language model embeddings, one embedding per 40x40 pixel patch. 

The model is released with two checkpoints:
- [`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
- [`thinkingmachines/Inkling-2-Small-MXFP8`](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8)

Both checkpoints contain quantized layers. `Inkling-2-Small` has both NVFP4 and MXFP8 quantized decoder layers. `Inkling-2-Small-MXFP8` contains only MXFP8 quantized layers. Currently only the `Inkling-2-Small-MXFP8` checkpoint is supported.


## Resources

- [Thinking Machines Blog](https://thinkingmachines.ai/news/inkling-2-small/)
- [Hugging Face Blog](https://huggingface.co/blog/thinkingmachines-inkling)


<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch

from transformers import pipeline

pipe = pipeline(
    "image-text-to-text", model="thinkingmachines/Inkling-2-Small-MXFP8", dtype=torch.bfloat16, device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "What animal is in this image, and where is it?"},
        ],
    },
]

output = pipe(messages, max_new_tokens=1024, return_full_text=False, skip_special_tokens=False)
print(output[0]["generated_text"])
# <|message_model|><|content_thinking|>The user is asking to identify the animal and location in the image. Need answer based on visible content. The animal looks like a Pallas's cat (Otocolobus manul), also called manul:
# ...
# <|end_message|><|message_model|><|content_text|>The animal is a **Pallas’s cat** (also called a **manul**). It’s walking through **snow** in front of birch logs, likely in a **zoo or enclosed habitat**. The exact geographic location can’t be determined from the image alone; in the wild they’re native to parts of **Central Asia**.<|end_message|><|content_model_end_sampling|>
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch

from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling-2-Small-MXFP8"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "What animal is in this image, and where is it?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
# <|message_model|><|content_thinking|>The user is asking to identify the animal and location in the image. Need answer based on visible content. The animal looks like a Pallas's cat (Otocolobus manul), also called manul:
# ...
# <|end_message|><|message_model|><|content_text|>The animal is a **Pallas’s cat** (also called a **manul**). It’s walking through **snow** in front of birch logs, likely in a **zoo or enclosed habitat**. The exact geographic location can’t be determined from the image alone; in the wild they’re native to parts of **Central Asia**.<|end_message|><|content_model_end_sampling|>
```

</hfoption>
</hfoptions>

## Notes

- Attention runs in eager mode only, because every layer adds a relative position bias and a sink.

## Inkling2Config

[[autodoc]] Inkling2Config

## Inkling2TextConfig

[[autodoc]] Inkling2TextConfig

## Inkling2VisionConfig

[[autodoc]] Inkling2VisionConfig

## Inkling2PreTrainedModel

[[autodoc]] Inkling2PreTrainedModel
    - forward

## Inkling2TextModel

[[autodoc]] Inkling2TextModel
    - forward

## Inkling2ForCausalLM

[[autodoc]] Inkling2ForCausalLM
    - forward

## Inkling2VisionModel

[[autodoc]] Inkling2VisionModel
    - forward

## Inkling2Model

[[autodoc]] Inkling2Model
    - forward

## Inkling2ForConditionalGeneration

[[autodoc]] Inkling2ForConditionalGeneration
    - forward

## Inkling2ImageProcessor

[[autodoc]] Inkling2ImageProcessor

## Inkling2Processor

[[autodoc]] Inkling2Processor
