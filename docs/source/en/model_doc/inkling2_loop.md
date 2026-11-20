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
*This model was contributed to Hugging Face Transformers on 2026-11-19.*

# Inkling 2 Mini

Inkling 2 Mini is a multimodal model from [Thinking Machines Lab](https://huggingface.co/thinkingmachines) that accepts text and image inputs and generates text outputs. It is based on [Inkling 2 Small](./inkling2) but has only 24 decoder layers instead of 48. The lower layer count is compensated for by a looping mechanism in which layers 6 to 17 run three times. Information is shared between loop passes through a learned message-passing mechanism that mixes the residuals of the previous pass into the attention and mixture-of-experts inputs of each loop layer. The decoder has a hidden size of 2304 and routes each token to 8 of 128 experts. The image encoder, tokenizer and processor are identical to those of Inkling 2 Small.

Inkling 2 Mini comes with an additional [research checkpoint](https://huggingface.co/thinkingmachines/TML-d24-varloop-research) that supports a variable number of loop passes. The architecture is identical to the main Inkling 2 Mini model. The number of loop passes can be adjusted to trade off speed against quality. Fewer passes are faster, while more passes improve quality.

Both checkpoints contain NVFP4 and MXFP8 quantized layers that Transformers dequantizes when loading the model.

**Checkpoints**

- [Inkling 2 Mini](http://hf.co/thinkingmachines/Inkling-2-Mini)
- [Inkling 2 Mini varloop research](http://hf.co/thinkingmachines/TML-d24-varloop-research)

## Resources

- [Hugging Face Blog](https://huggingface.co/blog/inkling2mini)

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch

from transformers import pipeline

pipe = pipeline(
    "image-text-to-text", model="thinkingmachines/inkling-1116-share", dtype=torch.bfloat16, device_map="auto"
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
# <|message_model|><|content_thinking|>The user is asking about an animal in an image. Let me
# analyze what I see: a small wild feline with thick, fluffy fur dusted with snow, short legs, a
# broad face, and black stripes/spots.
# ...
# <|end_message|><|message_model|><|content_text|>The animal appears to be a Pallas’s cat (manul) —
# a small, bear-like wild cat with thick speckled/striped fur. It’s standing on snow inside what
# looks like a zoo or wildlife enclosure, with snow-covered birch/log branches and a chain-link
# fence in the background.<|end_message|><|content_model_end_sampling|>
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch

from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/inkling-1116-share"
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
# <|message_model|><|content_thinking|>The user is asking what animal is in the image and where it
# is. Let me analyze the image.
# ...
# <|end_message|><|message_model|><|content_text|>The animal appears to be a Pallas’s cat (manul), a
# stocky small wild cat with thick, snow-dusted fur. It is standing on snow, likely inside a zoo or
# wildlife enclosure — you can see birch/aspen-like posts and chain-link fencing in the
# background.<|end_message|><|content_model_end_sampling|>
```

</hfoption>

<hfoption id="AutoModel Variable Loops">

```python
import torch

from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/TML-d24-varloop-research"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    text_config={"loop_num_passes": 3},  # number of loop passes, must be at least 1
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
# <|message_model|><|content_thinking|>We need answer image question. Need identify animal and where
# is it.
# ...
# <|end_message|><|message_model|><|content_text|>The animal is a Pallas’s cat (a.k.a. Manul), a
# stocky, densely furred small wild cat. It’s in the snow outdoors in a cold setting—likely a winter
# enclosure/habitat—standing/walking on snow at the center of the
# image.<|end_message|><|content_model_end_sampling|>
```

</hfoption>

</hfoptions>

## Notes

- Attention runs in eager mode only because every layer adds a relative position bias and a sink.
- Every pass through the loop layers keeps its own keys, values and convolution states, so the cache has one layer per layer visit (48 for the released checkpoint).
- `output_hidden_states` and `output_attentions` return one entry per layer visit.

## Inkling2LoopConfig

[[autodoc]] Inkling2LoopConfig

## Inkling2LoopTextConfig

[[autodoc]] Inkling2LoopTextConfig

## Inkling2LoopPreTrainedModel

[[autodoc]] Inkling2LoopPreTrainedModel
    - forward

## Inkling2LoopTextModel

[[autodoc]] Inkling2LoopTextModel
    - forward

## Inkling2LoopForCausalLM

[[autodoc]] Inkling2LoopForCausalLM
    - forward

## Inkling2LoopModel

[[autodoc]] Inkling2LoopModel
    - forward

## Inkling2LoopForConditionalGeneration

[[autodoc]] Inkling2LoopForConditionalGeneration
    - forward
