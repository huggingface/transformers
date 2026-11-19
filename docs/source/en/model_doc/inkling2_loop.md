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

# Looped Inkling 2

Looped Inkling 2 is a multimodal model from [Thinking Machines Lab](https://huggingface.co/thinkingmachines) that accepts text and image inputs and generates text outputs. It is [Inkling 2](./inkling2) with one change to how the decoder runs: a contiguous middle section of layers, the core, runs several times for every token with the same weights. The released checkpoint has 24 decoder layers of which layers 6 to 17 run three times, so every token goes through 48 layers with the weights of 24. Every pass after the first starts again from the output of the layers before the core, and reads the residual blocks of the previous pass through small learned messages that are mixed into the attention and mixture-of-experts inputs of each core layer. The decoder has a hidden size of 2304 and routes each token to 8 of 128 experts. The image encoder, tokenizer and processor are the ones of Inkling 2.

<hfoptions id="usage">
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
```

</hfoption>
</hfoptions>

## Notes

- Attention runs in eager mode only, because every layer adds a relative position bias and a sink.
- Every pass through the core keeps its own keys, values and convolution states, so the cache has one layer per layer visit (48 for the released checkpoint). The cache layers of the passes after the first follow those of the 24 decoder layers.
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
