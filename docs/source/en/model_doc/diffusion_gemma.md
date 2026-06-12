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


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-06-10.*


# DiffusionGemma

## Overview

DiffusionGemma is engineered to reduce the sequential bottlenecks of standard causal language models. It employs an encoder-decoder architecture specifically optimized for inference speed.

The encoder operates in a prefill capacity, processing the initial prompt and generating the KV cache. The decoder then utilizes bidirectional attention to process an input block (a 'canvas') of tokens, accessing the cached context via cross-attention.

During inference, DiffusionGemma leverages multi-canvas sampling. Rather than generating one token at a time, the model iteratively denoises a full block of tokens using a diffusion sampler. Once a canvas is fully denoised, it is processed by the encoder and appended to the KV cache, after which the model generates the next canvas. This block-autoregressive approach facilitates text generation at higher speeds.

You can find the model card and checkpoint [here](https://huggingface.co/google/diffusiongemma-26B-A4B-it). You can find a visual guide to the model [here](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma).

## Usage examples

Despite it being a text diffusion model and having a custom generation loop, most of the interface is shared with other models that can generate text with [`DiffusionGemmaGenerationMixin.generate`]. If you're using another `transformers` model in your app, you should be able to directly replace it with this model.

### Common caveats

- DiffusionGemma doesn't accept `use_cache`. It always uses a KV cache;
- Support for common flags like `top_k` won't be available at release day, but will be added over time if they are compatible with text diffusion.

### Basic example

```python
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor


model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    "google/diffusiongemma-26B-A4B-it", device_map="auto",
)
processor = AutoProcessor.from_pretrained("google/diffusiongemma-26B-A4B-it")

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    # Add the following to enable thinking
    # enable_thinking=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Set `cache_implementation="static"` in `generate` to trigger `torch.compile`.
# Compilation is much faster, after warming up!
output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output.sequences[0][input_len:], skip_special_tokens=True))
```

### Streaming

Like other models that can generate text, you can set a streamer class to stream text. Unlike other models, DiffusionGemma generates intermediate drafts before the final text. You can visualize them with `TextDiffusionStreamer`

```python
from transformers import TextDiffusionStreamer

# (... copy from the example above, up to the `generate` call)
streamer = TextDiffusionStreamer(tokenizer=processor.tokenizer)
model.generate(**inputs, max_new_tokens=256, streamer=streamer)
```

### Setting a starting denoising output

The model is trained to iteratively refine blocks of 256 tokens. On some applications, it may be beneficial to provide a starting point for the decoder, rather than starting from random tokens. You can use the `decoder_input_ids`, available on all model interfaces, to set the starting canvas.

```py
initial_estimate = ... # a tensor with shape (bsz, 256)
model.generate(**inputs, max_new_tokens=256, decoder_input_ids=initial_estimate)
```

## DiffusionGemmaTextConfig

[[autodoc]] DiffusionGemmaTextConfig

## DiffusionGemmaConfig

[[autodoc]] DiffusionGemmaConfig

## DiffusionGemmaGenerationOutput

[[autodoc]] DiffusionGemmaGenerationOutput

## DiffusionGemmaGenerationMixin

[[autodoc]] DiffusionGemmaGenerationMixin
    - generate

## DiffusionGemmaGenerationConfig

[[autodoc]] DiffusionGemmaGenerationConfig

## EntropyBoundSamplerConfig

[[autodoc]] EntropyBoundSamplerConfig

## EntropyBoundSampler

[[autodoc]] EntropyBoundSampler

## StableAndConfidentStoppingCriteria

[[autodoc]] StableAndConfidentStoppingCriteria

## LinearTemperatureScheduleLogitsProcessor

[[autodoc]] LinearTemperatureScheduleLogitsProcessor

## DiffusionGemmaPreTrainedModel

[[autodoc]] DiffusionGemmaPreTrainedModel
    - forward

## DiffusionGemmaModel

[[autodoc]] DiffusionGemmaModel
    - forward

## DiffusionGemmaEncoderModel

[[autodoc]] DiffusionGemmaEncoderModel
    - forward

## DiffusionGemmaEncoderTextModel

[[autodoc]] DiffusionGemmaEncoderTextModel
    - forward

## DiffusionGemmaDecoderModel

[[autodoc]] DiffusionGemmaDecoderModel
    - forward

## DiffusionGemmaForBlockDiffusion

[[autodoc]] DiffusionGemmaForBlockDiffusion
    - forward