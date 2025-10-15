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
*This model was released on 2025-02-27 and added to Hugging Face Transformers on 2025-05-07 and contributed by [eustlb](https://huggingface.co/eustlb).*

# CSM

[CSM](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) is an end-to-end multimodal transformer system that generates contextually appropriate, high-fidelity speech by interleaving text and audio tokens. It operates directly on Residual Vector Quantization (RVQ) audio tokens and splits processing into two transformers: a large multimodal backbone that predicts the zeroth codebook and a lightweight audio decoder that handles the remaining codebooks for real-time generation. This structure allows CSM to capture conversational context while maintaining low latency. To train efficiently, it uses a compute amortization technique—training the audio decoder on only a small random subset of frames—preserving quality while dramatically reducing memory and compute costs.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-to-audio", model="sesame/csm-1b", dtype="auto")
output = pipeline("Plants create energy through a process known as photosynthesis.")
audio = output["audio"]
```

</hfoption>
<hfoption id="CsmForConditionalGeneration">

```py
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("sesame/csm-1b")
model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b", dtype="auto")

conversation = [
    {"role": "0", "content": [{"type": "text", "text": "Plants generate energy through a process known as photosynthesis."}]},
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
)

audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "example_with_context.wav")
```

</hfoption>
</hfoptions>

## Usage tips

- CSM generates speech from conversations while maintaining voice consistency and contextual awareness.
- The model supports batched inference for processing multiple inputs simultaneously.
- Full-graph compilation with CUDA graphs accelerates inference performance.
- Training is supported through the Transformers integration.

## CsmConfig

[[autodoc]] CsmConfig

## CsmDepthDecoderConfig

[[autodoc]] CsmDepthDecoderConfig

## CsmProcessor

[[autodoc]] CsmProcessor
    - __call__

## CsmForConditionalGeneration

[[autodoc]] CsmForConditionalGeneration
    - forward
    - generate

## CsmDepthDecoderForCausalLM

[[autodoc]] CsmDepthDecoderForCausalLM

## CsmDepthDecoderModel

[[autodoc]] CsmDepthDecoderModel

## CsmBackboneModel

[[autodoc]] CsmBackboneModel
