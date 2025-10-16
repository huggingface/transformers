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
*This model was released on 2025-07-15 and added to Hugging Face Transformers on 2025-07-18 and contributed by [eustlb](https://huggingface.co/eustlb).*

# Voxtral

[Voxtral](https://huggingface.co/papers/2507.13264) are multimodal audio–text chat models designed to process both spoken audio and written language. They achieve state-of-the-art results on diverse audio benchmarks while maintaining strong text performance. Voxtral Small, despite its compact size and ability to run locally, surpasses several closed-source systems. Both models support a 32K context window—allowing analysis of up to 40-minute audio segments and long conversations—and are released under the Apache 2.0 license along with three new benchmarks for evaluating speech understanding in knowledge and trivia tasks.

<hfoptions id="usage">
<hfoption id="VoxtralForConditionalGeneration">

```py
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained( "mistralai/Voxtral-Mini-3B-2507")
model = VoxtralForConditionalGeneration.from_pretrained( "mistralai/Voxtral-Mini-3B-2507", dtype="auto")

inputs = processor.apply_transcription_request(language="en", audio="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3", model_id=mistralai/Voxtral-Mini-3B-2507)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated responses:")
print("=" * 80)
for decoded_output in decoded_outputs:
    print(decoded_output)
    print("=" * 80)
```

</hfoption>
</hfoptions>

## VoxtralConfig

[[autodoc]] VoxtralConfig

## VoxtralEncoderConfig

[[autodoc]] VoxtralEncoderConfig

## VoxtralProcessor

[[autodoc]] VoxtralProcessor

## VoxtralEncoder

[[autodoc]] VoxtralEncoder
    - forward

## VoxtralForConditionalGeneration

[[autodoc]] VoxtralForConditionalGeneration
    - forward
