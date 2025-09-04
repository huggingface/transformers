<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-10-26 and added to Hugging Face Transformers on 2021-12-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
           <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# WavLM

[WavLM](https://huggingface.co/papers/2110.13900) is a self-supervised speech representation model from Microsoft designed to work across the “full stack” of speech tasks, from automatic speech recognition (ASR) to speaker diarization and audio event detection. It builds on HuBERT’s masked prediction approach but introduces denoising and data augmentation to make the learned representations more robust in noisy and multi-speaker conditions.

You can find all the original WavLM checkpoints under the [WavLM](https://huggingface.co/models?other=wavlm) collection.

> [!TIP]
> This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
>
> Click on the WavLM models in the right sidebar for more examples of how to apply WavLM to different audio tasks.

The example below demonstrates how to extract audio features with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipe = pipeline(
    task="feature-extraction",
    model="microsoft/wavlm-base",
    torch_dtype=torch.float16,
    device=0
)

features = pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
print(features)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("microsoft/wavlm-base")
model = AutoModel.from_pretrained(
    "microsoft/wavlm-base",
    torch_dtype=torch.float16,
    device_map="auto"
)

audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
inputs = processor(audio_url, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.last_hidden_state)
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. 
Refer to the [Quantization](https://huggingface.co/docs/transformers/en/quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the weights to 4-bits.

```python
from transformers import WavLMForCTC, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = WavLMForCTC.from_pretrained(
    "microsoft/wavlm-large",
    quantization_config=bnb_config,
    device_map="auto"
)
```

## Notes
- WavLM processes raw 16kHz audio waveforms provided as 1D float arrays. Use `Wav2Vec2Processor` for preprocessing.
- For CTC-based fine-tuning, model outputs should be decoded with `Wav2Vec2CTCTokenizer`.
- The model works particularly well for tasks like speaker verification, identification, and diarization.

## Resources
- [Audio classification task guide](https://huggingface.co/docs/transformers/en/tasks/audio_classification)
- [Automatic speech recognition task guide](https://huggingface.co/docs/transformers/en/tasks/asr)

## WavLMConfig

[[autodoc]] WavLMConfig

## WavLMModel

[[autodoc]] WavLMModel
    - forward

## WavLMForCTC

[[autodoc]] WavLMForCTC
    - forward

## WavLMForSequenceClassification

[[autodoc]] WavLMForSequenceClassification
    - forward

## WavLMForAudioFrameClassification

[[autodoc]] WavLMForAudioFrameClassification
    - forward

## WavLMForXVector

[[autodoc]] WavLMForXVector
    - forward