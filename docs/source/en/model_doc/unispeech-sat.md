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
*This model was released on 2021-10-12 and added to Hugging Face Transformers on 2021-10-26 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# UniSpeech-SAT

[UniSpeech-SAT](https://huggingface.co/papers/2110.05752) enhances self-supervised learning for speaker representation by integrating multi-task learning with utterance-wise contrastive loss and an utterance mixing strategy for data augmentation. These methods are incorporated into the HuBERT framework, achieving state-of-the-art performance on the SUPERB benchmark, particularly for speaker identification tasks. Scaling up the training dataset to 94 thousand hours of public audio data further improves performance across all SUPERB tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="microsoft/unispeech-sat-base-100h-libri-ft", dtype="auto")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCTC

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
model = AutoModelForCTC.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## Usage tips

- UniSpeech accepts raw speech waveforms as float arrays. Use `Wav2Vec2Processor` for feature extraction.
- [`UniSpeech`] models fine-tune using connectionist temporal classification (CTC). Decode model outputs with [`Wav2Vec2CTCTokenizer`].

## UniSpeechSatConfig

[[autodoc]] UniSpeechSatConfig

## UniSpeechSat specific outputs

[[autodoc]] models.unispeech_sat.modeling_unispeech_sat.UniSpeechSatForPreTrainingOutput

## UniSpeechSatModel

[[autodoc]] UniSpeechSatModel
    - forward

## UniSpeechSatForCTC

[[autodoc]] UniSpeechSatForCTC
    - forward

## UniSpeechSatForSequenceClassification

[[autodoc]] UniSpeechSatForSequenceClassification
    - forward

## UniSpeechSatForAudioFrameClassification

[[autodoc]] UniSpeechSatForAudioFrameClassification
    - forward

## UniSpeechSatForXVector

[[autodoc]] UniSpeechSatForXVector
    - forward

## UniSpeechSatForPreTraining

[[autodoc]] UniSpeechSatForPreTraining
    - forward

