<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-10-14 and added to Hugging Face Transformers on 2023-02-03 and contributed by [Matthijs](https://huggingface.co/Matthijs).*

# SpeechT5

[SpeechT5](https://huggingface.co/papers/2110.07205) proposes a unified-modal framework that applies encoder-decoder pre-training for self-supervised learning of speech and text representations. The framework includes a shared encoder-decoder network and modal-specific pre/post-nets. It preprocesses speech and text inputs, models sequence-to-sequence transformations, and generates outputs in the respective modality. Using large-scale unlabeled data, SpeechT5 learns a unified representation to enhance modeling capabilities for both speech and text. A cross-modal vector quantization approach aligns textual and speech information in a unified semantic space. Evaluations demonstrate SpeechT5's effectiveness across tasks such as automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="microsoft/speecht5_asr", dtype="auto")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="SpeechT5ForSpeechToText">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, SpeechT5ForSpeechToText

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("microsoft/speecht5_asr")
model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr", dtype="auto")

inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
predicted_ids = model.generate(**inputs, max_length=100)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## SpeechT5Config

[[autodoc]] SpeechT5Config

## SpeechT5HifiGanConfig

[[autodoc]] SpeechT5HifiGanConfig

## SpeechT5Tokenizer

[[autodoc]] SpeechT5Tokenizer
    - __call__
    - save_vocabulary
    - decode
    - batch_decode

## SpeechT5FeatureExtractor

[[autodoc]] SpeechT5FeatureExtractor
    - __call__

## SpeechT5Processor

[[autodoc]] SpeechT5Processor
    - __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## SpeechT5Model

[[autodoc]] SpeechT5Model
    - forward

## SpeechT5ForSpeechToText

[[autodoc]] SpeechT5ForSpeechToText
    - forward

## SpeechT5ForTextToSpeech

[[autodoc]] SpeechT5ForTextToSpeech
    - forward
    - generate

## SpeechT5ForSpeechToSpeech

[[autodoc]] SpeechT5ForSpeechToSpeech
    - forward
    - generate_speech

## SpeechT5HifiGan

[[autodoc]] SpeechT5HifiGan
    - forward

