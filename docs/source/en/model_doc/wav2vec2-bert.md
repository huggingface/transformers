<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-04-17 and added to Hugging Face Transformers on 2024-01-18 and contributed by [ylacombe](https://huggingface.co/ylacombe).*

# Wav2Vec2-BERT

[Wav2Vec2-BERT](https://huggingface.co/papers/2304.08485) is part of the Seamless family of models developed by Meta AI. It was pre-trained on 4.5M hours of unlabeled audio data across over 143 languages and requires fine-tuning for tasks like Automatic Speech Recognition (ASR) or Audio Classification. The model is foundational for SeamlessM4T v2, which includes an updated UnitY2 framework and additional aligned data. SeamlessExpressive and SeamlessStreaming are built on this foundation, with SeamlessExpressive focusing on preserving vocal styles and prosody, and SeamlessStreaming enabling low-latency, simultaneous speech-to-speech/text translation. The models are evaluated using both automatic and human metrics, and safety measures like red-teaming, toxicity detection, and watermarking are implemented. Seamless, combining elements from SeamlessExpressive and SeamlessStreaming, is the first publicly available system for real-time expressive cross-lingual communication.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="hf-audio/wav2vec2-bert-CV16-en", dtype="auto")
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

processor = AutoProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
model = AutoModelForCTC.from_pretrained("hf-audio/wav2vec2-bert-CV16-en", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## Wav2Vec2BertConfig

[[autodoc]] Wav2Vec2BertConfig

## Wav2Vec2BertProcessor

[[autodoc]] Wav2Vec2BertProcessor
    - __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Wav2Vec2BertModel

[[autodoc]] Wav2Vec2BertModel
    - forward

## Wav2Vec2BertForCTC

[[autodoc]] Wav2Vec2BertForCTC
    - forward

## Wav2Vec2BertForSequenceClassification

[[autodoc]] Wav2Vec2BertForSequenceClassification
    - forward

## Wav2Vec2BertForAudioFrameClassification

[[autodoc]] Wav2Vec2BertForAudioFrameClassification
    - forward

## Wav2Vec2BertForXVector

[[autodoc]] Wav2Vec2BertForXVector
    - forward

