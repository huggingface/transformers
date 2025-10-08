<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-10-11 and added to Hugging Face Transformers on 2022-05-17 and contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).*

# Wav2Vec2-Conformer

[Wav2Vec2-Conformer](https://huggingface.co/papers/2010.05171) is an extension of the fairseq framework for speech-to-text tasks, including end-to-end speech recognition and speech-to-text translation. It supports scalable, extensible workflows covering data preprocessing, model training, and both offline and online inference. The framework implements state-of-the-art RNN, Transformer, and Conformer architectures, with open-source training recipes provided. Additionally, fairseq’s existing machine translation and language models can be integrated for multi-task or transfer learning.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="facebook/wav2vec2-conformer-rel-pos-large", dtype="auto")
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

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
```

</hfoption>
</hfoptions>

## Wav2Vec2ConformerConfig

[[autodoc]] Wav2Vec2ConformerConfig

## Wav2Vec2Conformer specific outputs

[[autodoc]] models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForPreTrainingOutput

## Wav2Vec2ConformerModel

[[autodoc]] Wav2Vec2ConformerModel
    - forward

## Wav2Vec2ConformerForCTC

[[autodoc]] Wav2Vec2ConformerForCTC
    - forward

## Wav2Vec2ConformerForSequenceClassification

[[autodoc]] Wav2Vec2ConformerForSequenceClassification
    - forward

## Wav2Vec2ConformerForAudioFrameClassification

[[autodoc]] Wav2Vec2ConformerForAudioFrameClassification
    - forward

## Wav2Vec2ConformerForXVector

[[autodoc]] Wav2Vec2ConformerForXVector
    - forward

## Wav2Vec2ConformerForPreTraining

[[autodoc]] Wav2Vec2ConformerForPreTraining
    - forward
