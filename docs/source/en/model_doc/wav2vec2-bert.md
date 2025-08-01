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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Wav2Vec2-BERT

[Wav2Vec2-BERT](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) is a powerful speech model pre-trained on 4.5M hours of unlabeled audio data covering more than 143 languages. It's like having a multilingual speech expert that understands the nuances of human communication across diverse languages and cultures. This model requires fine-tuning for downstream tasks like Automatic Speech Recognition (ASR) or Audio Classification.

You can find all the original [Wav2Vec2-BERT](https://huggingface.co/models?search=wav2vec2-bert) checkpoints on the Hugging Face Hub.

> [!TIP]
> Click on the Wav2Vec2-BERT models in the right sidebar for more examples of how to apply Wav2Vec2-BERT to different speech recognition and audio classification tasks.

The example below demonstrates how to perform automatic speech recognition using the [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

# Automatic Speech Recognition
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-bert-base-960h")
transcription = asr("path/to/audio.wav")

# Audio Classification
classifier = pipeline("audio-classification", model="facebook/wav2vec2-bert-base")
result = classifier("path/to/audio.wav")
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoProcessor, Wav2Vec2BertForCTC
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-bert-base-960h")
model = Wav2Vec2BertForCTC.from_pretrained("facebook/wav2vec2-bert-base-960h")

# Prepare audio input (assuming you have audio data)
# audio_input = load_audio("path/to/audio.wav")
# inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)

# Generate logits
# with torch.no_grad():
#     logits = model(**inputs).logits

# Decode
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)
```

</hfoption>
<hfoption id="transformers-cli">

<!-- No CLI usage available for Wav2Vec2-BERT, so this block is intentionally left empty. -->

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) to quantize the weights to 8-bit:

```python
from transformers import Wav2Vec2BertForCTC
import torch

model = Wav2Vec2BertForCTC.from_pretrained(
    "facebook/wav2vec2-bert-base-960h",
    load_in_8bit=True,
    device_map="auto"
)
```

<!-- Wav2Vec2-BERT does not use token-based attention visualization, so the AttentionMaskVisualizer is not included. -->

<!-- Optionally, you can add a generated waveform image here if available. -->
<div class="flex justify-center">
    <img src=""/>
</div>

## Notes

### Model Architecture
Wav2Vec2-BERT follows the same architecture as Wav2Vec2-Conformer but with key differences:
- Uses a causal depthwise convolutional layer
- Takes mel-spectrogram representation as input instead of raw waveform
- Introduces a Conformer-based adapter network instead of a simple convolutional network

### Position Embeddings
Wav2Vec2-BERT supports multiple position embedding types:
- No relative position embeddings
- Shaw-like position embeddings
- Transformer-XL-like position embeddings
- Rotary position embeddings

Configure the embedding type using `config.position_embeddings_type`.

### Training Data
The model was pre-trained on 4.5M hours of unlabeled audio data covering more than 143 languages, making it highly effective for multilingual applications.

### Fine-tuning Requirements
This model requires fine-tuning for downstream tasks such as:
- Automatic Speech Recognition (ASR)
- Audio Classification
- Other speech-related tasks

### Performance
The official results can be found in Section 3.2.1 of the [Seamless paper](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/).

## Resources

<PipelineTag pipeline="automatic-speech-recognition"/>

- [`Wav2Vec2BertForCTC`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition).
- You can also adapt these notebooks on [how to finetune a speech recognition model in English](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb), and [how to finetune a speech recognition model in any language](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb).

<PipelineTag pipeline="audio-classification"/>

- [`Wav2Vec2BertForSequenceClassification`] can be used by adapting this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification).
- See also: [Audio classification task guide](../tasks/audio_classification)

---

## API Reference

### Wav2Vec2BertConfig

[[autodoc]] Wav2Vec2BertConfig

### Wav2Vec2BertProcessor

[[autodoc]] Wav2Vec2BertProcessor
    - __call__
    - pad
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

### Wav2Vec2BertModel

[[autodoc]] Wav2Vec2BertModel
    - forward

### Wav2Vec2BertForCTC

[[autodoc]] Wav2Vec2BertForCTC
    - forward

### Wav2Vec2BertForSequenceClassification

[[autodoc]] Wav2Vec2BertForSequenceClassification
    - forward

### Wav2Vec2BertForAudioFrameClassification

[[autodoc]] Wav2Vec2BertForAudioFrameClassification
    - forward

### Wav2Vec2BertForXVector

[[autodoc]] Wav2Vec2BertForXVector
    - forward
