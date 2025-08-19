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
*This model was released on 2020-10-11 and added to Hugging Face Transformers on 2022-05-17.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Wav2Vec2-Conformer

[Wav2Vec2-Conformer](https://huggingface.co/papers/2010.05171) is an enhanced version of Wav2Vec2 that replaces the standard attention blocks with Conformer blocks, combining the power of convolution and attention for superior speech recognition performance. It's like upgrading your speech recognition system with a more sophisticated architecture that captures both local and global patterns in audio signals.

You can find all the original [Wav2Vec2-Conformer](https://huggingface.co/models?search=wav2vec2-conformer) checkpoints on the Hugging Face Hub.

> [!TIP]
> Click on the Wav2Vec2-Conformer models in the right sidebar for more examples of how to apply Wav2Vec2-Conformer to different speech recognition and audio classification tasks.

The example below demonstrates how to perform automatic speech recognition using the [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

# Automatic Speech Recognition
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-conformer-large-960h")
transcription = asr("path/to/audio.wav")

# Audio Classification
classifier = pipeline("audio-classification", model="facebook/wav2vec2-conformer-base")
result = classifier("path/to/audio.wav")
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoProcessor, Wav2Vec2ConformerForCTC
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-large-960h")
model = Wav2Vec2ConformerForCTC.from_pretrained("facebook/wav2vec2-conformer-large-960h")

#Prepare audio input (assuming you have audio data)
audio_input = load_audio("path/to/audio.wav")
inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)

Generate logits
with torch.no_grad():
    logits = model(**inputs).logits

Decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) to quantize the weights to 8-bit:

```python
from transformers import Wav2Vec2ConformerForCTC
import torch

model = Wav2Vec2ConformerForCTC.from_pretrained(
    "facebook/wav2vec2-conformer-large-960h",
    load_in_8bit=True,
    device_map="auto"
)
```

## Notes

### Model Architecture
Wav2Vec2-Conformer follows the same architecture as Wav2Vec2 but replaces the standard attention blocks with Conformer blocks as introduced in [Conformer: Convolution-augmented Transformer for Speech Recognition](https://huggingface.co/papers/2005.08100). This enhancement:
- Combines convolution and attention mechanisms
- Captures both local and global patterns in audio
- Requires more parameters than Wav2Vec2 for the same number of layers
- Yields improved word error rates

### Position Embeddings
Wav2Vec2-Conformer supports multiple position embedding types:
- No relative position embeddings
- Transformer-XL-like position embeddings
- Rotary position embeddings

Configure the embedding type using `config.position_embeddings_type`.

### Compatibility
- Uses the same tokenizer and feature extractor as Wav2Vec2
- Compatible with existing Wav2Vec2 pipelines and workflows
- Can be used as a drop-in replacement for Wav2Vec2 models

### Performance
The official results can be found in Table 3 and Table 4 of the [fairseq S2T paper](https://huggingface.co/papers/2010.05171).

### Model Availability
The Wav2Vec2-Conformer weights were released by the Meta AI team within the [Fairseq library](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md#pre-trained-models).

### Note on Wav2Vec2-BERT
Meta (FAIR) released [Wav2Vec2-BERT 2.0](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert) pretrained on 4.5M hours of audio. We especially recommend using it for fine-tuning tasks, as per [this guide](https://huggingface.co/blog/fine-tune-w2v2-bert).

## Resources

- [Audio classification task guide](../tasks/audio_classification)
- [Automatic speech recognition task guide](../tasks/asr)

---

## API Reference

### Wav2Vec2ConformerConfig

[[autodoc]] Wav2Vec2ConformerConfig

### Wav2Vec2Conformer specific outputs

[[autodoc]] models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForPreTrainingOutput

### Wav2Vec2ConformerModel

[[autodoc]] Wav2Vec2ConformerModel
    - forward

### Wav2Vec2ConformerForCTC

[[autodoc]] Wav2Vec2ConformerForCTC
    - forward

### Wav2Vec2ConformerForSequenceClassification

[[autodoc]] Wav2Vec2ConformerForSequenceClassification
    - forward

### Wav2Vec2ConformerForAudioFrameClassification

[[autodoc]] Wav2Vec2ConformerForAudioFrameClassification
    - forward

### Wav2Vec2ConformerForXVector

[[autodoc]] Wav2Vec2ConformerForXVector
    - forward

### Wav2Vec2ConformerForPreTraining

[[autodoc]] Wav2Vec2ConformerForPreTraining
    - forward
