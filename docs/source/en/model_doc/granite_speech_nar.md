<!--Copyright 2026 IBM and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-03-09 and contributed to Hugging Face Transformers on 2026-07-29.*

# GraniteSpeechNar

## Overview

GraniteSpeechNar is a non-autoregressive (NAR) speech recognition model based on [NLE: Non-autoregressive LLM-based ASR by Transcript Editing](https://huggingface.co/papers/2603.08397). It formulates ASR as conditional transcript editing, achieving fully parallel prediction with significant speedups over autoregressive baselines.

The model consists of:

1. **Conformer Encoder**: A conformer encoder trained with CTC on BPE targets, using block-attention and self-conditioned CTC from the middle layer.

2. **QFormer Projector**: A windowed query-transformer that maps multi-layer encoder features to the LLM embedding space while performing 5× temporal downsampling (15-frame windows → 3 queries). Compared to the BLIP-2 Q-Former ([Li et al., 2023](https://huggingface.co/papers/2301.12597)), the implementation differs in several undocumented ways: no query self-attention, pre-norm non-affine `LayerNorm`s, and queries initialized from learned embeddings plus mean-pooled 5-frame features.

3. **Bidirectional Granite LLM**: A Granite language model with bidirectional (non-causal) attention that refines CTC predictions in a single forward pass.

By default the model performs inference in a single pass: the encoder produces initial CTC predictions, which are interleaved with blank insertion slots (exploiting the identity mapping bias of Transformers) and fed alongside projected audio embeddings to the bidirectional LLM for refinement via a latent alignment objective. This refinement can optionally be repeated for several non-autoregressive editing passes (see `num_editing_steps`).

This model was contributed by [Avihu Dekel](https://huggingface.co/Avihu) and [Eustache Le Bihan](https://huggingface.co/eustlb).

## Usage

```python
from transformers import AutoModelForCTC, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "ibm-granite/granite-speech-4.1-2b-nar"
revision = "refs/pr/6"  # native-format weights; drop once merged to `main`
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
model = AutoModelForCTC.from_pretrained(model_id, revision=revision, device_map="auto")

url = "https://huggingface.co/buckets/huggingface/audio-samples/resolve/mister-quilter.mp3"
audio = load_audio(url, sampling_rate=processor.feature_extractor.sampling_rate)

inputs = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
output = model.generate(**inputs, return_dict_in_generate=True)
print(processor.batch_decode(output.sequences, skip_special_tokens=True))
# ['mrister quilter is the apostle of the middle classes and we are glad to welcome his gospel']
```

### Iterative editing

Pass `num_editing_steps > 1` to refine the transcription over several non-autoregressive editing passes. Each extra pass
feeds the previous CTC-collapsed output back to the bidirectional LLM as text input, reusing the cached audio embeddings
(so the encoder and projector run only once). `num_editing_steps=1` (the default) reproduces the single-pass behavior.

```python
from transformers import AutoModelForCTC, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "ibm-granite/granite-speech-4.1-2b-nar"
revision = "refs/pr/6"  # native-format weights; drop once merged to `main`
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
model = AutoModelForCTC.from_pretrained(model_id, revision=revision, device_map="auto")

url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/monte_cristo.flac"
audio = load_audio(url, sampling_rate=processor.feature_extractor.sampling_rate)

inputs = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)

one_step = model.generate(**inputs, num_editing_steps=1, return_dict_in_generate=True)
two_steps = model.generate(**inputs, num_editing_steps=2, return_dict_in_generate=True)

print(processor.decode(one_step.sequences[0], skip_special_tokens=True))
# "... qui avaient vu ses chevaux emportés comme par un tourbillon"
print(processor.decode(two_steps.sequences[0], skip_special_tokens=True))
# "... qui avaient vu ces chevaux emportés comme par un tourbillon"  # "ses" -> "ces" after a second editing pass
```

## GraniteSpeechNarConfig

[[autodoc]] GraniteSpeechNarConfig

## GraniteSpeechNarEncoderConfig

[[autodoc]] GraniteSpeechNarEncoderConfig

## GraniteSpeechNarEncoderProjectorConfig

[[autodoc]] GraniteSpeechNarEncoderProjectorConfig

## GraniteSpeechNarTextConfig

[[autodoc]] GraniteSpeechNarTextConfig

## GraniteSpeechNarProcessor

[[autodoc]] GraniteSpeechNarProcessor
    - __call__

## GraniteSpeechNarFeatureExtractor

[[autodoc]] GraniteSpeechNarFeatureExtractor

## GraniteSpeechNarModel

[[autodoc]] GraniteSpeechNarModel
    - forward

## GraniteSpeechNarForCTC

[[autodoc]] GraniteSpeechNarForCTC
    - forward
    - generate
