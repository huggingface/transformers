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
*This model was contributed to Hugging Face Transformers on 2026-08-25.*

# GraniteSpeech5

## Overview

Granite Speech 5.0 Turbo CTC is a lightweight (~470M parameters) conformer encoder for automatic speech recognition, trained with Connectionist Temporal Classification (CTC) on BPE targets. It is a fast, encoder-only member of the [Granite Speech](https://huggingface.co/papers/2505.08699) family: transcription requires a single forward pass followed by greedy CTC decoding, with no autoregressive decoder.

Architecturally, it extends the Granite Speech conformer CTC encoder with:

1. **Frame stacking + block-wise time subsampling**: the feature extractor stacks pairs of log-mel(+delta) frames (2x), and the first two conformer blocks each subsample time by 2 through a stride-2 depthwise convolution (with a mean-pooled residual), for a total 8x time reduction at 10 ms mel hop.

2. **Block attention with Shaw's relative positional embeddings**: attention is computed over fixed-size blocks (the sequence is right-padded to a whole number of blocks, with padded frames masked out), using separate bias-free query/key/value projections.

3. **Self-conditioned CTC**: the CTC posteriors of the middle layer are projected and fed back into the hidden states, and the CTC head is shared between this mid-layer self-conditioning and the final prediction.

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb).

## Usage

### `GraniteSpeech5ForCTC` usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline


pipe = pipeline("automatic-speech-recognition", model="ibm-granite/granite-speech-5.0-470m-turboctc")
out = pipe("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
print(out)
# {'text': 'yesterday it was 35 degrees in barcelona but today the temperature will go down to -20 degrees'}
```

</hfoption>
<hfoption id="AutoModel">

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor

model_id = "ibm-granite/granite-speech-5.0-470m-turboctc"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]

# `device` computes the log-mel front-end on the model's accelerator, saving a host-to-device copy
inputs = processor(
    speech_samples, sampling_rate=processor.feature_extractor.sampling_rate, device=model.device
)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(processor.batch_decode(outputs, skip_special_tokens=True))
# ['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel', ...]
```

</hfoption>
</hfoptions>

## GraniteSpeech5CTCConfig

[[autodoc]] GraniteSpeech5CTCConfig

## GraniteSpeech5EncoderConfig

[[autodoc]] GraniteSpeech5EncoderConfig

## GraniteSpeech5FeatureExtractor

[[autodoc]] GraniteSpeech5FeatureExtractor

## GraniteSpeech5Processor

[[autodoc]] GraniteSpeech5Processor

## GraniteSpeech5Encoder

[[autodoc]] GraniteSpeech5Encoder
    - forward

## GraniteSpeech5ForCTC

[[autodoc]] GraniteSpeech5ForCTC
    - forward
    - generate
