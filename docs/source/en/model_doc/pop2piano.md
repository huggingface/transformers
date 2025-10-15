<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2022-11-02 and added to Hugging Face Transformers on 2023-08-21 and contributed by [susnato](https://huggingface.co/susnato).*

# Pop2Piano

[Pop2Piano](https://huggingface.co/papers/2211.00895) generates piano covers from pop music audio waveforms without requiring melody and chord extraction. It is an encoder-decoder Transformer model that transforms audio waveforms into latent representations and then autoregressively generates token ids representing time, velocity, note, and special tokens. These tokens are decoded into MIDI files, producing plausible piano covers.

<hfoptions id="usage">
<hfoption id="Pop2PianoForConditionalGeneration">

```py
from datasets import load_dataset
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano", dtype="auto")
processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")
inputs = processor(
    audio=ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"], return_tensors="pt"
)
model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
tokenizer_output = processor.batch_decode(
    token_ids=model_output, feature_extractor_output=inputs
)["pretty_midi_objects"][0]
tokenizer_output.write("./Outputs/midi_output.mid")
```

## Usage tips

- Install the 🤗 Transformers library and these third-party modules: `pip install pretty-midi==0.2.9 essentia==2.1b6.dev1034 librosa scipy`. Restart your runtime after installation.
- Pop2Piano uses an Encoder-Decoder architecture similar to T5. It generates MIDI files from audio sequences.
- Different composers in [`Pop2PianoForConditionalGeneration.generate`] produce varied results. Load audio files at 44.1 kHz sampling rate for optimal performance.
- While trained primarily on Korean Pop music, Pop2Piano performs well on Western Pop and Hip Hop songs.

## Pop2PianoConfig

[[autodoc]] Pop2PianoConfig

## Pop2PianoFeatureExtractor

[[autodoc]] Pop2PianoFeatureExtractor
    - __call__

## Pop2PianoForConditionalGeneration

[[autodoc]] Pop2PianoForConditionalGeneration
    - forward
    - generate

## Pop2PianoTokenizer

[[autodoc]] Pop2PianoTokenizer
    - __call__

## Pop2PianoProcessor

[[autodoc]] Pop2PianoProcessor
    - __call__

