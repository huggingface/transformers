<!--Copyright 2025 The NVIDIA NeMo Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-10-07 and contributed by [nithinraok](https://huggingface.co/nithinraok), [eustlb](https://huggingface.co/eustlb), and [bezzam](https://huggingface.co/bezzam).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Parakeet

[Parakeet](https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/) is an ASR (automatic speech recognition) model family built on Fast Conformer, an optimized Conformer architecture featuring 8× depthwise-separable convolutional downsampling, modified kernel sizes, and an efficient subsampling module. It supports end-to-end training with either RNNT or CTC decoders and uses limited context attention with global tokens to enable efficient inference on extremely long audio—up to 13 hours—in a single pass on an NVIDIA A100 80GB GPU. The models achieve strong performance-speed tradeoffs, with CTC variants offering especially fast inference (RTF ≈ 2×10⁻³ for 30 s audio). Overall, Parakeet balances accuracy, scalability, and memory efficiency for long-form transcription tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("automatic-speech-recognition", model="nvidia/parakeet-ctc-1.1b")
pipeline("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio

processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b", dtype="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(f"Transcription: {processor.batch_decode(outputs)}")

```

</hfoption>
</hfoptions>

## ParakeetTokenizerFast

[[autodoc]] ParakeetTokenizerFast 

## ParakeetFeatureExtractor

[[autodoc]] ParakeetFeatureExtractor
    - __call__

## ParakeetProcessor

[[autodoc]] ParakeetProcessor
    - __call__
    - batch_decode
    - decode

## ParakeetEncoderConfig

[[autodoc]] ParakeetEncoderConfig 

## ParakeetCTCConfig

[[autodoc]] ParakeetCTCConfig 

## ParakeetEncoder

[[autodoc]] ParakeetEncoder

## ParakeetForCTC

[[autodoc]] ParakeetForCTC

