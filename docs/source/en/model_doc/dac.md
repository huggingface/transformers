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
*This model was released on 2023-06-11 and added to Hugging Face Transformers on 2024-08-19 and contributed by [kamilakesbi](https://huggingface.co/kamilakesbi).*

# DAC

[DAC](https://huggingface.co/papers/2306.06546) is a high-fidelity universal neural audio compression algorithm that compresses 44.1 KHz audio into tokens at 8kbps bandwidth, achieving approximately 90x compression. It combines advancements in high-fidelity audio generation with improved vector quantization techniques from the image domain, enhanced adversarial and reconstruction losses, and a single universal model for various audio domains including speech, environment, and music. This method outperforms competing audio compression algorithms and is supported by open-source code and trained model weights.

<hfoptions id="usage">
<hfoption id="DacModel">

```py
from datasets import load_dataset, Audio
from transformers import DacModel, AutoProcessor

model = DacModel.from_pretrained("descript/dac_16khz", dtype="auto")
processor = AutoProcessor.from_pretrained("descript/dac_16khz")

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[-1]["audio"]["array"]
inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

encoder_outputs = model.encode(inputs["input_values"])
audio_codes = encoder_outputs.audio_codes
audio_values = model.decode(encoder_outputs.quantized_representation)
audio_values = model(inputs["input_values"]).audio_values
```

</hfoption>
</hfoptions>

## DacConfig

[[autodoc]] DacConfig

## DacFeatureExtractor

[[autodoc]] DacFeatureExtractor
    - __call__

## DacModel

[[autodoc]] DacModel
    - decode
    - encode
    - forward

