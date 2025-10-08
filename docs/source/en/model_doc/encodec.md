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
*This model was released on 2022-10-24 and added to Hugging Face Transformers on 2023-06-14 and contributed by [Matthijs](https://huggingface.co/Matthijs), [patrickvonplaten](https://huggingface.co/patrickvonplaten), and [ArthurZ](https://huggingface.co/ArthurZ).*

# EnCodec

[EnCodec](https://huggingface.co/papers/2210.13438) is a real-time, high-fidelity audio codec using a neural network-based streaming encoder-decoder architecture with a quantized latent space trained end-to-end. It employs a multiscale spectrogram adversary to minimize artifacts and enhance sample quality, and a novel loss balancer mechanism to stabilize training. The model also incorporates lightweight Transformer models for additional compression, achieving up to 40% reduction while maintaining real-time processing speed. The design includes detailed descriptions of the training objective, architectural modifications, and perceptual loss functions. Extensive subjective evaluations (MUSHRA tests) and ablation studies across various bandwidths and audio domains (speech, noisy-reverberant speech, music) demonstrate superior performance compared to baseline methods for both 24 kHz monophonic and 48 kHz stereophonic audio.

<hfoptions id="usage">
<hfoption id="EncodecModel">

```python 
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[-1]["audio"]["array"]
inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

model = EncodecModel.from_pretrained("facebook/encodec_24khz", dtype="auto")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
audio_values = model.decode(**encoder_outputs, padding_mask=inputs["padding_mask"])[0]
audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

</hfoption>
</hfoptions>

## EncodecConfig

[[autodoc]] EncodecConfig

## EncodecFeatureExtractor

[[autodoc]] EncodecFeatureExtractor
    - __call__

## EncodecModel

[[autodoc]] EncodecModel
    - decode
    - encode
    - forward

