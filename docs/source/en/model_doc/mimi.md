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
*This model was released on 2024-09-17 and added to Hugging Face Transformers on 2024-09-18.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Mimi

[Mimi](huggingface.co/papers/2410.00037) is a neural audio codec model with pretrained and quantized variants, designed for efficient speech representation and compression. The model operates at 1.1 kbps with a 12 Hz frame rate and uses a convolutional encoder-decoder architecture combined with a residual vector quantizer of 16 codebooks. Mimi outputs dual token streams i.e. semantic and acoustic to balance linguistic richness with high fidelity reconstruction. Key features include a causal streaming encoder for low-latency use, dual-path tokenization for flexible downstream generation, and integration readiness with large speech models like Moshi.

You can find the original Mimi checkpoints under the [Kyutai](https://huggingface.co/kyutai/models?search=mimi) organization.

>[!TIP]
> This model was contributed by [ylacombe](https://huggingface.co/ylacombe).
>
> Click on the Mimi models in the right sidebar for more examples of how to apply Mimi.

The example below demonstrates how to encode and decode audio with the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```python 
>>> from datasets import load_dataset, Audio
>>> from transformers import MimiModel, AutoFeatureExtractor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> # load model and feature extractor
>>> model = MimiModel.from_pretrained("kyutai/mimi")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

>>> # load audio sample
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
>>> audio_values = model.decode(encoder_outputs.audio_codes, inputs["padding_mask"])[0]
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

</hfoption>
</hfoptions>

## MimiConfig

[[autodoc]] MimiConfig

## MimiModel

[[autodoc]] MimiModel
    - decode
    - encode
    - forward
