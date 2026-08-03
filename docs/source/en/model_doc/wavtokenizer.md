<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2024-08-29 and contributed to Hugging Face Transformers on 2026-08-03.*

# WavTokenizer

## Overview

WavTokenizer is a discrete acoustic codec tokenizer proposed in [WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling](https://huggingface.co/papers/2408.16532) by Shengpeng Ji, Ziyue Jiang, Wen Wang, Yifu Chen, Minghui Fang, Jialong Zuo, Qian Yang, Xize Cheng, Zehan Wang, Ruiqi Li, Ziang Zhang, Xiaoda Yang, Rongjie Huang, Yidi Jiang, Qian Chen, Siqi Zheng, Zhou Zhao (ICLR 2025).

WavTokenizer compresses 24 kHz audio into a **single codebook** of discrete tokens at an extremely low frame rate
(40 or 75 tokens per second), making it well suited as an audio tokenizer for language models. The encoder and
quantizer follow [EnCodec](encodec)'s SEANet encoder with a single-codebook vector quantization; the decoder is a
[Vocos](https://huggingface.co/papers/2306.00814)-style backbone (ConvNeXt blocks and a positional conv/attention net) with
an inverse STFT head.

The 40 tokens/s variant of WavTokenizer is used as the audio tokenizer of [Apertus 1.5](./apertus1p5).

The abstract from the paper is the following:

*Language models have been effectively applied to modeling natural signals, such as images, video, speech, and audio. A crucial component of these models is the codec tokenizer, which compresses high-dimensional natural signals into lower-dimensional discrete tokens. In this paper, we introduce WavTokenizer, which offers several advantages over previous SOTA acoustic codec models in the audio domain: 1) extreme compression. By compressing the layers of quantizers and the temporal dimension of the discrete codec, one-second audio of 24kHz sampling rate requires only a single quantizer with 40 or 75 tokens. 2) improved subjective quality. Despite the reduced number of tokens, WavTokenizer achieves state-of-the-art reconstruction quality with outstanding UTMOS scores and inherently contains richer semantic information.*

This port was contributed as part of the Apertus 1.5 integration by the
[SwissAI initiative](https://huggingface.co/swiss-ai). The original code (MIT license) can be found
[here](https://github.com/jishengpeng/WavTokenizer). The implementation is inference-only: it covers encoding audio
into discrete codes and decoding codes back to audio, while the original training stack (GAN discriminators, loss
modules, differentiable quantization) is not ported. The released raw
PyTorch Lightning checkpoints use one of two temporal configurations:

|  Token rate | `upsampling_ratios` | Hop length | ISTFT FFT size |
|------------:|:-------------------:|-----------:|---------------:|
| 40 tokens/s |   `[6, 5, 5, 4]`    |        600 |           2400 |
| 75 tokens/s |   `[8, 5, 4, 2]`    |        320 |           1280 |

The small, medium, large, domain-specific, and v2 releases share the remaining inference architecture.

## Available checkpoints

All released checkpoints are available on the Hub already converted to the Transformers format and can be loaded
directly with `from_pretrained`:

| Checkpoint                                                                                                                        | Domain      | Token rate | Converted from                                                                                                              |
|-----------------------------------------------------------------------------------------------------------------------------------|-------------|-----------:|-----------------------------------------------------------------------------------------------------------------------------|
| [swiss-ai/wavtokenizer-small-speech-40token](https://huggingface.co/swiss-ai/wavtokenizer-small-speech-40token)                   | speech      |       40/s | [novateur/WavTokenizer](https://huggingface.co/novateur/WavTokenizer)                                                       |
| [swiss-ai/wavtokenizer-small-speech-75token](https://huggingface.co/swiss-ai/wavtokenizer-small-speech-75token)                   | speech      |       75/s | [novateur/WavTokenizer](https://huggingface.co/novateur/WavTokenizer)                                                       |
| [swiss-ai/wavtokenizer-medium-speech-75token](https://huggingface.co/swiss-ai/wavtokenizer-medium-speech-75token)                 | speech      |       75/s | [novateur/WavTokenizer-medium-speech-75token](https://huggingface.co/novateur/WavTokenizer-medium-speech-75token)           |
| [swiss-ai/wavtokenizer-medium-speech-75token-v2](https://huggingface.co/swiss-ai/wavtokenizer-medium-speech-75token-v2)           | speech      |       75/s | [novateur/WavTokenizer-medium-speech-75token](https://huggingface.co/novateur/WavTokenizer-medium-speech-75token)           |
| [swiss-ai/wavtokenizer-medium-music-audio-75token](https://huggingface.co/swiss-ai/wavtokenizer-medium-music-audio-75token)       | music/audio |       75/s | [novateur/WavTokenizer-medium-music-audio-75token](https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token) |
| [swiss-ai/wavtokenizer-medium-music-audio-75token-v2](https://huggingface.co/swiss-ai/wavtokenizer-medium-music-audio-75token-v2) | music/audio |       75/s | [novateur/WavTokenizer-medium-music-audio-75token](https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token) |
| [swiss-ai/wavtokenizer-large-unify-40token](https://huggingface.co/swiss-ai/wavtokenizer-large-unify-40token)                     | unified     |       40/s | [novateur/WavTokenizer-large-unify-40token](https://huggingface.co/novateur/WavTokenizer-large-unify-40token)               |
| [swiss-ai/wavtokenizer-large-speech-75token-v2](https://huggingface.co/swiss-ai/wavtokenizer-large-speech-75token-v2)             | speech      |       75/s | [novateur/WavTokenizer-large-speech-75token](https://huggingface.co/novateur/WavTokenizer-large-speech-75token)             |

## Converting custom checkpoints

Custom or self-trained original-format checkpoints can be converted with the conversion script. It infers the
temporal configuration from checkpoint tensor shapes and saves it in `config.json`. Subsequent `from_pretrained`
calls construct the model from that saved configuration and do not infer architecture from the weights.

```bash
python src/transformers/models/wavtokenizer/convert_wavtokenizer_checkpoint.py \
    --checkpoint_path /path/to/original.ckpt \
    --output_dir /path/to/converted-model
```

A converted checkpoint is also the audio half of an [Apertus 1.5](./apertus1p5) composite; see that page for
the full conversion path.

## Usage example

```python
import torch
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, WavTokenizerModel

model = WavTokenizerModel.from_pretrained("swiss-ai/wavtokenizer-large-unify-40token")
feature_extractor = AutoFeatureExtractor.from_pretrained("swiss-ai/wavtokenizer-large-unify-40token")

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio = dataset[0]["audio"]["array"]

inputs = feature_extractor(audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
with torch.no_grad():
    audio_codes = model.encode(**inputs).audio_codes  # (batch, 1, ceil(samples / 600))
    reconstruction = model.decode(audio_codes).audio_values
```

Decoded audio is always returned in `float32` (the ISTFT head upcasts internally). A single code is supported and
decodes to one configuration-dependent hop (600 samples for 40 tokens/s or 320 samples for 75 tokens/s).

> [!WARNING]
> Load and run the tokenizer in `float32` (the default). Code assignment is a nearest-neighbour argmin over the
> codebook, so half precision (`dtype=torch.bfloat16`/`float16`) flips a large fraction of codes near decision
> boundaries (~66% disagreement vs. `float32` measured in bf16) and breaks reproducible tokenization.

## WavTokenizerConfig

[[autodoc]] WavTokenizerConfig

## WavTokenizerFeatureExtractor

[[autodoc]] WavTokenizerFeatureExtractor
    - __call__

## WavTokenizerModel

[[autodoc]] WavTokenizerModel
    - decode
    - encode
    - forward
