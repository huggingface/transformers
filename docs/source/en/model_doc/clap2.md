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


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-12.*

# CLAP2

## Overview

CLAP2 was proposed in [LAION-Audio-XL: A Web-Scale Corpus and Scaling Studies for Contrastive Language-Audio Pretraining](https://huggingface.co/papers/2611.11592). It is a family of contrastive audio-text models from LAION, trained on [LAION-Audio-XL](https://huggingface.co/datasets/laion/laion-audio-xl). It embeds audio clips and texts in a shared space for zero-shot audio classification and audio-text retrieval.

The audio encoder is a NaFlex vision transformer, as introduced by [SigLIP2](./siglip2), on log-mel patches: a clip of any length becomes a sequence of 16x16 patches of a 128-bin log-mel spectrogram, with 2D axial rotary position embeddings over frequency and time, a register token, SwiGLU MLPs with layer scale, and attention pooling. The text encoder is based on [ModernBERT](./modernbert), with SwiGLU and RMSNorm instead of GeGLU and LayerNorm: a bidirectional transformer over the GPT-2 vocabulary with rotary position embeddings, QK normalization and attention pooling.

The checkpoints come in five sizes trained for 1B samples on [LAION-Audio-XL](https://huggingface.co/datasets/laion/laion-audio-xl): [xxs](https://huggingface.co/laion/clap2-xxs-audioxl-s1B), [xs](https://huggingface.co/laion/clap2-xs-audioxl-s1B), [small](https://huggingface.co/laion/clap2-small-audioxl-s1B), [medium](https://huggingface.co/laion/clap2-medium-audioxl-s1B) and [large](https://huggingface.co/laion/clap2-large-audioxl-s1B).

This model was contributed by [LAION](https://huggingface.co/laion). The original code can be found in [OpenCLIP](https://github.com/mlfoundations/open_clip).

## Usage examples

CLAP2 embeds audio clips and texts in a shared space: the cosine similarity of an audio and a text embedding scores how well the text describes the clip. [Sentence Transformers](https://sbert.net) (`>=6.2.0`) handles the preprocessing and the similarity. With [`AutoModel`], use [`AutoProcessor`] and the `audio_embeds` and `text_embeds` of the model outputs. The feature extractor does not resample, so pass mono audio at 32 kHz.

<hfoptions id="usage">
<hfoption id="Sentence Transformers">

```bash
pip install -U sentence-transformers
```

```python
from datasets import Audio, load_dataset
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("laion/clap2-small-audioxl-s1B")

dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=32000))
audio = dataset[0]["audio"]
texts = ["A dog barks.", "Rain falls steadily.", "A vacuum cleaner is running."]

audio_embeddings = model.encode([audio])
text_embeddings = model.encode(texts)
print(audio_embeddings.shape, text_embeddings.shape)
# (1, 512) (3, 512)

# (1, 3) matrix of cosine similarities
print(model.similarity(audio_embeddings, text_embeddings))
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from datasets import Audio, load_dataset
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("laion/clap2-small-audioxl-s1B")
processor = AutoProcessor.from_pretrained("laion/clap2-small-audioxl-s1B")

dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=32000))
audio = dataset[0]["audio"]["array"]
texts = ["A dog barks.", "Rain falls steadily.", "A vacuum cleaner is running."]

inputs = processor(text=texts, audio=audio, sampling_rate=32000, padding=True, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

# (1, 3) matrix of cosine similarities
print(outputs.audio_embeds @ outputs.text_embeds.T)

# zero-shot classification: probabilities over the texts
print(outputs.logits_per_audio.softmax(dim=-1))
```

</hfoption>
</hfoptions>

## Clap2Config

[[autodoc]] Clap2Config

## Clap2TextConfig

[[autodoc]] Clap2TextConfig

## Clap2AudioConfig

[[autodoc]] Clap2AudioConfig

## Clap2FeatureExtractor

[[autodoc]] Clap2FeatureExtractor
    - __call__

## Clap2Processor

[[autodoc]] Clap2Processor

## Clap2Model

[[autodoc]] Clap2Model
    - forward
    - get_text_features
    - get_audio_features

## Clap2TextModel

[[autodoc]] Clap2TextModel
    - forward

## Clap2AudioModel

[[autodoc]] Clap2AudioModel
    - forward
