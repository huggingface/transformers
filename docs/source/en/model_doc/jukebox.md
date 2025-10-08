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
*This model was released on 2020-04-30 and added to Hugging Face Transformers on 2023-06-20 and contributed by [ArthurZ](https://huggingface.co/ArthurZ).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2. You can do so by running the following command: pip install -U transformers==4.40.2.

# Jukebox

[Jukebox](https://huggingface.co/papers/2005.00341) generates music with singing in the raw audio domain using a multiscale VQ-VAE to compress audio into discrete codes and autoregressive Transformers to model those codes. The model can produce high-fidelity and diverse songs up to multiple minutes in length, conditioned on artist, genre, and unaligned lyrics. It consists of three decoder-only priors, each with an AudioConditioner module that upsamples outputs to raw audio tokens. Metadata such as artist, genre, and timing are incorporated via start tokens and positional embeddings.

<hfoptions id="usage">
<hfoption id="Jukebox">

```py
import torch
from transformers import AutoTokenizer, JukeboxModel

model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0, dtype="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("openai/jukebox-1b-lyrics")

lyrics = "Cowboys ain't easy to love and they're harder to hold"
artist = "Waylon Jennings"
genre = "Country"
inputs = tokenizer(artist=artist, genres=genre, lyrics=lyrics)
music_tokens = model.ancestral_sample(inputs.input_ids, sample_length=400)

with torch.no_grad():
    model.decode(music_tokens)[:, :10].squeeze(-1)
```

</hfoption>
</hfoptions>

## JukeboxConfig

[[autodoc]] JukeboxConfig

## JukeboxPriorConfig

[[autodoc]] JukeboxPriorConfig

## JukeboxVQVAEConfig

[[autodoc]] JukeboxVQVAEConfig

## JukeboxTokenizer

[[autodoc]] JukeboxTokenizer
    - save_vocabulary

## JukeboxModel

[[autodoc]] JukeboxModel
    - ancestral_sample
    - primed_sample
    - continue_sample
    - upsample
    - _sample

## JukeboxPrior

[[autodoc]] JukeboxPrior
    - sample
    - forward

## JukeboxVQVAE

[[autodoc]] JukeboxVQVAE
    - forward
    - encode
    - decode

