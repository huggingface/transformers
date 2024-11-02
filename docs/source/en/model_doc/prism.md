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

# PRISM

## Overview

The `Prism` model, a state-of-the-art multilingual neural machine translation (NMT) system developed for translation. The model supports translation across 39 languages, leveraging a zero-shot paraphrasing approach that does not require human judgments for training.

The `Prism` model was designed to be a lexically/syntactically unbiased paraphraser. The core idea is to treat paraphrasing as a zero-shot translation task, which allows the model to cover a wide range of languages effectively.

The model was proposed in [Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing](https://aclanthology.org/2020.emnlp-main.8.pdf) by Brian Thompson and Matt Post.

The abstract from the paper is the following:

*We frame the task of machine translation evaluation as one of scoring machine translation output with a sequence-to-sequence paraphraser, conditioned on a human reference. Wepropose training the paraphraser as a multilingual NMT system, treating paraphrasing as a zero-shot translation task (e.g., Czech to Czech). This results in the paraphraser’s output mode being centered around a copy of the input sequence, which represents the best case scenario where the MT system output matches a human reference. Our method is simple and intuitive, and does not require human judgements for training. Our single model (trained in 39 languages) outperforms or statistically ties with all prior metrics on the WMT 2019 segment-level shared metrics task in all languages (excluding Gujarati where the model had no training data). We also explore using our model for the task of quality estimation as a metric—conditioning on the source instead of the reference—and find that it significantly outperforms every submission to the WMT2019 shared task on quality estimation in every language pair.*

This model was contributed by [dariast](https://huggingface.co/dariast/).
The original code can be found [here](https://github.com/thompsonb/prism/tree/master) and the original documentation is found [here](https://github.com/thompsonb/prism/blob/master/translation/README.md).


## Usage tips

To use `PrismTokenizer`, ensure that the `sentencepiece` package is installed, as it is a required dependency for handling multilingual tokenization.

```bash
pip install sentencepiece
```

## Example
```python
from transformers import PrismForConditionalGeneration, PrismTokenizer

uk_text = "Життя як коробка шоколаду"
ja_text = "人生はチョコレートの箱のようなもの。"

model = PrismForConditionalGeneration.from_pretrained("dariast/prism")
tokenizer = PrismTokenizer.from_pretrained("dariast/prism")

# Translate Ukrainian to French
tokenizer.src_lang = "uk"
encoded_uk = tokenizer(uk_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_uk, forced_bos_token_id=tokenizer.get_lang_id("fr"), max_new_tokens=20)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => 'La vie comme une boîte de chocolat.'

# Translate Japanese to English
tokenizer.src_lang = "ja"
encoded_ja = tokenizer(ja_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_ja, forced_bos_token_id=tokenizer.get_lang_id("en"), max_new_tokens=20)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => 'Life is like a box of chocolate.'
```

## Languages Covered
Albanian (sq), Arabic (ar), Bengali (bn), Bulgarian (bg), Catalan; Valencian (ca), Chinese (zh), Croatian (hr), Czech (cs), Danish (da), Dutch (nl), English (en), Esperanto (eo), Estonian (et), Finnish (fi), French (fr), German (de), Greek, Modern (el), Hebrew (modern) (he), Hungarian (hu), Indonesian (id), Italian (it), Japanese (ja), Kazakh (kk), Latvian (lv), Lithuanian (lt), Macedonian (mk), Norwegian (no), Polish (pl), Portuguese (pt), Romanian, Moldovan (ro), Russian (ru), Serbian (sr), Slovak (sk), Slovene (sl), Spanish; Castilian (es), Swedish (sv), Turkish (tr), Ukrainian (uk), Vietnamese (vi).


## Resources

- [Translation task guide](../tasks/translation)

## PrismConfig

[[autodoc]] PrismConfig

## PrismTokenizer

[[autodoc]] PrismTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## PrismModel

[[autodoc]] PrismModel
    - forward

## PrismForConditionalGeneration

[[autodoc]] PrismForConditionalGeneration
    - forward

## Using Flash Attention 2

Flash Attention 2 is a faster, optimized version of the attention scores computation which relies on `cuda` kernels.

### Installation 

First, check whether your hardware is compatible with Flash Attention 2. The latest list of compatible hardware can be found in the [official documentation](https://github.com/Dao-AILab/flash-attention#installation-and-features).

Next, [install](https://github.com/Dao-AILab/flash-attention#installation-and-features) the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```
