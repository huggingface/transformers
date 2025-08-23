<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Speech2Text

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
</div>

## Overview

The Speech2Text model was proposed in [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://huggingface.co/papers/2010.05171) by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino. It's a
transformer-based seq2seq (encoder-decoder) model designed for end-to-end Automatic Speech Recognition (ASR) and Speech
Translation (ST). It uses a convolutional downsampler to reduce the length of speech inputs by 3/4th before they are
fed into the encoder. The model is trained with standard autoregressive cross-entropy loss and generates the
transcripts/translations autoregressively. Speech2Text has been fine-tuned on several datasets for ASR and ST:
[LibriSpeech](http://www.openslr.org/12), [CoVoST 2](https://github.com/facebookresearch/covost), [MuST-C](https://ict.fbk.eu/must-c/).

This model was contributed by [valhalla](https://huggingface.co/valhalla). The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text).

## Inference

Speech2Text is a speech model that accepts a float tensor of log-mel filter-bank features extracted from the speech
signal. It's a transformer-based seq2seq model, so the transcripts/translations are generated autoregressively. The
`generate()` method can be used for inference.

The [`Speech2TextFeatureExtractor`] class is responsible for extracting the log-mel filter-bank
features. The [`Speech2TextProcessor`] wraps [`Speech2TextFeatureExtractor`] and
[`Speech2TextTokenizer`] into a single instance to both extract the input features and decode the
predicted token ids.

The feature extractor depends on `torchaudio` and the tokenizer depends on `sentencepiece` so be sure to
install those packages before running the examples. You could either install those as extra speech dependencies with
`pip install transformers"[speech, sentencepiece]"` or install the packages separately with `pip install torchaudio sentencepiece`. Also `torchaudio` requires the development version of the [libsndfile](http://www.mega-nerd.com/libsndfile/) package which can be installed via a system package manager. On Ubuntu it can
be installed as follows: `apt install libsndfile1-dev`

- ASR and Speech Translation

```python
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> transcription
['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel']
```

- Multilingual speech translation

  For multilingual speech translation models, `eos_token_id` is used as the `decoder_start_token_id` and
  the target language id is forced as the first generated token. To force the target language id as the first
  generated token, pass the `forced_bos_token_id` parameter to the `generate()` method. The following
  example shows how to translate English speech to French text using the *facebook/s2t-medium-mustc-multilingual-st*
  checkpoint.

```python
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(
...     inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"],
... )

>>> translation = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> translation
["(Vidéo) Si M. Kilder est l'apossible des classes moyennes, et nous sommes heureux d'être accueillis dans son évangile."]
```

See the [model hub](https://huggingface.co/models?filter=speech_to_text) to look for Speech2Text checkpoints.

## Speech2TextConfig

[[autodoc]] Speech2TextConfig

## Speech2TextTokenizer

[[autodoc]] Speech2TextTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## Speech2TextFeatureExtractor

[[autodoc]] Speech2TextFeatureExtractor
    - __call__

## Speech2TextProcessor

[[autodoc]] Speech2TextProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

<frameworkcontent>
<pt>

## Speech2TextModel

[[autodoc]] Speech2TextModel
    - forward

## Speech2TextForConditionalGeneration

[[autodoc]] Speech2TextForConditionalGeneration
    - forward

</pt>
<tf>

## TFSpeech2TextModel

[[autodoc]] TFSpeech2TextModel
    - call

## TFSpeech2TextForConditionalGeneration

[[autodoc]] TFSpeech2TextForConditionalGeneration
    - call

</tf>
</frameworkcontent>
