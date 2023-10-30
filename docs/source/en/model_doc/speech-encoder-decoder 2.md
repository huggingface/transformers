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

# Speech Encoder Decoder Models

The [`SpeechEncoderDecoderModel`] can be used to initialize a speech-to-text model
with any pretrained speech autoencoding model as the encoder (*e.g.* [Wav2Vec2](wav2vec2), [Hubert](hubert)) and any pretrained autoregressive model as the decoder.

The effectiveness of initializing speech-sequence-to-text-sequence models with pretrained checkpoints for speech
recognition and speech translation has *e.g.* been shown in [Large-Scale Self- and Semi-Supervised Learning for Speech
Translation](https://arxiv.org/abs/2104.06678) by Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli,
Alexis Conneau.

An example of how to use a [`SpeechEncoderDecoderModel`] for inference can be seen in [Speech2Text2](speech_to_text_2).

## Randomly initializing `SpeechEncoderDecoderModel` from model configurations.

[`SpeechEncoderDecoderModel`] can be randomly initialized from an encoder and a decoder config. In the following example, we show how to do this using the default [`Wav2Vec2Model`] configuration for the encoder
and the default [`BertForCausalLM`] configuration for the decoder.

```python
>>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

>>> config_encoder = Wav2Vec2Config()
>>> config_decoder = BertConfig()

>>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = SpeechEncoderDecoderModel(config=config)
```

## Initialising `SpeechEncoderDecoderModel` from a pretrained encoder and a pretrained decoder.

[`SpeechEncoderDecoderModel`] can be initialized from a pretrained encoder checkpoint and a pretrained decoder checkpoint. Note that any pretrained Transformer-based speech model, *e.g.* [Wav2Vec2](wav2vec2), [Hubert](hubert) can serve as the encoder and both pretrained auto-encoding models, *e.g.* BERT, pretrained causal language models, *e.g.* GPT2, as well as the pretrained decoder part of sequence-to-sequence models, *e.g.* decoder of BART, can be used as the decoder.
Depending on which architecture you choose as the decoder, the cross-attention layers might be randomly initialized.
Initializing [`SpeechEncoderDecoderModel`] from a pretrained encoder and decoder checkpoint requires the model to be fine-tuned on a downstream task, as has been shown in [the *Warm-starting-encoder-decoder blog post*](https://huggingface.co/blog/warm-starting-encoder-decoder).
To do so, the `SpeechEncoderDecoderModel` class provides a [`SpeechEncoderDecoderModel.from_encoder_decoder_pretrained`] method.

```python
>>> from transformers import SpeechEncoderDecoderModel

>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "facebook/hubert-large-ll60k", "bert-base-uncased"
... )
```

## Loading an existing `SpeechEncoderDecoderModel` checkpoint and perform inference.

To load fine-tuned checkpoints of the `SpeechEncoderDecoderModel` class, [`SpeechEncoderDecoderModel`] provides the `from_pretrained(...)` method just like any other model architecture in Transformers.

To perform inference, one uses the [`generate`] method, which allows to autoregressively generate text. This method supports various forms of decoding, such as greedy, beam search and multinomial sampling.

```python
>>> from transformers import Wav2Vec2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> # load a fine-tuned speech translation model and corresponding processor
>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
>>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

>>> # let's perform inference on a piece of English speech (which we'll translate to German)
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> # autoregressively generate transcription (uses greedy decoding by default)
>>> generated_ids = model.generate(input_values)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
Mr. Quilter ist der Apostel der Mittelschicht und wir freuen uns, sein Evangelium willkommen heißen zu können.
```

## Training

Once the model is created, it can be fine-tuned similar to BART, T5 or any other encoder-decoder model on a dataset of (speech, text) pairs.
As you can see, only 2 inputs are required for the model in order to compute a loss: `input_values` (which are the
speech inputs) and `labels` (which are the `input_ids` of the encoded target sequence).

```python
>>> from transformers import AutoTokenizer, AutoFeatureExtractor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset

>>> encoder_id = "facebook/wav2vec2-base-960h"  # acoustic model encoder
>>> decoder_id = "bert-base-uncased"  # text decoder

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
>>> tokenizer = AutoTokenizer.from_pretrained(decoder_id)
>>> # Combine pre-trained encoder and pre-trained decoder to form a Seq2Seq model
>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id)

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> # load an audio input and pre-process (normalise mean/std to 0/1)
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> # load its corresponding transcription and tokenize to generate labels
>>> labels = tokenizer(ds[0]["text"], return_tensors="pt").input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(input_values=input_values, labels=labels).loss
>>> loss.backward()
```

## SpeechEncoderDecoderConfig

[[autodoc]] SpeechEncoderDecoderConfig

## SpeechEncoderDecoderModel

[[autodoc]] SpeechEncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained

## FlaxSpeechEncoderDecoderModel

[[autodoc]] FlaxSpeechEncoderDecoderModel
    - __call__
    - from_encoder_decoder_pretrained
