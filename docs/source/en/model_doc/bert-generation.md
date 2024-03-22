<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BertGeneration

## Overview

The BertGeneration model is a BERT model that can be leveraged for sequence-to-sequence tasks using
[`EncoderDecoderModel`] as proposed in [Leveraging Pre-trained Checkpoints for Sequence Generation
Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.

The abstract from the paper is the following:

*Unsupervised pretraining of large neural models has recently revolutionized Natural Language Processing. By
warm-starting from the publicly released checkpoints, NLP practitioners have pushed the state-of-the-art on multiple
benchmarks while saving significant amounts of compute time. So far the focus has been mainly on the Natural Language
Understanding tasks. In this paper, we demonstrate the efficacy of pre-trained checkpoints for Sequence Generation. We
developed a Transformer-based sequence-to-sequence model that is compatible with publicly available pre-trained BERT,
GPT-2 and RoBERTa checkpoints and conducted an extensive empirical study on the utility of initializing our model, both
encoder and decoder, with these checkpoints. Our models result in new state-of-the-art results on Machine Translation,
Text Summarization, Sentence Splitting, and Sentence Fusion.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The original code can be
found [here](https://tfhub.dev/s?module-type=text-generation&subtype=module,placeholder).

## Usage examples and tips

The model can be used in combination with the [`EncoderDecoderModel`] to leverage two pretrained BERT checkpoints for 
subsequent fine-tuning:

```python
>>> # leverage checkpoints for Bert2Bert model...
>>> # use BERT's cls token as BOS token and sep token as EOS token
>>> encoder = BertGenerationEncoder.from_pretrained("google-bert/bert-large-uncased", bos_token_id=101, eos_token_id=102)
>>> # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
>>> decoder = BertGenerationDecoder.from_pretrained(
...     "google-bert/bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
... )
>>> bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

>>> # create tokenizer...
>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")

>>> input_ids = tokenizer(
...     "This is a long article to summarize", add_special_tokens=False, return_tensors="pt"
... ).input_ids
>>> labels = tokenizer("This is a short summary", return_tensors="pt").input_ids

>>> # train...
>>> loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
>>> loss.backward()
```

Pretrained [`EncoderDecoderModel`] are also directly available in the model hub, e.g.:

```python
>>> # instantiate sentence fusion model
>>> sentence_fuser = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
>>> tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

>>> input_ids = tokenizer(
...     "This is the first sentence. This is the second sentence.", add_special_tokens=False, return_tensors="pt"
... ).input_ids

>>> outputs = sentence_fuser.generate(input_ids)

>>> print(tokenizer.decode(outputs[0]))
```

Tips:

- [`BertGenerationEncoder`] and [`BertGenerationDecoder`] should be used in
  combination with [`EncoderDecoder`].
- For summarization, sentence splitting, sentence fusion and translation, no special tokens are required for the input.
  Therefore, no EOS token should be added to the end of the input.

## BertGenerationConfig

[[autodoc]] BertGenerationConfig

## BertGenerationTokenizer

[[autodoc]] BertGenerationTokenizer
    - save_vocabulary

## BertGenerationEncoder

[[autodoc]] BertGenerationEncoder
    - forward

## BertGenerationDecoder

[[autodoc]] BertGenerationDecoder
    - forward
