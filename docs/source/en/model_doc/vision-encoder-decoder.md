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
*This model was released on 2021-09-21 and added to Hugging Face Transformers on 2021-10-13.*

# Vision Encoder Decoder Models

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The [`VisionEncoderDecoderModel`] can be used to initialize an image-to-text model with any
pretrained Transformer-based vision model as the encoder (*e.g.* [ViT](vit), [BEiT](beit), [DeiT](deit), [Swin](swin))
and any pretrained language model as the decoder (*e.g.* [RoBERTa](roberta), [GPT2](gpt2), [BERT](bert), [DistilBERT](distilbert)).

The effectiveness of initializing image-to-text-sequence models with pretrained checkpoints has been shown in (for
example) [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://huggingface.co/papers/2109.10282) by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang,
Zhoujun Li, Furu Wei.

After such a [`VisionEncoderDecoderModel`] has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples below
for more information).

An example application is image captioning, in which the encoder is used to encode the image, after which an autoregressive language model generates
the caption. Another example is optical character recognition. Refer to [TrOCR](trocr), which is an instance of [`VisionEncoderDecoderModel`].

## Randomly initializing `VisionEncoderDecoderModel` from model configurations.

[`VisionEncoderDecoderModel`] can be randomly initialized from an encoder and a decoder config. In the following example, we show how to do this using the default [`ViTModel`] configuration for the encoder
and the default [`BertForCausalLM`] configuration for the decoder.

```python
>>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

>>> config_encoder = ViTConfig()
>>> config_decoder = BertConfig()

>>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = VisionEncoderDecoderModel(config=config)
```

## Initialising `VisionEncoderDecoderModel` from a pretrained encoder and a pretrained decoder.

[`VisionEncoderDecoderModel`] can be initialized from a pretrained encoder checkpoint and a pretrained decoder checkpoint. Note that any pretrained Transformer-based vision model, *e.g.* [Swin](swin), can serve as the encoder and both pretrained auto-encoding models, *e.g.* BERT, pretrained causal language models, *e.g.* GPT2, as well as the pretrained decoder part of sequence-to-sequence models, *e.g.* decoder of BART, can be used as the decoder.
Depending on which architecture you choose as the decoder, the cross-attention layers might be randomly initialized.
Initializing [`VisionEncoderDecoderModel`] from a pretrained encoder and decoder checkpoint requires the model to be fine-tuned on a downstream task, as has been shown in [the *Warm-starting-encoder-decoder blog post*](https://huggingface.co/blog/warm-starting-encoder-decoder).
To do so, the `VisionEncoderDecoderModel` class provides a [`VisionEncoderDecoderModel.from_encoder_decoder_pretrained`] method.

```python
>>> from transformers import VisionEncoderDecoderModel

>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "microsoft/swin-base-patch4-window7-224-in22k", "google-bert/bert-base-uncased"
... )
```

## Loading an existing `VisionEncoderDecoderModel` checkpoint and perform inference.

To load fine-tuned checkpoints of the `VisionEncoderDecoderModel` class, [`VisionEncoderDecoderModel`] provides the `from_pretrained(...)` method just like any other model architecture in Transformers.

To perform inference, one uses the [`generate`] method, which allows to autoregressively generate text. This method supports various forms of decoding, such as greedy, beam search and multinomial sampling.

```python
>>> import requests
>>> from PIL import Image

>>> from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

>>> # load a fine-tuned image captioning model and corresponding tokenizer and image processor
>>> model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> # let's perform inference on an image
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> # autoregressively generate caption (uses greedy decoding by default)
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
a cat laying on a blanket next to a cat laying on a bed
```

## Training

Once the model is created, it can be fine-tuned similar to BART, T5 or any other encoder-decoder model on a dataset of (image, text) pairs.
As you can see, only 2 inputs are required for the model in order to compute a loss: `pixel_values` (which are the
images) and `labels` (which are the `input_ids` of the encoded target sequence).

```python
>>> from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
>>> from datasets import load_dataset

>>> image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
... )

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> labels = tokenizer(
...     "an image of two cats chilling on a couch",
...     return_tensors="pt",
... ).input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(pixel_values=pixel_values, labels=labels).loss
```

This model was contributed by [nielsr](https://github.com/nielsrogge).

## VisionEncoderDecoderConfig

[[autodoc]] VisionEncoderDecoderConfig

## VisionEncoderDecoderModel

[[autodoc]] VisionEncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained
