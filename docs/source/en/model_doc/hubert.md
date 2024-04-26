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

# Hubert

## Overview

Hubert was proposed in [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan
Salakhutdinov, Abdelrahman Mohamed.

The abstract from the paper is the following:

*Self-supervised approaches for speech representation learning are challenged by three unique problems: (1) there are
multiple sound units in each input utterance, (2) there is no lexicon of input sound units during the pre-training
phase, and (3) sound units have variable lengths with no explicit segmentation. To deal with these three problems, we
propose the Hidden-Unit BERT (HuBERT) approach for self-supervised speech representation learning, which utilizes an
offline clustering step to provide aligned target labels for a BERT-like prediction loss. A key ingredient of our
approach is applying the prediction loss over the masked regions only, which forces the model to learn a combined
acoustic and language model over the continuous inputs. HuBERT relies primarily on the consistency of the unsupervised
clustering step rather than the intrinsic quality of the assigned cluster labels. Starting with a simple k-means
teacher of 100 clusters, and using two iterations of clustering, the HuBERT model either matches or improves upon the
state-of-the-art wav2vec 2.0 performance on the Librispeech (960h) and Libri-light (60,000h) benchmarks with 10min, 1h,
10h, 100h, and 960h fine-tuning subsets. Using a 1B parameter model, HuBERT shows up to 19% and 13% relative WER
reduction on the more challenging dev-other and test-other evaluation subsets.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).

# Usage tips

- Hubert is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
- Hubert model was fine-tuned using connectionist temporal classification (CTC) so the model output has to be decoded
  using [`Wav2Vec2CTCTokenizer`].


## Using Flash Attention 2

Flash Attention 2 is an faster, optimized version of the model.

### Installation 

First, check whether your hardware is compatible with Flash Attention 2. The latest list of compatible hardware can be found in the [official documentation](https://github.com/Dao-AILab/flash-attention#installation-and-features). If your hardware is not compatible with Flash Attention 2, you can still benefit from attention kernel optimisations through Better Transformer support covered [above](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

Next, [install](https://github.com/Dao-AILab/flash-attention#installation-and-features) the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

### Usage

Below is an expected speedup diagram comparing the pure inference time between the native implementation in transformers of `facebook/hubert-large-ls960-ft`, the flash-attention-2 and the sdpa (scale-dot-product-attention) version. We show the average speedup obtained on the `librispeech_asr` `clean` validation split: 

```python
>>> from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/hubert-large-ls960-ft", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
...
```

### Expected speedups

Below is an expected speedup diagram comparing the pure inference time between the native implementation in transformers of the `facebook/hubert-large-ls960-ft` model and the flash-attention-2 and sdpa (scale-dot-product-attention) versions. . We show the average speedup obtained on the `librispeech_asr` `clean` validation split: 


<div style="text-align: center">
<img src="https://huggingface.co/datasets/kamilakesbi/transformers_image_doc/resolve/main/data/Hubert_speedup.png">
</div>


## Resources

- [Audio classification task guide](../tasks/audio_classification)
- [Automatic speech recognition task guide](../tasks/asr)

## HubertConfig

[[autodoc]] HubertConfig

<frameworkcontent>
<pt>

## HubertModel

[[autodoc]] HubertModel
    - forward

## HubertForCTC

[[autodoc]] HubertForCTC
    - forward

## HubertForSequenceClassification

[[autodoc]] HubertForSequenceClassification
    - forward

</pt>
<tf>

## TFHubertModel

[[autodoc]] TFHubertModel
    - call

## TFHubertForCTC

[[autodoc]] TFHubertForCTC
    - call

</tf>
</frameworkcontent>
