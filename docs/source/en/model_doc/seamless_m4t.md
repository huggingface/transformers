<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SeamlessM4T

## Overview

The SeamlessM4T model was proposed in [SeamlessM4T — Massively Multilingual & Multimodal Machine Translation](https://dl.fbaipublicfiles.com/seamless/seamless_m4t_paper.pdf) by the Seamless Communication team from Meta AI.

SeamlessM4T is a collection of models designed to provide high quality translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.

SeamlessM4T enables multiple tasks without relying on separate models:

- Speech-to-speech translation (S2ST)
- Speech-to-text translation (S2TT)
- Text-to-speech translation (T2ST)
- Text-to-text translation (T2TT)
- Automatic speech recognition (ASR)

[`SeamlessM4TModel`] can perform all the above tasks, but each task also has its own dedicated sub-model.

The abstract from the paper is the following:

*What does it take to create the Babel Fish, a tool that can help individuals translate speech between any two languages? While recent breakthroughs in text-based models have pushed machine translation coverage beyond 200 languages, unified speech-to-speech translation models have yet to achieve similar strides. More specifically, conventional speech-to-speech translation systems rely on cascaded systems that perform translation progressively, putting high-performing unified systems out of reach. To address these gaps, we introduce SeamlessM4T, a single model that supports speech-to-speech translation, speech-to-text translation, text-to-speech translation, text-to-text translation, and automatic speech recognition for up to 100 languages. To build this, we used 1 million hours of open speech audio data to learn self-supervised speech representations with w2v-BERT 2.0. Subsequently, we created a multimodal corpus of automatically aligned speech translations. Filtered and combined with human-labeled and pseudo-labeled data, we developed the first multilingual system capable of translating from and into English for both speech and text. On FLEURS, SeamlessM4T sets a new standard for translations into multiple target languages, achieving an improvement of 20% BLEU over the previous SOTA in direct speech-to-text translation. Compared to strong cascaded models, SeamlessM4T improves the quality of into-English translation by 1.3 BLEU points in speech-to-text and by 2.6 ASR-BLEU points in speech-to-speech. Tested for robustness, our system performs better against background noises and speaker variations in speech-to-text tasks compared to the current SOTA model. Critically, we evaluated SeamlessM4T on gender bias and added toxicity to assess translation safety. Finally, all contributions in this work are open-sourced and accessible at https://github.com/facebookresearch/seamless_communication*

## Usage

First, load the processor and a checkpoint of the model:

```python
>>> from transformers import AutoProcessor, SeamlessM4TModel

>>> processor = AutoProcessor.from_pretrained("ylacombe/hf-seamless-m4t-medium")
>>> model = SeamlessM4TModel.from_pretrained("ylacombe/hf-seamless-m4t-medium")
```

You can seamlessly use this model on text or on audio, to generated either translated text or translated audio.

### Speech

You can easily generate translated speech with [`SeamlessM4TModel.generate`]. Here is an example showing how to generate speech from English to Russian.

```python
>>> inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")

>>> audio_array = model.generate(**inputs, tgt_lang="rus")
>>> audio_array = audio_array[0].cpu().numpy().squeeze()
```

You can also translate directly from a speech waveform. Here is an example from Arabic to English:

```python
from datasets import load_dataset

>>> dataset = load_dataset("arabic_speech_corpus", split="test[0:1]")

>>> audio_sample = dataset["audio"][0]["array"]
 
>>> inputs = processor(audios = audio_sample, return_tensors="pt")

>>> audio_array = model.generate(**inputs, tgt_lang="rus")
>>> audio_array = audio_array[0].cpu().numpy().squeeze()
```

#### Tips

[`SeamlessM4TModel`] is transformers top level model to generate speech and text, but you can also use dedicated models that perform the task without additional components, thus reducing the memory footprint.
For example, you can replace the previous snippet with the model dedicated to the S2ST task:

```python
>>> from transformers import SeamlessM4TForSpeechToSpeech
>>> model = SeamlessM4TForSpeechToSpeech.from_pretrained("ylacombe/hf-seamless-m4t-medium")
```


### Text

Similarly, you can generate translated text from text or audio files, this time using the dedicated models.

```python
>>> from transformers import SeamlessM4TForSpeechToText
>>> model = SeamlessM4TForSpeechToText.from_pretrained("ylacombe/hf-seamless-m4t-medium")
>>> audio_sample = dataset["audio"][0]["array"]
 
>>> inputs = processor(audios = audio_sample, return_tensors="pt")

>>> output_tokens = model.generate(**inputs, tgt_lang="fra")
>>> translated_text = processor.decode(output_tokens.tolist()[0], skip_special_tokens=True)
```

And from text:

```python
>>> from transformers import SeamlessM4TForTextToText
>>> model = SeamlessM4TForTextToText.from_pretrained("ylacombe/hf-seamless-m4t-medium")
>>> inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")

>>> output_tokens = model.generate(**inputs, tgt_lang="fra")
>>> translated_text = processor.decode(output_tokens.tolist()[0], skip_special_tokens=True)
```

#### Tips

Three last tips:

1. [`SeamlessM4TModel`] can generate text and/or speech. Pass `generate_speech=False` to [`SeamlessM4TModel.generate`] to only generate text. You also have the possibility to pass `return_intermediate_token_ids=True`, to get both text token ids and the generated speech.
2. You have the possibility to change the speaker used for speech synthesis with the `spkr_id` argument.
3. You can use different [generation strategies](./generation_strategies) for speech and text generation, e.g `.generate(input_ids=input_ids, text_num_beams=4, speech_do_sample=True)` which will successively perform beam-search decoding on the text model, and multinomial sampling on the speech model.



This model was contributed by [ylacombe](https://huggingface.co/ylacombe). The original code can be found [here](https://github.com/facebookresearch/seamless_communication).

## SeamlessM4TModel

[[autodoc]] SeamlessM4TModel
    - generate


## SeamlessM4TForTextToSpeech

[[autodoc]] SeamlessM4TForTextToSpeech
    - generate


## SeamlessM4TForSpeechToSpeech

[[autodoc]] SeamlessM4TForSpeechToSpeech
    - generate


## SeamlessM4TForTextToText

[[autodoc]] transformers.SeamlessM4TForTextToText
    - forward
    - generate

## SeamlessM4TForSpeechToText

[[autodoc]] transformers.SeamlessM4TForSpeechToText
    - forward
    - generate

## SeamlessM4TConfig

[[autodoc]] SeamlessM4TConfig


## SeamlessM4TTokenizer

[[autodoc]] SeamlessM4TTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## SeamlessM4TTokenizerFast

[[autodoc]] SeamlessM4TTokenizerFast


## SeamlessM4TFeatureExtractor

[[autodoc]] SeamlessM4TFeatureExtractor

## SeamlessM4TProcessor

[[autodoc]] SeamlessM4TProcessor


## SeamlessM4TCodeHifiGan

[[autodoc]] SeamlessM4TCodeHifiGan


## SeamlessM4THifiGan

[[autodoc]] SeamlessM4THifiGan


## SeamlessM4TTextToUnitForConditionalGeneration

[[autodoc]] SeamlessM4TTextToUnitForConditionalGeneration


## SeamlessM4TTextToUnitModel

[[autodoc]] SeamlessM4TTextToUnitModel


