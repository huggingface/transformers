.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Speech2Text
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Speech2Text model was proposed in `fairseq S2T: Fast Speech-to-Text Modeling with fairseq
<https://arxiv.org/abs/2010.05171>`__ by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino. It's a
transformer-based seq2seq (encoder-decoder) model designed for end-to-end Automatic Speech Recognition (ASR) and Speech
Translation (ST). It uses a convolutional downsampler to reduce the length of speech inputs by 3/4th before they are
fed into the encoder. The model is trained with standard autoregressive cross-entropy loss and generates the
transcripts/translations autoregressively. Speech2Text has been fine-tuned on several datasets for ASR and ST:
`LibriSpeech <http://www.openslr.org/12>`__, `CoVoST 2 <https://github.com/facebookresearch/covost>`__, `MuST-C
<https://ict.fbk.eu/must-c/>`__.

This model was contributed by `valhalla <https://huggingface.co/valhalla>`__. The original code can be found `here
<https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text>`__.


Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Speech2Text is a speech model that accepts a float tensor of log-mel filter-bank features extracted from the speech
signal. It's a transformer-based seq2seq model, so the transcripts/translations are generated autoregressively. The
:obj:`generate()` method can be used for inference.

The :class:`~transformers.Speech2TextFeatureExtractor` class is responsible for extracting the log-mel filter-bank
features. The :class:`~transformers.Speech2TextProcessor` wraps :class:`~transformers.Speech2TextFeatureExtractor` and
:class:`~transformers.Speech2TextTokenizer` into a single instance to both extract the input features and decode the
predicted token ids.

The feature extractor depends on :obj:`torchaudio` and the tokenizer depends on :obj:`sentencepiece` so be sure to
install those packages before running the examples. You could either install those as extra speech dependencies with
``pip install transformers"[speech, sentencepiece]"`` or install the packages seperately with ``pip install torchaudio
sentencepiece``. Also ``torchaudio`` requires the development version of the `libsndfile
<http://www.mega-nerd.com/libsndfile/>`__ package which can be installed via a system package manager. On Ubuntu it can
be installed as follows: ``apt install libsndfile1-dev``


- ASR and Speech Translation

.. code-block::

        >>> import torch
        >>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        >>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
        >>> generated_ids = model.generate(input_ids=inputs["input_features"], attention_mask=inputs["attention_mask"])

        >>> transcription = processor.batch_decode(generated_ids)


- Multilingual speech translation

    For multilingual speech translation models, :obj:`eos_token_id` is used as the :obj:`decoder_start_token_id` and
    the target language id is forced as the first generated token. To force the target language id as the first
    generated token, pass the :obj:`forced_bos_token_id` parameter to the :obj:`generate()` method. The following
    example shows how to transate English speech to French text using the `facebook/s2t-medium-mustc-multilingual-st`
    checkpoint.

.. code-block::

        >>> import torch
        >>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
        >>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
        >>> generated_ids = model.generate(input_ids=inputs["input_features"], attention_mask=inputs["attention_mask], forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"])

        >>> translation = processor.batch_decode(generated_ids)


See the `model hub <https://huggingface.co/models?filter=speech_to_text>`__ to look for Speech2Text checkpoints.


Speech2TextConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2TextConfig
    :members:


Speech2TextTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2TextTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


Speech2TextFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2TextFeatureExtractor
    :members: __call__


Speech2TextProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2TextProcessor
    :members: __call__, from_pretrained, save_pretrained, batch_decode, decode, as_target_processor


Speech2TextModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2TextModel
    :members: forward


Speech2TextForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2TextForConditionalGeneration
    :members: forward
