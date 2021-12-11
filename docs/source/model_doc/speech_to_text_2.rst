.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Speech2Text2
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Speech2Text2 model is used together with :doc:`Wav2Vec2 <wav2vec2>` for Speech Translation models proposed in
`Large-Scale Self- and Semi-Supervised Learning for Speech Translation <https://arxiv.org/abs/2104.06678>`__ by
Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau.

Speech2Text2 is a *decoder-only* transformer model that can be used with any speech *encoder-only*, such as
:doc:`Wav2Vec2 <wav2vec2>` or :doc:`HuBERT <hubert>` for Speech-to-Text tasks. Please refer to the
:doc:`SpeechEncoderDecoder <speechencoderdecoder>` class on how to combine Speech2Text2 with any speech *encoder-only*
model.

This model was contributed by `Patrick von Platen <https://huggingface.co/patrickvonplaten>`__.

The original code can be found `here
<https://github.com/pytorch/fairseq/blob/1f7ef9ed1e1061f8c7f88f8b94c7186834398690/fairseq/models/wav2vec/wav2vec2_asr.py#L266>`__.


Tips:

- Speech2Text2 achieves state-of-the-art results on the CoVoST Speech Translation dataset. For more information, see
  the `official models <https://huggingface.co/models?other=speech2text2>`__ .
- Speech2Text2 is always used within the :doc:`SpeechEncoderDecoder <speechencoderdecoder>` framework.
- Speech2Text2's tokenizer is based on `fastBPE <https://github.com/glample/fastBPE>`.

Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Speech2Text2's :class:`~transformers.SpeechEncoderDecoderModel` model accepts raw waveform input values from speech and
makes use of :func:`~transformers.generation_utils.GenerationMixin.generate` to translate the input speech
autoregressively to the target language.

The :class:`~transformers.Wav2Vec2FeatureExtractor` class is responsible for preprocessing the input speech and
:class:`~transformers.Speech2Text2Tokenizer` decodes the generated target tokens to the target string. The
:class:`~transformers.Speech2Text2Processor` wraps :class:`~transformers.Wav2Vec2FeatureExtractor` and
:class:`~transformers.Speech2Text2Tokenizer` into a single instance to both extract the input features and decode the
predicted token ids.

- Step-by-step Speech Translation

.. code-block::

        >>> import torch
        >>> from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
        >>> processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")

        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
        >>> generated_ids = model.generate(input_ids=inputs["input_values"], attention_mask=inputs["attention_mask"])

        >>> transcription = processor.batch_decode(generated_ids)


- Speech Translation via Pipelines

    The automatic speech recognition pipeline can also be used to translate speech in just a couple lines of code

.. code-block::

        >>> from datasets import load_dataset
        >>> from transformers import pipeline

        >>> librispeech_en = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> asr = pipeline("automatic-speech-recognition", model="facebook/s2t-wav2vec2-large-en-de", feature_extractor="facebook/s2t-wav2vec2-large-en-de")

        >>> translation_de = asr(librispeech_en[0]["file"])


See `model hub <https://huggingface.co/models?filter=speech2text2>`__ to look for Speech2Text2 checkpoints.


Speech2Text2Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2Text2Config
    :members:


Speech2TextTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2Text2Tokenizer
    :members: batch_decode, decode, save_vocabulary


Speech2Text2Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2Text2Processor
    :members: __call__, from_pretrained, save_pretrained, batch_decode, decode, as_target_processor


Speech2Text2ForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Speech2Text2ForCausalLM
    :members: forward
