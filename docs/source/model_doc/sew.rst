.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

SEW
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SEW (Squeezed and Efficient Wav2Vec) was proposed in `Performance-Efficiency Trade-offs in Unsupervised Pre-training
for Speech Recognition <https://arxiv.org/abs/2109.06870>`__ by Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q.
Weinberger, Yoav Artzi.

The abstract from the paper is the following:

*This paper is a study of performance-efficiency trade-offs in pre-trained models for automatic speech recognition
(ASR). We focus on wav2vec 2.0, and formalize several architecture designs that influence both the model performance
and its efficiency. Putting together all our observations, we introduce SEW (Squeezed and Efficient Wav2vec), a
pre-trained model architecture with significant improvements along both performance and efficiency dimensions across a
variety of training setups. For example, under the 100h-960h semi-supervised setup on LibriSpeech, SEW achieves a 1.9x
inference speedup compared to wav2vec 2.0, with a 13.5% relative reduction in word error rate. With a similar inference
time, SEW reduces word error rate by 25-50% across different model sizes.*

Tips:

- SEW is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
- SEWForCTC is fine-tuned using connectionist temporal classification (CTC) so the model output has to be decoded using
  :class:`~transformers.Wav2Vec2CTCTokenizer`.

This model was contributed by `anton-l <https://huggingface.co/anton-l>`__.


SEWConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SEWConfig
    :members:


SEWModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SEWModel
    :members: forward


SEWForCTC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SEWForCTC
    :members: forward


SEWForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SEWForSequenceClassification
    :members: forward
