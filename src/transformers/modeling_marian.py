# coding=utf-8
# Copyright 2020 Marian Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MarianMTModel model, ported from the Marian C++ repo."""


from .configuration_marian import MarianConfig
from .modeling_bart import BartForConditionalGeneration


# See all Marian models at https://huggingface.co/models?search=Helsinki-NLP


class MarianMTModel(BartForConditionalGeneration):
    config_class = MarianConfig
    r"""
    Pytorch version of marian-nmt's transformer.h (c++). Designed for the OPUS-NMT translation checkpoints.
    Model API is identical to BartForConditionalGeneration.
    Available models are listed at `Model List <https://huggingface.co/models?search=Helsinki-NLP>`__

    Examples::

        >>> from transformers import MarianTokenizer, MarianMTModel
        >>> from typing import List
        >>> src = 'fr'  # source language
        >>> trg = 'en'  # target language
        >>> sample_text = "où est l'arrêt de bus ?"
        >>> mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'

        >>> model = MarianMTModel.from_pretrained(mname)
        >>> tok = MarianTokenizer.from_pretrained(mname)
        >>> batch = tok.prepare_seq2seq_batch(src_texts=[sample_text])  # don't need tgt_text for inference
        >>> gen = model.generate(**batch)  # for forward pass: model(**batch)
        >>> words: List[str] = tok.batch_decode(gen, skip_special_tokens=True)  # returns "Where is the bus stop ?"

    """

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        logits[:, self.config.pad_token_id] = float("-inf")  # never predict pad token.
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits
