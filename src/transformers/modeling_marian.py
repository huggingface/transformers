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


from transformers.modeling_bart import BartForConditionalGeneration


PRETRAINED_MODEL_ARCHIVE_MAP = {
    "opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/pytorch_model.bin",
}


class MarianMTModel(BartForConditionalGeneration):
    """Pytorch version of marian-nmt's transformer.h (c++). Designed for the OPUS-NMT translation checkpoints.
    Model API is identical to BartForConditionalGeneration"""

    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP

    def prepare_scores_for_generation(self, scores, cur_len, max_length):
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(scores, self.config.eos_token_id)
        return scores
