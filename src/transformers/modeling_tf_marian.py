# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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
"""TF BART model, ported from the fairseq repo."""
from .configuration_marian import MarianConfig
from .file_utils import add_start_docstrings
from .modeling_tf_bart import BART_START_DOCSTRING, TFBartForConditionalGeneration
from .utils import logging


_CONFIG_FOR_DOC = "MarianConfig"

START_DOCSTRING = BART_START_DOCSTRING.replace(
    "inherits from :class:`~transformers.TFPreTrainedModel`",
    "inherits from :class:`~transformers.TFBartForConditionalGeneration`",
).replace("BartConfig", _CONFIG_FOR_DOC)


logger = logging.get_logger(__name__)


@add_start_docstrings("Marian model for machine translation", START_DOCSTRING)
class TFMarianMTModel(TFBartForConditionalGeneration):
    authorized_missing_keys = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
        "model.encoder.embed_positions.weight",
        "model.decoder.embed_positions.weight",
    ]
    config_class = MarianConfig

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        self._force_token_id_to_be_generated(logits, self.config.pad_token_id, inverted=True)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            logits = self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits
