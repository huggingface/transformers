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
"""TF Pegasus model, ported from the fairseq repo."""
from .configuration_pegasus import PegasusConfig
from .file_utils import add_start_docstrings
from .modeling_tf_bart import BART_START_DOCSTRING, TFBartForConditionalGeneration
from .utils import logging


_CONFIG_FOR_DOC = "PegasusConfig"

START_DOCSTRING = BART_START_DOCSTRING.replace(
    "inherits from :class:`~transformers.TFPreTrainedModel`",
    "inherits from :class:`~transformers.TFBartForConditionalGeneration`",
).replace("BartConfig", _CONFIG_FOR_DOC)


logger = logging.get_logger(__name__)


@add_start_docstrings("Pegasus model for summarization", START_DOCSTRING)
class TFPegasusForConditionalGeneration(TFBartForConditionalGeneration):
    authorized_missing_keys = [
        r"final_logits_bias",
        r"model.encoder.embed_positions.weight",
        r"model.decoder.embed_positions.weight",
    ]
    config_class = PegasusConfig
    # All the code is in src/transformers/modeling_tf_bart.py
