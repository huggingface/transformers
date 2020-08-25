# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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
""" PEGASUS model configuration """

import logging

from .configuration_bart import BART_CONFIG_ARGS_DOC, BartConfig
from .file_utils import add_start_docstrings_to_callable


logger = logging.getLogger(__name__)

# These config values do not vary between checkpoints
DEFAULTS = dict(
    vocab_size=96103,
    max_position_embeddings=512,
    d_model=1024,
    encoder_ffn_dim=4096,
    decoder_ffn_dim=4096,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    encoder_layers=16,
    decoder_layers=16,
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.1,
    pad_token_id=0,
    eos_token_id=1,
    is_encoder_decoder=True,
    normalize_before=True,
    scale_embedding=True,
    normalize_embedding=False,
    add_final_layer_norm=True,
    static_position_embeddings=True,
    num_beams=8,
    activation_function="relu",
)
# Config values that vary between checkpoints: for testing and conversion
max_gen_length = {
    # See appendix C of paper
    "xsum": 64,
    "cnn_dailymail": 128,
    "newsroom": 128,
    "wikihow": 256,
    "multi_news": 256,
    "reddit_tifu": 128,
    "big_patent": 256,
    "arxiv": 256,
    "pubmed": 256,
    "gigaword": 32,
    "aeslc": 32,
    "billsum": 256,
    "large": 256,  # @sshleifer chose arbitrarily
}
max_model_length = {
    "xsum": 512,
    "cnn_dailymail": 1024,
    "newsroom": 512,
    "wikihow": 512,
    "multi_news": 1024,
    "reddit_tifu": 512,
    "big_patent": 1024,
    "arxiv": 1024,
    "pubmed": 1024,
    "gigaword": 128,
    "aeslc": 512,
    "billsum": 1024,
    "large": 1024,
}
expected_alpha = {
    "multinews": 0.9,
    "wikihow": 0.6,
    "reddit_tifu": 0.6,
    "big_patent": 0.7,
    "gigaword": 0.6,
    "aeslc": 0.6,
    "billsum": 0.6,
}  # otherwise 0.8


@add_start_docstrings_to_callable(BART_CONFIG_ARGS_DOC)
class PegasusConfig(BartConfig):
    r"""
        :class:`~transformers.PegasusConfig` is the configuration class to store the configuration of a
        `PegasusModel`.
    """
    model_type = "pegasus"
    # The implementation of the config object is in BartConfig
