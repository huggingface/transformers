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

from .configuration_bart import BART_CONFIG_ARGS_DOC, BartConfig
from .file_utils import add_start_docstrings_to_callable
from .utils import logging


logger = logging.get_logger(__name__)

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
task_specific_params = {
    # These are task specific params for pegasus-large and normal params for finetuned checkpoints
    "summarization_xsum": {"length_penalty": 0.6, "max_length": 64, "max_position_embeddings": 512},
    "summarization_cnn_dailymail": {"length_penalty": 0.8, "max_length": 128, "max_position_embeddings": 1024},
    "summarization_newsroom": {"length_penalty": 0.8, "max_length": 128, "max_position_embeddings": 512},
    "summarization_wikihow": {"length_penalty": 0.6, "max_length": 256, "max_position_embeddings": 512},
    "summarization_multi_news": {"length_penalty": 0.8, "max_length": 256, "max_position_embeddings": 1024},
    "summarization_reddit_tifu": {"length_penalty": 0.6, "max_length": 128, "max_position_embeddings": 512},
    "summarization_big_patent": {"length_penalty": 0.7, "max_length": 256, "max_position_embeddings": 1024},
    "summarization_arxiv": {"length_penalty": 0.8, "max_length": 256, "max_position_embeddings": 1024},
    "summarization_pubmed": {"length_penalty": 0.8, "max_length": 256, "max_position_embeddings": 1024},
    "summarization_gigaword": {"length_penalty": 0.6, "max_length": 32, "max_position_embeddings": 128},
    "summarization_aeslc": {"length_penalty": 0.6, "max_length": 32, "max_position_embeddings": 512},
    "summarization_billsum": {"length_penalty": 0.6, "max_length": 256, "max_position_embeddings": 1024},
    # this last entry is useless -- just for consistency
    "summarization_large": {"length_penalty": 0.8, "max_length": 256, "max_position_embeddings": 1024},
}


@add_start_docstrings_to_callable(BART_CONFIG_ARGS_DOC)
class PegasusConfig(BartConfig):
    r"""
    :class:`~transformers.PegasusConfig` is the configuration class to store the configuration of a
    `PegasusModel`.
    """
    model_type = "pegasus"
    # The implementation of the config object is in BartConfig
