# coding=utf-8
# Copyright 2022 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
""" FastSpeech2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

FASTSPEECH2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "fastspeech2": "https://huggingface.co/fastspeech2/resolve/main/config.json",
    # See all FastSpeech2 models at https://huggingface.co/models?filter=fastspeech2
}


class FastSpeech2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2Model`]. It is used to instantiate an
    FastSpeech2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the FastSpeech2
    [fastspeech2](https://huggingface.co/fastspeech2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 75):
            Vocabulary size of the FastSpeech2 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`~FastSpeech2Model`] or [`~TFFastSpeech2Model`].
        encoder_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the FastSpeech2 encoder.
        decoder_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the FastSpeech2 decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the FastSpeech2 encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the FastSpeech2 decoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_source_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        Example:

    ```python
    >>> from transformers import FastSpeech2Model, FastSpeech2Config

    >>> # Initializing a FastSpeech2 fastspeech2 style configuration
    >>> configuration = FastSpeech2Config()

    >>> # Initializing a model from the fastspeech2 style configuration
    >>> model = FastSpeech2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "fastspeech2"

    def __init__(
        self,
        n_frames_per_step=1,
        output_frame_dim=80,
        encoder_embed_dim=256,
        speaker_embed_dim=64,
        dropout=0.2,
        max_source_positions=1024,
        encoder_attention_heads=2,
        fft_hidden_dim=1024,
        fft_kernel_size=9,
        attention_dropout=0,
        encoder_layers=4,
        decoder_embed_dim=256,
        decoder_attention_heads=2,
        decoder_layers=4,
        add_postnet=False,
        postnet_conv_dim=512,
        postnet_conv_kernel_size=5,
        postnet_layers=5,
        postnet_dropout=0.5,
        vocab_size=75,
        num_speakers=1,
        var_pred_n_bins=256,
        var_pred_hidden_dim=256,
        var_pred_kernel_size=3,
        var_pred_dropout=0.5,
        pitch_max=5.733940816898645,
        pitch_min=-4.660287183665281,
        energy_max=3.2244551181793213,
        energy_min=-4.9544901847839355,
        initializer_range=0.0625,
        mean=True,
        std=True,
        pad_token_id=1,
        **kwargs
    ):
        if fft_kernel_size % 2 == 0:
            raise ValueError(f"`fft_kernel_size` must be odd, but got {fft_kernel_size} instead.")
        if postnet_conv_kernel_size % 2 == 0:
            raise ValueError(f"`postnet_conv_kernel_size` must be odd, but got {postnet_conv_kernel_size} instead.")
        if var_pred_kernel_size % 2 == 0:
            raise ValueError(f"`var_pred_kernel_size` must be odd, but got {var_pred_kernel_size} instead.")
        self.n_frames_per_step = n_frames_per_step
        self.output_frame_dim = output_frame_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.speaker_embed_dim = speaker_embed_dim
        self.dropout = dropout
        self.max_source_positions = max_source_positions
        self.encoder_attention_heads = encoder_attention_heads
        self.fft_hidden_dim = fft_hidden_dim
        self.fft_kernel_size = fft_kernel_size
        self.attention_dropout = attention_dropout
        self.encoder_layers = encoder_layers
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_layers = decoder_layers
        self.add_postnet = add_postnet
        self.postnet_conv_dim = postnet_conv_dim
        self.postnet_conv_kernel_size = postnet_conv_kernel_size
        self.postnet_layers = postnet_layers
        self.postnet_dropout = postnet_dropout
        self.vocab_size = vocab_size
        self.num_speakers = num_speakers
        self.var_pred_n_bins = var_pred_n_bins
        self.var_pred_hidden_dim = var_pred_hidden_dim
        self.var_pred_kernel_size = var_pred_kernel_size
        self.var_pred_dropout = var_pred_dropout
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.initializer_range = initializer_range
        self.mean = mean
        self.std = std
        super().__init__(pad_token_id=pad_token_id, **kwargs)
