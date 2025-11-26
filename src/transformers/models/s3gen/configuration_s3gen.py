# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""S3Gen model configuration"""

from ...configuration_utils import PretrainedConfig


class S3GenConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`S3GenModel`]. It is used to instantiate a S3Gen
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the S3Gen
    [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 6561):
            Vocabulary size of the S3 speech tokenizer.
        token_embed_dim (`int`, *optional*, defaults to 512):
            Dimension of the token embeddings.
        speaker_feat_dim (`int`, *optional*, defaults to 80):
            Number of mel bins for speaker encoder input.
        speaker_embed_dim (`int`, *optional*, defaults to 192):
            Dimension of the speaker embeddings.
        encoder_output_size (`int`, *optional*, defaults to 512):
            Output size of the conformer encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the conformer encoder.
        encoder_linear_units (`int`, *optional*, defaults to 2048):
            Dimension of the feedforward network in the conformer encoder.
        encoder_num_blocks (`int`, *optional*, defaults to 6):
            Number of conformer encoder blocks.
        encoder_dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout probability for the encoder.
        decoder_in_channels (`int`, *optional*, defaults to 320):
            Number of input channels for the conditional decoder (encoder output + speaker embedding).
        decoder_out_channels (`int`, *optional*, defaults to 80):
            Number of output channels for the decoder (mel bins).
        decoder_channels (`List[int]`, *optional*, defaults to `[256]`):
            List of channel dimensions for the decoder U-Net.
        decoder_n_blocks (`int`, *optional*, defaults to 4):
            Number of transformer blocks in each decoder stage.
        decoder_num_mid_blocks (`int`, *optional*, defaults to 12):
            Number of middle blocks in the decoder U-Net.
        decoder_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the decoder.
        decoder_attention_head_dim (`int`, *optional*, defaults to 64):
            Dimension of each attention head in the decoder.
        decoder_act_fn (`str`, *optional*, defaults to `"gelu"`):
            Activation function for the decoder.
        cfm_sigma_min (`float`, *optional*, defaults to 1e-6):
            Minimum sigma for conditional flow matching.
        cfm_solver (`str`, *optional*, defaults to `"euler"`):
            ODE solver for conditional flow matching.
        cfm_t_scheduler (`str`, *optional*, defaults to `"cosine"`):
            Time scheduler for conditional flow matching.
        cfm_inference_cfg_rate (`float`, *optional*, defaults to 0.7):
            Classifier-free guidance rate for inference.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Audio sampling rate in Hz.
        mel_bins (`int`, *optional*, defaults to 80):
            Number of mel frequency bins.
        n_fft (`int`, *optional*, defaults to 1920):
            FFT size for mel spectrogram extraction (24kHz sampling rate).
        hop_length (`int`, *optional*, defaults to 480):
            Hop length for mel spectrogram extraction (24kHz sampling rate).
        win_size (`int`, *optional*, defaults to 1920):
            Window size for mel spectrogram extraction.
        fmin (`int`, *optional*, defaults to 0):
            Minimum frequency for mel filter bank.
        fmax (`int`, *optional*, defaults to 8000):
            Maximum frequency for mel filter bank.
        input_frame_rate (`int`, *optional*, defaults to 25):
            Frame rate of the S3 tokenizer (25 fps).
        token_mel_ratio (`int`, *optional*, defaults to 2):
            Ratio between mel frames and tokens (2 mel frames per token).
        pre_lookahead_len (`int`, *optional*, defaults to 3):
            Pre-lookahead length for causal streaming.

    Example:

    ```python
    >>> from transformers import S3GenConfig, S3GenModel

    >>> # Initializing a S3Gen configuration
    >>> configuration = S3GenConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = S3GenModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "s3gen"

    def __init__(
        self,
        # Token embedding
        vocab_size=6561,
        token_embed_dim=512,
        # Speaker encoder (CAMPPlus)
        speaker_feat_dim=80,
        speaker_embed_dim=192,
        # UpsampleConformerEncoder
        encoder_output_size=512,
        encoder_attention_heads=8,
        encoder_linear_units=2048,
        encoder_num_blocks=6,
        encoder_dropout_rate=0.1,
        # ConditionalDecoder (U-Net)
        decoder_in_channels=320,
        decoder_out_channels=80,
        decoder_channels=[256],
        decoder_n_blocks=4,
        decoder_num_mid_blocks=12,
        decoder_num_heads=8,
        decoder_attention_head_dim=64,
        decoder_act_fn="gelu",
        # CFM params
        cfm_sigma_min=1e-6,
        cfm_solver="euler",
        cfm_t_scheduler="cosine",
        cfm_inference_cfg_rate=0.7,
        # Audio params (mel extraction for reference audio at 24kHz)
        sampling_rate=24000,
        mel_bins=80,
        n_fft=1920,
        hop_length=480,
        win_size=1920,
        fmin=0,
        fmax=8000,
        # Flow params
        input_frame_rate=25,
        token_mel_ratio=2,
        pre_lookahead_len=3,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.speaker_feat_dim = speaker_feat_dim
        self.speaker_embed_dim = speaker_embed_dim
        self.encoder_output_size = encoder_output_size
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_linear_units = encoder_linear_units
        self.encoder_num_blocks = encoder_num_blocks
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_in_channels = decoder_in_channels
        self.decoder_out_channels = decoder_out_channels
        self.decoder_channels = decoder_channels
        self.decoder_n_blocks = decoder_n_blocks
        self.decoder_num_mid_blocks = decoder_num_mid_blocks
        self.decoder_num_heads = decoder_num_heads
        self.decoder_attention_head_dim = decoder_attention_head_dim
        self.decoder_act_fn = decoder_act_fn
        self.cfm_sigma_min = cfm_sigma_min
        self.cfm_solver = cfm_solver
        self.cfm_t_scheduler = cfm_t_scheduler
        self.cfm_inference_cfg_rate = cfm_inference_cfg_rate
        self.sampling_rate = sampling_rate
        self.mel_bins = mel_bins
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.input_frame_rate = input_frame_rate
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        super().__init__(**kwargs)


__all__ = ["S3GenConfig"]
