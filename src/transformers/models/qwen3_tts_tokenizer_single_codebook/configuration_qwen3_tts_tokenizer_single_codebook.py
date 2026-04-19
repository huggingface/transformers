# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


@strict
class Qwen3TTSTokenizerSingleCodebookDiTConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS SingleCodebook DiT decoder.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the DiT model.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of transformer blocks.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        ff_mult (`int`, *optional*, defaults to 2):
            Feedforward layer multiplier.
        emb_dim (`int`, *optional*, defaults to 512):
            Codec embedding dimension.
        head_dim (`int`, *optional*, defaults to 64):
            Attention head dimension.
        repeats (`int`, *optional*, defaults to 2):
            Number of times codec embeddings are repeated.
        num_embeds (`int`, *optional*, defaults to 8193):
            Number of unique codec embeddings.
        mel_dim (`int`, *optional*, defaults to 80):
            Mel-spectrogram dimension.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        block_size (`int`, *optional*, defaults to 24):
            Block size for block-diagonal attention mask.
        look_ahead_layers (`list[int]`, *optional*, defaults to `[10]`):
            Layer indices that use look-ahead attention.
        look_backward_layers (`list[int]`, *optional*, defaults to `[0, 20]`):
            Layer indices that use look-backward attention.
        enc_emb_dim (`int`, *optional*, defaults to 192):
            Speaker embedding dimension.
        enc_dim (`int`, *optional*, defaults to 128):
            Encoder output dimension.
        enc_channels (`list[int]`, *optional*, defaults to `[256, 256, 256, 256, 768]`):
            Encoder channel sizes.
        enc_kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            Encoder kernel sizes.
        enc_dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            Encoder dilations.
        enc_attention_channels (`int`, *optional*, defaults to 64):
            Encoder attention channels.
        enc_res2net_scale (`int`, *optional*, defaults to 2):
            Encoder Res2Net scale.
        enc_se_channels (`int`, *optional*, defaults to 64):
            Encoder SE channels.
        rope_parameters (`RopeParameters`, *optional*):
            RoPE configuration.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length.
    """

    model_type = "qwen3_tts_dit"

    def __init__(
        self,
        hidden_size: int | None = 1024,
        num_hidden_layers: int | None = 22,
        num_attention_heads: int | None = 16,
        ff_mult: int | None = 2,
        emb_dim: int | None = 512,
        head_dim: int | None = 64,
        repeats: int | None = 2,
        num_embeds: int | None = 8193,
        mel_dim: int | None = 80,
        dropout: float | None = 0.1,
        block_size: int | None = 24,
        look_ahead_layers: list[int] | None = None,
        look_backward_layers: list[int] | None = None,
        enc_emb_dim: int | None = 192,
        enc_dim: int | None = 128,
        enc_channels: list[int] | None = None,
        enc_kernel_sizes: list[int] | None = None,
        enc_dilations: list[int] | None = None,
        enc_attention_channels: int | None = 64,
        enc_res2net_scale: int | None = 2,
        enc_se_channels: int | None = 64,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        max_position_embeddings: int | None = 32768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ff_mult = ff_mult
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.repeats = repeats
        self.num_embeds = num_embeds
        self.mel_dim = mel_dim
        self.dropout = dropout
        self.block_size = block_size
        self.look_ahead_layers = look_ahead_layers if look_ahead_layers is not None else [10]
        self.look_backward_layers = look_backward_layers if look_backward_layers is not None else [0, 20]
        self.enc_emb_dim = enc_emb_dim
        self.enc_dim = enc_dim
        self.enc_channels = enc_channels if enc_channels is not None else [256, 256, 256, 256, 768]
        self.enc_kernel_sizes = enc_kernel_sizes if enc_kernel_sizes is not None else [5, 3, 3, 3, 1]
        self.enc_dilations = enc_dilations if enc_dilations is not None else [1, 2, 3, 4, 1]
        self.enc_attention_channels = enc_attention_channels
        self.enc_res2net_scale = enc_res2net_scale
        self.enc_se_channels = enc_se_channels
        self.rope_parameters = rope_parameters
        self.max_position_embeddings = max_position_embeddings


@strict
class Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS SingleCodebook BigVGAN vocoder.

    Args:
        mel_dim (`int`, *optional*, defaults to 80):
            Mel-spectrogram input dimension.
        upsample_initial_channel (`int`, *optional*, defaults to 1536):
            Initial channel count for the upsampling stack.
        resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
            Kernel sizes for each residual block.
        resblock_dilation_sizes (`list[list[int]]`, *optional*):
            Dilation sizes for each residual block.
        upsample_rates (`list[int]`, *optional*, defaults to `[5, 3, 2, 2, 2, 2]`):
            Upsampling rates for each layer.
        upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[11, 7, 4, 4, 4, 4]`):
            Kernel sizes for each upsampling layer.
    """

    model_type = "qwen3_tts_tokenizer_single_codebook_decoder_bigvgan"

    def __init__(
        self,
        mel_dim: int | None = 80,
        upsample_initial_channel: int | None = 1536,
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mel_dim = mel_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes if resblock_kernel_sizes is not None else [3, 7, 11]
        self.resblock_dilation_sizes = (
            resblock_dilation_sizes if resblock_dilation_sizes is not None else [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        )
        self.upsample_rates = upsample_rates if upsample_rates is not None else [5, 3, 2, 2, 2, 2]
        self.upsample_kernel_sizes = (
            upsample_kernel_sizes if upsample_kernel_sizes is not None else [11, 7, 4, 4, 4, 4]
        )


@strict
class Qwen3TTSTokenizerSingleCodebookDecoderConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS SingleCodebook decoder (DiT + BigVGAN).

    Args:
        dit_config (`dict`, *optional*):
            Configuration for the DiT sub-model.
        bigvgan_config (`dict`, *optional*):
            Configuration for the BigVGAN sub-model.
    """

    model_type = "qwen3_tts_tokenizer_single_codebook_decoder"
    sub_configs = {
        "dit_config": Qwen3TTSTokenizerSingleCodebookDiTConfig,
        "bigvgan_config": Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig,
    }

    def __init__(
        self,
        dit_config: dict | None = None,
        bigvgan_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if dit_config is None:
            dit_config = {}
            logger.info("dit_config is None. Initializing DiT with default values.")
        if bigvgan_config is None:
            bigvgan_config = {}
            logger.info("bigvgan_config is None. Initializing BigVGAN with default values.")

        self.dit_config = (
            Qwen3TTSTokenizerSingleCodebookDiTConfig(**dit_config) if isinstance(dit_config, dict) else dit_config
        )
        self.bigvgan_config = (
            Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig(**bigvgan_config)
            if isinstance(bigvgan_config, dict)
            else bigvgan_config
        )


@strict
class Qwen3TTSTokenizerSingleCodebookEncoderConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS SingleCodebook Whisper-based VQ encoder.

    Args:
        n_mels (`int`, *optional*, defaults to 128):
            Number of mel filterbanks.
        n_ctx (`int`, *optional*, defaults to 1500):
            Maximum context length.
        n_state (`int`, *optional*, defaults to 1024):
            Hidden state dimension.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads.
        n_layer (`int`, *optional*, defaults to 24):
            Number of transformer layers.
        n_window (`int`, *optional*, defaults to 128):
            Window size for windowed attention.
        output_dim (`int`, *optional*, defaults to 512):
            VQ output dimension.
        grad_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to use gradient checkpointing.
        enable_mp (`bool`, *optional*, defaults to `False`):
            Whether to enable mixed precision.
        audio_sequence_parallel (`bool`, *optional*, defaults to `False`):
            Whether to use sequence parallelism for audio.
        audio_vq_type (`str`, *optional*, defaults to `"GRVQ"`):
            Type of vector quantization.
        audio_vq_layers (`int`, *optional*, defaults to 1):
            Number of VQ layers.
        audio_vq_codebook_size (`int`, *optional*, defaults to 512):
            Codebook size.
        audio_vq_codebook_dim (`int`, *optional*, defaults to 512):
            Codebook vector dimension.
        audio_vq_pe (`str`, *optional*, defaults to `"rope"`):
            Position encoding type for VQ.
        audio_vq_ds_rate (`int`, *optional*, defaults to 2):
            Downsampling rate inside the VQ module.
    """

    model_type = "qwen3_tts_tokenizer_SingleCodebook_encoder"

    def __init__(
        self,
        n_mels: int | None = 128,
        n_ctx: int | None = 1500,
        n_state: int | None = 1024,
        n_head: int | None = 16,
        n_layer: int | None = 24,
        n_window: int | None = 128,
        output_dim: int | None = 512,
        grad_checkpointing: bool | None = False,
        enable_mp: bool | None = False,
        audio_sequence_parallel: bool | None = False,
        audio_vq_type: str | None = "GRVQ",
        audio_vq_layers: int | None = 1,
        audio_vq_codebook_size: int | None = 512,
        audio_vq_codebook_dim: int | None = 512,
        audio_vq_pe: str | None = "rope",
        audio_vq_ds_rate: int | None = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_window = n_window
        self.output_dim = output_dim
        self.grad_checkpointing = grad_checkpointing
        self.enable_mp = enable_mp
        self.audio_sequence_parallel = audio_sequence_parallel
        self.audio_vq_type = audio_vq_type
        self.audio_vq_layers = audio_vq_layers
        self.audio_vq_codebook_size = audio_vq_codebook_size
        self.audio_vq_codebook_dim = audio_vq_codebook_dim
        self.audio_vq_pe = audio_vq_pe
        self.audio_vq_ds_rate = audio_vq_ds_rate


@strict
class Qwen3TTSTokenizerSingleCodebookConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS SingleCodebook tokenizer (Whisper VQ encoder + DiT/BigVGAN decoder).

    Args:
        encoder_config (`dict`, *optional*):
            Configuration for the Whisper-based VQ encoder.
        decoder_config (`dict`, *optional*):
            Configuration for the DiT+BigVGAN decoder.
        input_sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate of the input audio.
        output_sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate of the decoded output audio.
        decode_upsample_rate (`int`, *optional*, defaults to 200):
            Upsampling rate applied during decoding.
        encode_downsample_rate (`int`, *optional*, defaults to 200):
            Downsampling rate applied during encoding.
    """

    model_type = "qwen3_tts_tokenizer_single_codebook"
    sub_configs = {
        "encoder_config": Qwen3TTSTokenizerSingleCodebookEncoderConfig,
        "decoder_config": Qwen3TTSTokenizerSingleCodebookDecoderConfig,
    }

    def __init__(
        self,
        encoder_config: dict | None = None,
        decoder_config: dict | None = None,
        input_sample_rate: int | None = 24000,
        output_sample_rate: int | None = 24000,
        decode_upsample_rate: int | None = 200,
        encode_downsample_rate: int | None = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if encoder_config is None:
            encoder_config = {}
            logger.info("encoder_config is None. Initializing SingleCodebook encoder with default values.")
        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. Initializing SingleCodebook decoder with default values.")

        self.encoder_config = (
            Qwen3TTSTokenizerSingleCodebookEncoderConfig(**encoder_config)
            if isinstance(encoder_config, dict)
            else encoder_config
        )
        self.decoder_config = (
            Qwen3TTSTokenizerSingleCodebookDecoderConfig(**decoder_config)
            if isinstance(decoder_config, dict)
            else decoder_config
        )

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.decode_upsample_rate = decode_upsample_rate
        self.encode_downsample_rate = encode_downsample_rate


__all__ = [
    "Qwen3TTSTokenizerSingleCodebookDiTConfig",
    "Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig",
    "Qwen3TTSTokenizerSingleCodebookDecoderConfig",
    "Qwen3TTSTokenizerSingleCodebookEncoderConfig",
    "Qwen3TTSTokenizerSingleCodebookConfig",
]
