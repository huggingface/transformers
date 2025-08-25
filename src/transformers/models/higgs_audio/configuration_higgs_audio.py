# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""HiggsAudioConfig."""

from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


class HiggsAudioEncoderConfig(PretrainedConfig):
    """Configuration of the Audio encoder in Higgs-Audio."""

    model_type = "higgs_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        encoder_layerdrop=0.0,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        init_std=0.02,
        max_source_positions=1500,
        pad_token_id=128001,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.init_std = init_std
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.pad_token_id = pad_token_id


class HiggsAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class for the HiggsAudioModel. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) architecture.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text backbone.
        audio_encoder_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the whisper encoder.
            The audio encoder will be bidirectional and will be only available for audio understanding.
        audio_adapter_type (`str`, *optional*, defaults to `"stack"`):
            The type of audio adapter to use. We support three types of adapter:
            - stack:
                We stack additional Transformer layers after the main LLM backbone for audio generation.
            - dual_ffn:
                For selected part of the LLM backbone, we replace the text FFN with a dual FFN architecture
                that contains an additional audio FFN. The audio FFN will be triggered when the location is marked for audio tokens.
            - dual_ffn_fast_forward:
                We pick a few layers in the LLM backbone to plug-in the audio FFN. For the remaining layers,
                the audio hidden states will be directly fast-forward to the next layer.
        audio_embed_avg (`bool`, *optional*, defaults to `False`):
            Whether to average the audio embeddings before sending them to the text attention layer.
            The hidden size of the audio feedforward network in dual-path FFN
            The intermediate size of the audio feedforward network in dual-path FFN
            The layers in the LLM backbone to plug-in the dual FFN layer (mixture of audio FFN and text FFN).
        audio_dual_ffn_layers (`list[int]`, *optional*):
            The layers in the LLM backbone to plug-in the dual FFN layer (mixture of audio FFN and text FFN).
        encode_audio_in_tokens (`bool`, *optional*, defaults to `False`):
            Whether to encode the input audio directly as discrete audio tokens.
            When True, the model uses `audio_in_token`
            positions filled with audio tokens extracted via the audio tokenizer.
            Note that `encode_audio_in_tokens` can be combined with `encode_whisper_embed`.
        use_delay_pattern (`bool`, *optional*, defaults to `False`):
            Whether to use delay pattern in the audio decoder.
        skip_audio_tower (`bool`, *optional*, defaults to `False`):
            Whether to skip the audio tower in the audio encoder.
        use_audio_out_embed_projector (`bool`, *optional*, defaults to `False`):
            Whether to use an embedding projector to map audio out embeddings.
        use_audio_out_self_attention (`bool`, *optional*, defaults to `False`):
            Whether to use self-attention to aggregate information from audio-tokens before sending to the text attention layer.
        audio_num_codebooks (`int`, *optional*, defaults to 8):
            The number of codebooks in RVQGAN.
        audio_codebook_size (`int`, *optional*, defaults to 1024):
            The size of each codebook in RVQGAN.
            The id of the bos in the audio stream
            The id of the eos in the audio stream
        audio_stream_bos_id (`int`, *optional*, defaults to 1024):
            The token ID in the audio codebook representing the beginning of a streaming audio sequence.
            Used to signal the start of an audio stream input when generating
            audio tokens in the model.
        audio_stream_eos_id (`int`, *optional*, defaults to 1025):
            The token ID in the audio codebook representing the end of a streaming audio sequence.
            Used to signal the end of an audio stream input when generating
            audio tokens in the model.
        audio_bos_token (`str`, *optional*, defaults to `"<|audio_bos|>"`):
            The special `<|audio_bos|>` token. In Higgs-Audio, it is mapped to 128011,
            which is the index of `<|reserved_special_token_3|>` in Llama-3.1-8B-Instruct's tokenizer.
        audio_eos_token (`str`, *optional*, defaults to `"<|audio_eos|>"`):
            The special `<|audio_eos|>` token. We use 128012 as the default value,
            which is the index of `<|reserved_special_token_4|>` in Llama-3.1-8B-Instruct's tokenizer.
        audio_out_bos_token (`str`, *optional*, defaults to `"<|audio_out_bos|>"`):
            The special `<|audio_out_bos|>` token. We use 128013 as the default value,
            which is the index of `<|reserved_special_token_5|>` in Llama-3.1-8B-Instruct's tokenizer.
        audio_in_token (`str`, *optional*, defaults to `"<|AUDIO|>"`):
            The special `<|AUDIO|>` token. We use 128015 as the default value,
            which is the index of `<|reserved_special_token_7|>` in Llama-3.1-8B-Instruct's tokenizer.
            This token indicates that the location should be filled in with whisper features.
        audio_out_token (`str`, *optional*, defaults to `"<|AUDIO_OUT|>"`):
            The special `<|AUDIO_OUT|>` token. We use 128016 as the default value,
            which is the index of `<|reserved_special_token_8|>` in Llama-3.1-8B-Instruct's tokenizer.
            This token indicates that the location should be filled in with audio tokens extracted via audio tokenizer.
        audio_in_token_idx (`int`, *optional*, defaults to 128015):
            The token ID corresponding to `audio_in_token` ("<|AUDIO|>").
            Used to indicate positions in the input sequence where audio features
            (e.g., whisper features) should be inserted.
        audio_out_token_idx (`int`, *optional*, defaults to 128016):
            The token ID corresponding to `audio_out_token` ("<|AUDIO_OUT|>").
            Used to indicate positions in the output sequence where audio tokens
            generated by the audio tokenizer should appear.
        pad_token_id (`int`, *optional*, defaults to 128001):
            The token ID used for padding sequences to a fixed length.
        audio_out_bos_token_id (`int`, *optional*, defaults to 128013):
            The token ID corresponding to `audio_out_bos_token` ("<|audio_out_bos|>").
            Marks the beginning of an audio output segment.
        audio_eos_token_id (`int`, *optional*, defaults to 128012):
            The token ID corresponding to `audio_eos_token` ("<|audio_eos|>").
            Marks the end of an audio segment.

    """

    model_type = "higgs_audio"
    is_composition = True

    sub_configs = {
        "text_config": AutoConfig,
        "audio_encoder_config": HiggsAudioEncoderConfig,
    }

    def __init__(
        self,
        text_config=None,
        audio_encoder_config=None,
        audio_adapter_type="stack",
        audio_embed_avg=False,
        audio_dual_ffn_layers=None,
        encode_audio_in_tokens=False,
        use_delay_pattern=False,
        skip_audio_tower=False,
        use_audio_out_embed_projector=False,
        use_audio_out_self_attention=False,
        audio_num_codebooks=8,
        audio_codebook_size=1024,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_out_bos_token="<|audio_out_bos|>",
        audio_in_token="<|AUDIO|>",
        audio_out_token="<|AUDIO_OUT|>",
        audio_in_token_idx=128015,
        audio_out_token_idx=128016,
        pad_token_id=128001,
        audio_out_bos_token_id=128013,
        audio_eos_token_id=128012,
        **kwargs,
    ):
        if isinstance(audio_encoder_config, dict):
            audio_encoder_config = HiggsAudioEncoderConfig(**audio_encoder_config)
        elif audio_encoder_config is None:
            audio_encoder_config = HiggsAudioEncoderConfig()

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        assert audio_adapter_type in [
            "stack",
            "dual_ffn",
            "dual_ffn_fast_forward",
        ], f"Invalid audio adapter type: {audio_adapter_type}"
        if audio_adapter_type.startswith("dual_ffn"):
            assert audio_dual_ffn_layers is not None, (
                "audio_dual_ffn_layers must be specified when using dual_ffn adapter."
            )
        self.text_config = text_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_adapter_type = audio_adapter_type
        self.audio_embed_avg = audio_embed_avg
        self.audio_dual_ffn_layers = audio_dual_ffn_layers
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.use_delay_pattern = use_delay_pattern
        self.skip_audio_tower = skip_audio_tower
        self.use_audio_out_embed_projector = use_audio_out_embed_projector
        self.use_audio_out_self_attention = use_audio_out_self_attention

        self.audio_num_codebooks = audio_num_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_out_bos_token = audio_out_bos_token
        self.audio_in_token = audio_in_token
        self.audio_out_token = audio_out_token
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id

        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id


__all__ = ["HiggsAudioEncoderConfig", "HiggsAudioConfig"]
