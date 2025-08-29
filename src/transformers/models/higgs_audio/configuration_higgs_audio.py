# coding=utf-8
# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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


class HiggsAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class for the HiggsAudioModel. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) architecture.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text backbone.
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
        use_audio_out_embed_projector (`bool`, *optional*, defaults to `False`):
            Whether to use an embedding projector to map audio out embeddings.
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

    sub_configs = {"text_config": AutoConfig}

    def __init__(
        self,
        text_config=None,
        audio_adapter_type="stack",
        audio_embed_avg=False,
        audio_dual_ffn_layers=None,
        encode_audio_in_tokens=False,
        use_delay_pattern=False,
        use_audio_out_embed_projector=False,
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
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        if audio_adapter_type not in [
            "stack",
            "dual_ffn",
            "dual_ffn_fast_forward",
        ]:
            raise ValueError("Invalid audio adapter type: {audio_adapter_type}")
        if audio_adapter_type.startswith("dual_ffn"):
            if audio_dual_ffn_layers is None:
                raise ValueError(
                    "audio_dual_ffn_layers must be specified when using dual_ffn adapter."
                )
        self.text_config = text_config
        self.audio_adapter_type = audio_adapter_type
        self.audio_embed_avg = audio_embed_avg
        self.audio_dual_ffn_layers = audio_dual_ffn_layers
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.use_delay_pattern = use_delay_pattern
        self.use_audio_out_embed_projector = use_audio_out_embed_projector

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

        super().__init__(pad_token_id=pad_token_id, **kwargs)


__all__ = ["HiggsAudioConfig"]
