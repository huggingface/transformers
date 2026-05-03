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
"""PyTorch Qwen3TTS Multi-Codebook Tokenizer model."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ...cache_utils import DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...utils import auto_docstring, logging
from ..mimi.modeling_mimi import (
    MimiEncoderOutput,
    MimiEuclideanCodebook,
    MimiModel,
    MimiPreTrainedModel,
    MimiResidualVectorQuantizer,
    MimiSplitResidualVectorQuantizer,
    MimiVectorQuantization,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import SnakeBeta
from ..qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavAttention,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavDecoderResidualUnit,
    Qwen3OmniMoeCode2WavMlp,
    Qwen3OmniMoeCode2WavRMSNorm,
    Qwen3OmniMoeCode2WavTransformerLayer,
    Qwen3OmniMoeConvNeXtBlock,
)
from .configuration_qwen3_tts_tokenizer_multi_codebook import (
    Qwen3TTSTokenizerMultiCodebookCode2WavConfig,
    Qwen3TTSTokenizerMultiCodebookConfig,
)


logger = logging.get_logger(__name__)


# ─── Component Aliases ────────────────────────────────────────────────────────


class Qwen3TTSTokenizerMultiCodebookCausalConvNet(Qwen3OmniMoeCausalConvNet):
    pass


class Qwen3TTSTokenizerMultiCodebookCausalTransConvNet(Qwen3OmniMoeCausalTransConvNet):
    pass


class Qwen3TTSTokenizerMultiCodebookConvNeXtBlock(Qwen3OmniMoeConvNeXtBlock):
    pass


class Qwen3TTSTokenizerMultiCodebookRotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class Qwen3TTSTokenizerMultiCodebookAttention(Qwen3OmniMoeCode2WavAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rotary_emb = Qwen3TTSTokenizerMultiCodebookRotaryEmbedding(config=config)


class Qwen3TTSTokenizerMultiCodebookRMSNorm(Qwen3OmniMoeCode2WavRMSNorm):
    pass


class Qwen3TTSTokenizerMultiCodebookMlp(Qwen3OmniMoeCode2WavMlp):
    pass


class Qwen3TTSTokenizerMultiCodebookResidualUnit(Qwen3OmniMoeCode2WavDecoderResidualUnit):
    pass


class Qwen3TTSTokenizerMultiCodebookBlock(Qwen3OmniMoeCode2WavTransformerLayer):
    pass


# ─── Output dataclasses ───────────────────────────────────────────────────────


class Qwen3TTSTokenizerMultiCodebookEncoderOutput(MimiEncoderOutput):
    pass


@dataclass
@auto_docstring
class Qwen3TTSTokenizerMultiCodebookOutput(ModelOutput):
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerMultiCodebook.
        Each tensor has shape (segment_length_i).
    """

    audio_values: list[torch.FloatTensor] = None


# ─── PreTrainedModel base ─────────────────────────────────────────────────────


class Qwen3TTSTokenizerMultiCodebookPreTrainedModel(MimiPreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False


class Qwen3TTSTokenizerMultiCodebookDecoderPreTrainedModel(Qwen3TTSTokenizerMultiCodebookPreTrainedModel):
    config_class = Qwen3TTSTokenizerMultiCodebookCode2WavConfig
    _no_split_modules = ["Qwen3TTSTokenizerMultiCodebookBlock"]


# ─── Transformer model (decoder side) ────────────────────────────────────────


class Qwen3TTSTokenizerMultiCodebookDecoderTransformerModel(Qwen3TTSTokenizerMultiCodebookDecoderPreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerMultiCodebookCode2WavConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3TTSTokenizerMultiCodebookBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSTokenizerMultiCodebookRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTokenizerMultiCodebookRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.window_size = config.sliding_window
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)
        self.post_init()

    @auto_docstring(
        custom_args="""
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Used to update the cache
            in the correct position and to infer the complete sequence length.
        """
    )
    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is not None:
            inputs_embeds = self.input_proj(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ─── Decoder block ────────────────────────────────────────────────────────────


class Qwen3TTSTokenizerMultiCodebookDecoderBlock(Qwen3OmniMoeCode2WavDecoderBlock):
    pass


# ─── VQ / RVQ classes ────────────────────────────────────────────────────────


class Qwen3TTSTokenizerMultiCodebookEuclideanCodebook(MimiEuclideanCodebook):
    pass


class Qwen3TTSTokenizerMultiCodebookVectorQuantization(MimiVectorQuantization):
    pass


class Qwen3TTSTokenizerMultiCodebookResidualVectorQuantizer(MimiResidualVectorQuantizer):
    pass


class Qwen3TTSTokenizerMultiCodebookSplitResidualVectorQuantizer(MimiSplitResidualVectorQuantizer):
    pass


# ─── Decoder ─────────────────────────────────────────────────────────────────


class Qwen3TTSTokenizerMultiCodebookDecoder(Qwen3TTSTokenizerMultiCodebookDecoderPreTrainedModel):
    config_class = Qwen3TTSTokenizerMultiCodebookCode2WavConfig

    def __init__(self, config: Qwen3TTSTokenizerMultiCodebookCode2WavConfig):
        super().__init__(config)
        self.total_upsample = int(np.prod(list(config.upsample_rates) + list(config.upsampling_ratios)))
        self.pre_transformer = Qwen3TTSTokenizerMultiCodebookDecoderTransformerModel._from_config(config)

        # Bridge our decoder config into the attribute names MimiSplitResidualVectorQuantizer expects.
        quantizer_config = SimpleNamespace(
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim // 2,
            frame_rate=0,
            num_quantizers=config.num_quantizers,
            num_semantic_quantizers=1,
            vector_quantization_hidden_dimension=config.codebook_dim // 2,
            hidden_size=config.codebook_dim,
        )
        self.quantizer = Qwen3TTSTokenizerMultiCodebookSplitResidualVectorQuantizer(quantizer_config)

        self.pre_conv = Qwen3TTSTokenizerMultiCodebookCausalConvNet(
            config.codebook_dim, config.latent_dim, kernel_size=3
        )

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3TTSTokenizerMultiCodebookCausalTransConvNet(
                            config.latent_dim, config.latent_dim, factor, factor
                        ),
                        Qwen3TTSTokenizerMultiCodebookConvNeXtBlock(config.latent_dim),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3TTSTokenizerMultiCodebookCausalConvNet(config.latent_dim, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3TTSTokenizerMultiCodebookDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3TTSTokenizerMultiCodebookCausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)
        self.post_init()

    def forward(self, codes, **kwargs):
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}")
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)


# ─── Encoder (Mimi-based, encoder-only) ──────────────────────────────────────


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerMultiCodebook encoder model, based on MimiModel but only using the encoder path.
    """
)
class Qwen3TTSTokenizerMultiCodebookEncoderModel(MimiModel):
    def __init__(self, config):
        super().__init__(config)
        # Nullify decoder components — encoder-only model does not need them.
        # MimiModel.__init__ creates MimiDecoder (compatible with MimiConfig),
        # then we null it out along with upsample and decoder_transformer.
        self.decoder = None
        self.decoder_transformer = None
        self.upsample = None


# ─── Top-level Model ──────────────────────────────────────────────────────────


@auto_docstring
class Qwen3TTSTokenizerMultiCodebookModel(Qwen3TTSTokenizerMultiCodebookPreTrainedModel):
    config_class = Qwen3TTSTokenizerMultiCodebookConfig
    main_input_name = "input_values"

    def __init__(self, config: Qwen3TTSTokenizerMultiCodebookConfig):
        super().__init__(config)
        self.config = config

        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers
        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate
        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        self.encoder = Qwen3TTSTokenizerMultiCodebookEncoderModel._from_config(self.config.encoder_config)
        self.decoder = Qwen3TTSTokenizerMultiCodebookDecoder._from_config(self.config.decoder_config)

        self.post_init()

    def get_input_sample_rate(self):
        return self.input_sample_rate

    def get_output_sample_rate(self):
        return self.output_sample_rate

    def get_encode_downsample_rate(self):
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self):
        return self.decode_upsample_rate

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple[list[torch.Tensor]] | Qwen3TTSTokenizerMultiCodebookEncoderOutput:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked*
                or 0 for *masked*.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        encoded_frames = self.encoder.encode(input_values=input_values.unsqueeze(1), return_dict=True)
        audio_codes = encoded_frames.audio_codes[:, : self.encoder_valid_num_quantizers]
        audio_codes = [
            code[..., : -(-mask.sum() // self.encode_downsample_rate)].transpose(0, 1)
            for code, mask in zip(audio_codes, padding_mask)
        ]

        if not return_dict:
            return (audio_codes,)

        return Qwen3TTSTokenizerMultiCodebookEncoderOutput(audio_codes=audio_codes)

    def decode(
        self,
        audio_codes: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | Qwen3TTSTokenizerMultiCodebookOutput:
        """
        Decodes the given frames into an output audio waveform.

        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, codes_length, num_quantizers)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate

        audio_codes = torch.clamp(audio_codes, min=0)
        audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(1)
        audio_values = [a[:length] for a, length in zip(audio_values, audio_lengths)]

        if not return_dict:
            return (audio_values,)

        return Qwen3TTSTokenizerMultiCodebookOutput(audio_values)


__all__ = ["Qwen3TTSTokenizerMultiCodebookModel", "Qwen3TTSTokenizerMultiCodebookPreTrainedModel"]
