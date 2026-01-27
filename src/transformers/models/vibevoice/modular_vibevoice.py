# Copyright 2026 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import AutoModel
from ..llama.modeling_llama import LlamaMLP
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from ..qwen2.modeling_qwen2 import Qwen2RMSNorm
from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerEncoder,
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceAcousticTokenizerPreTrainedModel,
)
from .configuration_vibevoice import VibeVoiceConfig, VibeVoiceSemanticTokenizerConfig
from .generation_vibevoice import VibeVoiceGenerationMixin


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    VibeVoice base model outputs.
    """
)
class VibeVoiceBaseModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    audio_features (`torch.FloatTensor` of shape `(batch_size * seq_length, hidden_size)`):
        Audio features extracted from the input audio waveform.
    """

    audio_features: torch.FloatTensor | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    VibeVoice causal language model outputs.
    """
)
class VibeVoiceCausalLMOutputWithPast(VibeVoiceBaseModelOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    diffusion_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` for diffusion are provided):
        Diffusion head loss (for acoustic token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    loss: torch.FloatTensor | None = None
    diffusion_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None


class VibeVoiceRMSNorm(Qwen2RMSNorm):
    pass


class VibeVoiceConv1dPaddingCache(MimiConv1dPaddingCache):
    pass


class VibeVoiceDiffusionHeadTimestepEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_1 = nn.Linear(config.frequency_embedding_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.layer_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, timesteps, max_period=10000):
        dim = self.config.frequency_embedding_size // 2
        freq = torch.exp(-math.log(max_period) * torch.arange(dim, dtype=torch.float32) / dim)
        args = timesteps[:, None].float() * freq[None].to(timesteps.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.config.frequency_embedding_size % 2:
            embedding = nn.functional.pad(embedding, (0, 1))
        return self.layer_2(self.act(self.layer_1(embedding.to(timesteps.dtype))))


class VibeVoiceMLP(LlamaMLP):
    pass


class VibeVoiceDiffusionHeadAdaLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_chunks = 3
        self.ffn = VibeVoiceMLP(config)
        self.norm = VibeVoiceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear = nn.Linear(config.hidden_size, config.hidden_size * self.num_chunks, bias=False)

    def forward(self, hidden_states, condition):
        shift_ffn, scale_ffn, gate_ffn = self.linear(self.act_fn(condition)).chunk(self.num_chunks, dim=-1)
        modulated_hidden_states = self.norm(hidden_states) * (1 + scale_ffn) + shift_ffn
        hidden_states = hidden_states + gate_ffn * self.ffn(modulated_hidden_states)
        return hidden_states


class VibeVoiceDiffusionHeadFinalLayer(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.num_chunks = 2
        # Inline RMS normalization since there is no weight scaling (unlike `VibeVoiceRMSNorm`)
        self.norm_eps = config.rms_norm_eps
        self.linear_1 = nn.Linear(config.hidden_size, self.num_chunks * config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, output_size, bias=False)

    def forward(self, hidden_states, condition):
        shift, scale = self.linear_1(self.act_fn(condition)).chunk(self.num_chunks, dim=-1)
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class VibeVoiceDiffusionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.noisy_images_proj = nn.Linear(
            config.acoustic_tokenizer_config.hidden_size, config.hidden_size, bias=False
        )
        self.cond_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.timestep_embedder = VibeVoiceDiffusionHeadTimestepEmbedder(config)
        self.layers = nn.ModuleList(
            [VibeVoiceDiffusionHeadAdaLayerNorm(config) for _ in range(config.num_head_layers)]
        )
        self.final_layer = VibeVoiceDiffusionHeadFinalLayer(
            config, output_size=config.acoustic_tokenizer_config.hidden_size
        )

    def forward(self, noisy_images, timesteps, condition):
        """
        Forward pass of the prediction head.

        Args:
            noisy_images (`torch.Tensor`): Noisy images/latents to denoise
            timesteps (`torch.Tensor`): Timesteps for diffusion
            condition (`torch.Tensor`): Conditioning information

        Returns:
            `torch.Tensor`: The predicted noise/velocity
        """
        hidden_states = self.noisy_images_proj(noisy_images)
        embedded_timesteps = self.timestep_embedder(timesteps)
        condition = self.cond_proj(condition)
        condition = condition + embedded_timesteps
        for layer in self.layers:
            hidden_states = layer(hidden_states, condition)
        hidden_states = self.final_layer(hidden_states, condition)
        return hidden_states


class VibeVoiceMultiModelProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = VibeVoiceRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features):
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


@auto_docstring
class VibeVoicePreTrainedModel(VibeVoiceAcousticTokenizerPreTrainedModel):
    config: VibeVoiceConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["VibeVoiceEncoderLayer"]
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if hasattr(module, "latent_scaling_factor"):
            nn.init.constant_(module.latent_scaling_factor, 1.0)
        if hasattr(module, "latent_bias_factor"):
            nn.init.constant_(module.latent_bias_factor, 0.0)


@dataclass
@auto_docstring
class VibeVoiceSemanticTokenizerOutput(ModelOutput):
    """
    latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for semantic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceConv1dPaddingCache`, *optional*, returned when `use_cache=True` is passed):
        A [`VibeVoiceConv1dPaddingCache`] instance containing cached convolution states for each layer that
        can be passed to subsequent forward calls.
    """

    latents: torch.FloatTensor | None = None
    padding_cache: Optional["VibeVoiceConv1dPaddingCache"] = None


class VibeVoiceEncoder(VibeVoiceAcousticTokenizerEncoder):
    def __init__(self, config):
        super().__init__(config)

        # Parameters for cache creation
        self.num_conv_layers = sum(depth + 1 for depth in config.depths) + 1
        self.per_conv_layer_padding = [self.stem.conv.causal_padding]
        self.per_conv_layer_in_channels = [self.stem.conv.conv.in_channels]
        self.per_conv_layer_padding.extend([block.mixer.causal_padding for block in self.stem.stage])
        self.per_conv_layer_in_channels.extend([block.mixer.conv.in_channels for block in self.stem.stage])

        for layer in self.conv_layers:
            self.per_conv_layer_padding.append(layer.conv.causal_padding)
            self.per_conv_layer_in_channels.append(layer.conv.conv.in_channels)
            self.per_conv_layer_padding.extend([block.mixer.causal_padding for block in layer.stage])
            self.per_conv_layer_in_channels.extend([block.mixer.conv.in_channels for block in layer.stage])

        self.per_conv_layer_padding.append(self.head.causal_padding)
        self.per_conv_layer_in_channels.append(self.head.conv.in_channels)
        self.per_conv_layer_padding_mode = ["constant" for _ in self.per_conv_layer_padding]


@auto_docstring(
    custom_intro="""
    Semantic tokenizer which only encodes audio into semantic tokens, namely no decoding.
    """
)
class VibeVoiceSemanticTokenizerModel(VibeVoiceAcousticTokenizerModel):
    config: VibeVoiceSemanticTokenizerConfig
    base_model_prefix = "vibevoice_semantic_tokenizer"
    main_input_name = "audio"
    _no_split_modules = ["VibeVoiceSemanticTokenizerEncoder"]

    def __init__(self, config):
        super().__init__(config)
        del self.decoder

    @can_return_tuple
    @auto_docstring
    def encode(self, audio, padding_cache=None, use_cache=None):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        padding_cache (`VibeVoiceConv1dPaddingCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceConv1dPaddingCache(
                num_layers=self.encoder.num_conv_layers,
                per_layer_padding=self.encoder.per_conv_layer_padding,
                per_layer_padding_mode=self.encoder.per_conv_layer_padding_mode,
                per_layer_in_channels=self.encoder.per_conv_layer_in_channels,
            )
        latents = self.encoder(audio, padding_cache=padding_cache)

        return VibeVoiceSemanticTokenizerOutput(
            latents=latents,
            padding_cache=padding_cache if use_cache else None,
        )

    def sample(self):
        raise NotImplementedError("Sample method is not implemented for VibeVoiceSemanticTokenizerModel.")

    def decode(self, latents, padding_cache=None, use_cache=False):
        raise NotImplementedError("Decode method is not implemented for VibeVoiceSemanticTokenizerModel.")

    def forward(self, audio, padding_cache=None, use_cache=None, **kwargs: Unpack[TransformersKwargs]):
        raise NotImplementedError("Forward method is not implemented for VibeVoiceSemanticTokenizerModel.")


@auto_docstring(
    custom_intro="""
    The VibeVoice model which consists of audio tokenizers and an LLM backbone, without a language modeling head.
    """
)
class VibeVoiceModel(VibeVoicePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = AutoModel.from_config(config.semantic_tokenizer_config)
        self.acoustic_connector = VibeVoiceMultiModelProjector(
            config.acoustic_tokenizer_config.hidden_size, config.text_config.hidden_size
        )
        self.semantic_connector = VibeVoiceMultiModelProjector(
            config.semantic_tokenizer_config.hidden_size, config.text_config.hidden_size
        )
        self.diffusion_head = VibeVoiceDiffusionHead(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_audio_features(self, input_values, padding_mask, latent_scaling_factor, latent_bias_factor):
        """
        This method is used to get the audio embeddings from the input features (normalized audio).

        Args:
            input_values (`torch.FloatTensor`):
                Float values of (normalized) audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, padded_audio_length)`):
                Padding mask to remove padded parts of audio.
            latent_scaling_factor (`torch.FloatTensor`):
                Scaling factor for acoustic latents.
            latent_bias_factor (`torch.FloatTensor`):
                Bias factor for acoustic latents.

        Returns:
            `torch.FloatTensor`:
                The audio embeddings.
        """
        # adjust padding mask according to tokenizer compression
        num_audio_tokens = torch.ceil(padding_mask.sum(dim=-1) / self.acoustic_tokenizer.config.hop_length).to(
            torch.int64
        )
        padding_mask = torch.arange(max(num_audio_tokens)) < num_audio_tokens[:, None].cpu()

        with torch.no_grad():
            acoustic_latents = self.acoustic_tokenizer.encode(input_values).latents
            acoustic_latents = self.acoustic_tokenizer.sample(acoustic_latents).latents
        acoustic_features = (
            acoustic_latents + latent_bias_factor.to(acoustic_latents.device)
        ) * latent_scaling_factor.to(acoustic_latents.device)
        return self.acoustic_connector(acoustic_features)[padding_mask], acoustic_features[padding_mask]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        latent_scaling_factor: torch.FloatTensor | None = None,
        latent_bias_factor: torch.FloatTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_features = None
        if input_values is not None and input_ids is not None:
            audio_embeds, audio_features = self.get_audio_features(
                input_values, padding_mask, latent_scaling_factor, latent_bias_factor
            )

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_diffusion_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs = self.language_model(inputs_embeds=inputs_embeds, **kwargs)
        return VibeVoiceBaseModelOutputWithPast(audio_features=audio_features, **outputs)


@auto_docstring(
    custom_intro="""
    The VibeVoice model, which consists of a language model, audio tokenizers, connectors, and a diffusion head.
    """
)
class VibeVoiceForConditionalGeneration(VibeVoicePreTrainedModel, VibeVoiceGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceModel(config)
        self.latent_scaling_factor = nn.Parameter(torch.tensor(1.0))
        self.latent_bias_factor = nn.Parameter(torch.tensor(0.0))
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | slice = 0,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        acoustic_loss_mask: torch.BoolTensor | None = None,
        **kwargs,
    ) -> tuple | VibeVoiceCausalLMOutputWithPast:
        r"""
        input_values (`torch.FloatTensor`, *optional*):
            Preprocessed audio waveform for voice cloning.
        padding_mask (`torch.BoolTensor`, *optional*):
            Masks indicating valid input frames.
        acoustic_loss_mask (`torch.BoolTensor`, *optional*):
            Mask to compute diffusion loss only on specific acoustic tokens.
        """

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            input_values=input_values,
            padding_mask=padding_mask,
            latent_scaling_factor=self.latent_scaling_factor,
            latent_bias_factor=self.latent_bias_factor,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        diffusion_loss = None
        if acoustic_loss_mask is not None and outputs.audio_features is not None:
            raise NotImplementedError("Diffusion loss computation is not yet implemented.")

        return VibeVoiceCausalLMOutputWithPast(loss=loss, diffusion_loss=diffusion_loss, logits=logits, **outputs)


__all__ = [
    "VibeVoiceForConditionalGeneration",
    "VibeVoicePreTrainedModel",
    "VibeVoiceModel",
    "VibeVoiceSemanticTokenizerModel",
]
