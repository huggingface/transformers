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
import numpy as np

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ..auto import CONFIG_MAPPING, AutoConfig
from ...modeling_outputs import BaseModelOutputWithPast
from ...utils import auto_docstring, can_return_tuple
from ..auto import AutoModel
from ..llama.modeling_llama import LlamaMLP
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from ..qwen2.modeling_qwen2 import Qwen2RMSNorm
from ..vibevoice_acoustic_tokenizer.configuration_vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerConfig
from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceAcousticTokenizerPreTrainedModel,
    VibeVoiceAcousticTokenizerEncoderOutput,
)
from .configuration_vibevoice import VibeVoiceConfig, VibeVoiceSemanticTokenizerConfig
from .generation_vibevoice import VibeVoiceGenerationMixin
from ...configuration_utils import PretrainedConfig


# TODO after VibeVoice ASR is merged: https://github.com/huggingface/transformers/pull/43625
# can use the encoder only object `VibeVoiceAsrEncoderModel` from there
class VibeVoiceSemanticTokenizerConfig(VibeVoiceAcousticTokenizerConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceSemanticTokenizerModel`]. It is used to
    instantiate a VibeVoice semantic tokenizer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    semantic tokenizer of [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        channels (`int`, *optional*, defaults to 1):
            Number of input channels.
        hidden_size (`int`, *optional*, defaults to 128):
            Dimensionality of latent representations.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for convolutional layers.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon value for RMSNorm layers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
            Initial value for layer scaling.
        initializer_range (`float`, *optional*, defaults to 0.01):
            Standard deviation for weight initialization.
        num_filters (`int`, *optional*, defaults to 32):
            Number of filters in initial convolutional layer, and doubles after each downsampling.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
            Downsampling ratios for each layer.
        depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
            Number of ConvNeXt blocks at each stage.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        ffn_expansion (`int`, *optional*, defaults to 4):
            Expansion factor for feed-forward networks.

    ```python
    >>> from transformers import VibeVoiceSemanticTokenizerModel, VibeVoiceSemanticTokenizerConfig

    >>> # Initializing a VibeVoice Semantic Tokenizer configuration
    >>> configuration = VibeVoiceSemanticTokenizerConfig()

    >>> # Initializing a model (with random weights)
    >>> model = VibeVoiceSemanticTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels=1,
        hidden_size=128,
        kernel_size=7,
        rms_norm_eps=1e-5,
        layer_scale_init_value=1e-6,
        initializer_range=1e-2,
        num_filters=32,
        downsampling_ratios=[2, 2, 4, 5, 5, 8],
        depths=[3, 3, 3, 3, 3, 3, 8],
        hidden_act="gelu",
        ffn_expansion=4,
        **kwargs,
    ):
        super().__init__(
            channels=channels,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            rms_norm_eps=rms_norm_eps,
            layer_scale_init_value=layer_scale_init_value,
            initializer_range=initializer_range,
            num_filters=num_filters,
            downsampling_ratios=downsampling_ratios,
            depths=depths,
            hidden_act=hidden_act,
            ffn_expansion=ffn_expansion,
            **kwargs,
        )

        del self.vae_std

    def upsampling_ratios(self):
        raise NotImplementedError("VibeVoiceAsrEncoderConfig does not need upsampling_ratios.")

    def decoder_depths(self):
        raise NotImplementedError("VibeVoiceAsrEncoderConfig does not need decoder_depths.")
    

class VibeVoiceConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceForConditionalGeneration`]. It is used to instantiate an
    VibeVoice model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults similar to that of [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        acoustic_tokenizer_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the acoustic tokenizer.
        semantic_tokenizer_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the semantic tokenizer.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text model.
        pad_token_id (`int`, *optional*, defaults to 151643):
            The token ID for padding.
        eos_token_id (`int`, *optional*, defaults to 151643):
            The token ID for the end of sequence.
        audio_bos_token_id (`int`, *optional*, defaults to 151652):
            The token ID indicating the start of audio tokens.
        audio_eos_token_id (`int`, *optional*, defaults to 151653):
            The token ID indicating the end of audio tokens.
        audio_diffusion_token_id (`int`, *optional*, defaults to 151654):
            The token ID indicating the start of audio diffusion tokens.
        num_head_layers (`int`, *optional*, defaults to 4):
            Number of layers in the diffusion head.
        intermediate_size (`int`, *optional*, defaults to 4608):
            The intermediate size of the feed-forward layers.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMSNorm layers.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The activation function used by the diffusion head.
        frequency_embedding_size (`int`, *optional*, defaults to 256):
            The size of the frequency embedding.

    ```python
    >>> from transformers import VibeVoiceForConditionalGeneration, VibeVoiceConfig

    >>> # Initializing a VibeVoice configuration
    >>> configuration = VibeVoiceConfig()

    >>> # Initializing a 1.5B model with random weights
    >>> model = VibeVoiceForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice"
    is_composition = True

    sub_configs = {
        "acoustic_tokenizer_config": AutoConfig,
        "semantic_tokenizer_config": AutoConfig,
        "text_config": AutoConfig,
    }

    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "language_model.layers.*.self_attn.q_proj": "colwise",
        "language_model.layers.*.self_attn.k_proj": "colwise",
        "language_model.layers.*.self_attn.v_proj": "colwise",
        "language_model.layers.*.self_attn.o_proj": "rowwise",
        "language_model.layers.*.mlp.gate_proj": "colwise",
        "language_model.layers.*.mlp.up_proj": "colwise",
        "language_model.layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        text_config=None,
        pad_token_id=151643,
        eos_token_id=151643,
        audio_bos_token_id=151652,
        audio_eos_token_id=151653,
        audio_diffusion_token_id=151654,
        num_head_layers=4,
        intermediate_size=4608,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        frequency_embedding_size=256,
        **kwargs,
    ):
        if isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = acoustic_tokenizer_config.get(
                "model_type", "vibevoice_acoustic_tokenizer"
            )
            acoustic_tokenizer_config = CONFIG_MAPPING[acoustic_tokenizer_config["model_type"]](
                **acoustic_tokenizer_config
            )
        elif acoustic_tokenizer_config is None:
            acoustic_tokenizer_config = CONFIG_MAPPING["vibevoice_acoustic_tokenizer"]()
        self.acoustic_tokenizer_config = acoustic_tokenizer_config

        # TODO after VibeVoice ASR is merged: https://github.com/huggingface/transformers/pull/43625
        # can use the encoder only object `VibeVoiceAsrEncoderModel` from there
        if isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = semantic_tokenizer_config.get(
                "model_type", "vibevoice_semantic_tokenizer"
            )
            semantic_tokenizer_config = CONFIG_MAPPING[semantic_tokenizer_config["model_type"]](
                **semantic_tokenizer_config
            )
        elif semantic_tokenizer_config is None:
            semantic_tokenizer_config = CONFIG_MAPPING["vibevoice_semantic_tokenizer"]()
        self.semantic_tokenizer_config = semantic_tokenizer_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()
        self.text_config = text_config

        self.vocab_size = text_config.vocab_size
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_diffusion_token_id = audio_diffusion_token_id
        self.num_head_layers = num_head_layers
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.frequency_embedding_size = frequency_embedding_size

        # NOTE (ebezzam) to use LlamaMLP via modular
        self.mlp_bias = False
        self.intermediate_size = intermediate_size

        kwargs.pop("tie_word_embeddings", None)  # remove if present to take priority from text_config
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=getattr(text_config, "tie_word_embeddings", False),
            **kwargs,
        )

    @property
    def initializer_range(self) -> float:
        return self.acoustic_tokenizer_config.initializer_range

    @property
    def layer_scale_init_value(self) -> float:
        return self.acoustic_tokenizer_config.layer_scale_init_value

    # NOTE (ebezzam) for modular usage of `LlamaMLP`
    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size


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


class VibeVoiceMultiModalProjector(nn.Module):
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
    _no_split_modules = None
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    # TODO (ebezzam) check below
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if hasattr(module, "latent_scaling_factor"):
            nn.init.constant_(module.latent_scaling_factor, 1.0)
        if hasattr(module, "latent_bias_factor"):
            nn.init.constant_(module.latent_bias_factor, 0.0)


class VibeVoiceSemanticTokenizerOutput(VibeVoiceAcousticTokenizerEncoderOutput):
    pass


# TODO after VibeVoice ASR is merged: https://github.com/huggingface/transformers/pull/43625
# can use the encoder only object `VibeVoiceAsrEncoderModel` from there
@auto_docstring(
    custom_intro="""
    Semantic tokenizer which only encodes audio into semantic tokens, namely no decoding.
    """
)
class VibeVoiceSemanticTokenizerModel(VibeVoiceAcousticTokenizerModel):
    config: VibeVoiceSemanticTokenizerConfig
    main_input_name = "input_values"
    input_modalities = "audio"

    def __init__(self, config):
        super().__init__(config)
        del self.decoder

    def encode(self, input_values, padding_cache=None, use_cache=None, sample=True):
        raise NotImplementedError("Encode method is not implemented.")

    def decode(self, latents, padding_cache=None, use_cache=False):
        raise NotImplementedError("Decode method is not implemented.")

    def forward(self, input_values, padding_cache=None, use_cache=None, **kwargs):
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        padding_cache (`VibeVoiceConv1dPaddingCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        if use_cache and padding_cache is None:
            per_layer_padding = [self.encoder.stem.conv.causal_padding]
            per_layer_in_channels = [self.encoder.stem.conv.conv.in_channels]
            per_layer_padding.extend([block.mixer.causal_padding for block in self.encoder.stem.stage])
            per_layer_in_channels.extend([block.mixer.conv.in_channels for block in self.encoder.stem.stage])
            for layer in self.encoder.conv_layers:
                per_layer_padding.append(layer.conv.causal_padding)
                per_layer_in_channels.append(layer.conv.conv.in_channels)
                per_layer_padding.extend([block.mixer.causal_padding for block in layer.stage])
                per_layer_in_channels.extend([block.mixer.conv.in_channels for block in layer.stage])
            per_layer_padding.append(self.encoder.head.causal_padding)
            per_layer_in_channels.append(self.encoder.head.conv.in_channels)

            padding_cache = VibeVoiceConv1dPaddingCache(
                num_layers=len(per_layer_padding),
                per_layer_padding=per_layer_padding,
                per_layer_padding_mode=["constant"] * len(per_layer_padding),
                per_layer_in_channels=per_layer_in_channels,
            )
        latents = self.encoder(input_values, padding_cache=padding_cache)

        return VibeVoiceSemanticTokenizerOutput(
            latents=latents,
            padding_cache=padding_cache if use_cache else None,
        )


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
        self.acoustic_connector = VibeVoiceMultiModalProjector(
            config.acoustic_tokenizer_config.hidden_size, config.text_config.hidden_size
        )
        self.semantic_connector = VibeVoiceMultiModalProjector(
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
        hop_length = np.prod(self.acoustic_tokenizer.config.downsampling_ratios)
        num_audio_tokens = torch.ceil(padding_mask.sum(dim=-1) / hop_length).to(torch.int64)
        padding_mask = torch.arange(max(num_audio_tokens)) < num_audio_tokens[:, None].cpu()

        with torch.no_grad():
            acoustic_latents = self.acoustic_tokenizer.encode(input_values, sample=True).latents
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
    "VibeVoiceConfig", 
    "VibeVoiceSemanticTokenizerConfig",
    "VibeVoiceForConditionalGeneration",
    "VibeVoicePreTrainedModel",
    "VibeVoiceModel",
    "VibeVoiceSemanticTokenizerModel",
]
