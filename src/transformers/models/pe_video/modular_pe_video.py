from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModelForImageClassification
from ..pe_audio.modeling_pe_audio import PEAudioEncoderEmbeddings
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from ..timm_wrapper import TimmWrapperConfig
from .configuration_pe_video import PEVideoConfig, PEVideoEncoderConfig


class PEVideoEncoderConfig(Qwen3Config):
    model_type = "pe_video_encoder"
    sub_configs = {"vision_config": TimmWrapperConfig}

    _default_vision_config_kwargs = {
        "architecture": "vit_pe_core_large_patch14_336",
        "do_pooling": True,
        "num_classes": 1024,
        "global_pool": "map",
        "initializer_range": 0.02,
    }

    def __init__(
        self,
        vision_config=None,
        hidden_size=1792,
        intermediate_size=4800,
        num_hidden_layers=4,
        num_attention_heads=14,
        num_key_value_heads=14,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=10000,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_theta=20000,
        rope_scaling=None,
        attention_bias=False,
        max_window_layers=28,
        attention_dropout=0.0,
        sliding_window=None,
        use_sliding_window=False,
        layer_types=None,
        tie_word_embeddings=False,
        vocab_size=None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "timm_wrapper")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]].from_dict(
                {**self._default_vision_config_kwargs, **vision_config}
            )
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["timm_wrapper"].from_dict(self._default_vision_config_kwargs)

        self.vision_config = vision_config

        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            vocab_size=vocab_size,
            layer_types=layer_types,
            tie_word_embeddings=tie_word_embeddings,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            **kwargs,
        )


class PEVideoConfig(PretrainedConfig):
    model_type = "pe_video"
    sub_configs = {"text_config": AutoConfig, "video_config": PEVideoEncoderConfig}

    def __init__(
        self,
        text_config=None,
        video_config=None,
        projection_dim=1024,
        nth_text_layer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "modernbert")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["modernbert"]()
            # TODO: add log

        if isinstance(video_config, dict):
            video_config = PEVideoEncoderConfig(**video_config)
        elif video_config is None:
            video_config = PEVideoEncoderConfig()
            # TODO: add log

        self.text_config = text_config
        self.video_config = video_config

        self.projection_dim = projection_dim
        self.nth_text_layer = nth_text_layer


class PEVideoEncoderEmbeddings(PEAudioEncoderEmbeddings): ...


class PEVideoEncoderAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.sliding_window = None


class PEVideoEncoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        del self.attention_type


class PEVideoRMSNorm(Qwen3RMSNorm): ...


class PEVideoRotaryEmbedding(Qwen3RotaryEmbedding): ...


@auto_docstring
class PEVideoPreTrainedModel(PreTrainedModel):
    config: PEVideoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PEVideoEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": PEVideoEncoderLayer,
        "attentions": PEVideoEncoderAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)

        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, PEVideoEncoderEmbeddings):
            embed_dim = module.class_embedding.shape[-1]
            nn.init.normal_(module.class_embedding, mean=0.0, std=embed_dim**-0.5 * std)


@auto_docstring(
    custom_intro="""
    The PEVideo Encoder model.
    """
)
class PEVideoEncoder(PEVideoPreTrainedModel):
    config: PEVideoEncoderConfig

    def __init__(self, config: PEVideoEncoderConfig):
        super().__init__(config)
        # Vision feature extraction stack (pre-embeddings)
        # NOTE: we use `AutoModelForImageClassification` instead of `AutoModel`
        # because `TimmWrapperModel` forces `num_classes=0` which drops the final linear projection.
        self.vision_model = AutoModelForImageClassification.from_config(config.vision_config)
        self.proj = nn.Linear(config.vision_config.num_labels, config.hidden_size, bias=False)
        self.data_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # TODO: should it be named patch_embedding?
        self.embeddings = PEVideoEncoderEmbeddings(config)
        self.layers = nn.ModuleList(
            [PEVideoEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PEVideoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PEVideoRotaryEmbedding(config=config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask_videos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = pixel_values_videos.shape

        pixel_values_videos = pixel_values_videos.view(-1, *input_shape[2:])
        vision_encoder_outputs = self.vision_model(pixel_values_videos)

        logits = vision_encoder_outputs.logits.view(*input_shape[:2], -1)
        logits = F.normalize(logits, dim=-1)

        vision_features = self.proj(logits)
        inputs_embeds = self.data_proj(vision_features)

        return inputs_embeds, padding_mask_videos

    @can_return_tuple
    @check_model_inputs
    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask_videos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        inputs_embeds, attention_mask = self.get_video_features(
            pixel_values_videos,
            padding_mask_videos=padding_mask_videos,
        )
        inputs_embeds, attention_mask = self.embeddings(inputs_embeds, padding_mask=attention_mask)

        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states[:, 1:],
            pooler_output=hidden_states[:, 0],
        )


__all__ = [
    "PEVideoEncoder",
    "PEVideoEncoderConfig",
    "PEVideoConfig",
]
