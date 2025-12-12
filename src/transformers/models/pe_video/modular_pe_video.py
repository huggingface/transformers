from typing import Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModelForImageClassification, AutoModel
from ..pe_audio.modeling_pe_audio import PEAudioContrastiveHead, PEAudioEncoderEmbeddings, PEAudioAttention
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
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
        num_key_value_heads=None,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=10000,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_parameters={
            "rope_theta": 20000,
        },
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
            rope_parameters=rope_parameters,
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


# TODO: not sure about the typing for text_model_output
@dataclass
# @auto_docstring
class PEVideoOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for video-text similarity.
    logits_per_video (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`):
        The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
        similarity scores.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`):
        The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
        similarity scores.
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`PEVideoTextModel`].
    video_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The video embeddings obtained by applying the projection layer to the pooled output of [`PEVideoVisionModel`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`PEVideoTextModel`].
    video_model_output (`BaseModelOutputWithPooling`):
        The output of the [`PEVideoVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_video: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    video_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    video_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "video_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class PEVideoContrastiveHead(PEAudioContrastiveHead): ...



class PEVideoEncoderEmbeddings(PEAudioEncoderEmbeddings): ...


class PEVideoAttention(PEAudioAttention): ...


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
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": PEVideoEncoderLayer,
        "attentions": PEVideoAttention,
    }
    _checkpoint_conversion_mapping = {
        r"^audio_video_encoder\.video_encoder": "video_encoder",
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
    base_model_prefix = "video_encoder"

    def __init__(self, config: PEVideoEncoderConfig):
        super().__init__(config)
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


class PEVideoModel(PEVideoPreTrainedModel):
    def __init__(self, config: PEVideoConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.video_encoder = PEVideoEncoder(config.video_config)

        self.text_video_head = PEVideoContrastiveHead(config.text_config.hidden_size, config.projection_dim)
        self.video_head = PEVideoContrastiveHead(config.video_config.hidden_size, config.projection_dim)

        self.video_logit_scale = nn.Parameter(torch.zeros(1))
        self.video_logit_bias = nn.Parameter(torch.zeros(1))

        self.post_init()

    def get_text_features(self, input_ids, attention_mask=None):
        # TODO: should it be named feature or embeds
        text_outputs: MaskedLMOutput = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        text_features = text_outputs.last_hidden_state
        text_features = self.text_video_head(text_features)
        return text_features

    def get_video_features(self, pixel_values_videos, padding_mask_videos=None):
        # TODO: should it be named feature or embeds
        video_outputs: BaseModelOutputWithPooling = self.video_encoder(
            pixel_values_videos=pixel_values_videos,
            padding_mask_videos=padding_mask_videos,
            return_dict=True,
        )
        video_features = self.video_head(video_outputs.pooler_output)
        return video_features

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        **kwargs,
    ) -> PEVideoOutput:
        video_output: BaseModelOutputWithPooling = self.video_encoder(
            pixel_values_videos=pixel_values_videos,
            padding_mask_videos=padding_mask_videos,
            **{**kwargs, "return_dict": True},
        )

        text_output: MaskedLMOutput = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{**kwargs, "return_dict": True},
            output_hidden_states=True,
        )

        video_embeds = video_output.pooler_output
        video_embeds = self.video_head(video_embeds)

        text_embeds = text_output.hidden_states[-1][:, 0]
        text_embeds = self.text_video_head(text_embeds)

        logits_per_video = video_embeds @ text_embeds.T
        logits_per_video = logits_per_video * self.video_logit_scale + self.video_logit_bias
        logits_per_text = logits_per_video.t()

        loss = None
        if return_loss:
            labels = torch.eye(text_embeds.shape[0], device=text_embeds.device)
            loss = -F.logsigmoid(labels * logits_per_text).sum() / text_embeds.shape[0]

        return PEVideoOutput(
            logits_per_text=logits_per_text,
            logits_per_video=logits_per_video,
            text_embeds=text_embeds,
            video_embeds=video_embeds,
            text_model_output=text_output,
            video_model_output=video_output,
            loss=loss,
        )


__all__ = [
    "PEVideoEncoder",
    "PEVideoModel",
    "PEVideoEncoderConfig",
    "PEVideoConfig",
]
