from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from ...modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ..auto import AutoModel
from ..pe_audio.modeling_pe_audio import PEAudioResnetBlock1d, PEAudioEncoderEmbeddings
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding


class SamAudioJudgeConfig(Qwen3Config):
    model_type = "sam_audio_judge"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    _default_text_config_kwargs = {
        "model_type": "modernbert",
        "hidden_size": 1024,
        "intermediate_size": 2624,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
    }

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        hidden_size=192,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=3,
        num_key_value_heads=3,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=10000,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        rope_parameters={
            "rope_theta": 20000.0,
        },
        attention_bias=False,
        max_window_layers=28,
        attention_dropout=0.0,
        sliding_window=None,
        use_sliding_window=False,
        layer_types=None,
        tie_word_embeddings=False,
        vocab_size=None,
        bottleneck_dim=256,
        **kwargs,
    ):
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

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "modernbert")
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["modernbert"](
                **self._default_text_config_kwargs
            )

        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "pe_audio_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["pe_audio_encoder"]()

        self.text_config = text_config
        self.audio_config = audio_config

        self.bottleneck_dim = bottleneck_dim


class SamAudioJudgeEmbeddings(PEAudioEncoderEmbeddings): ...


class SamAudioJudgeAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.sliding_window = None


class SamAudioJudgeEncoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        del self.attention_type


class SamAudioJudgeRMSNorm(Qwen3RMSNorm): ...


class SamAudioJudgeRotaryEmbedding(Qwen3RotaryEmbedding): ...


def stack_freqs(cos: torch.Tensor, sin: torch.Tensor):
    dim = cos.size(-1)
    cos = cos.narrow(-1, 0, dim // 2)
    sin = sin.narrow(-1, 0, dim // 2)
    freqs_cis = torch.stack((cos, -sin, sin, cos), dim=-1).view(*cos.size(), 2, 2)
    return freqs_cis


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    freqs_cis = stack_freqs(cos, sin)
    freqs_cis = freqs_cis.unsqueeze(unsqueeze_dim)
    q_ = q.reshape(*q.shape[:-1], -1, 1, 2)
    k_ = k.reshape(*k.shape[:-1], -1, 1, 2)
    return (q_ * freqs_cis).sum(5).flatten(3), (k_ * freqs_cis).sum(5).flatten(3)


class SamAudioJudgePretrainedModel(PreTrainedModel):
    config: SamAudioJudgeConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True


@dataclass
class SamAudioJudgeOutput(BaseModelOutput):
    overall: Optional[torch.Tensor] = None
    recall: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    faithfulness: Optional[torch.Tensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] if not k.endswith("model_output") else getattr(self, k).to_tuple() for k in self.keys())


class SamAudioJudgeModel(SamAudioJudgePretrainedModel):
    def __init__(self, config: SamAudioJudgeConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.audio_encoder = AutoModel.from_config(config.audio_config)

        self.cat_audio_proj = nn.Linear(2 * config.audio_config.hidden_size, config.bottleneck_dim)
        self.text_proj_1 = nn.Linear(in_features=config.text_config.hidden_size, out_features=config.audio_config.hidden_size, bias=False)
        self.text_proj_2 = nn.Linear(in_features=config.audio_config.hidden_size, out_features=config.bottleneck_dim)
        self.audio_text_proj_1 = nn.Linear(2 * config.bottleneck_dim, config.bottleneck_dim)
        self.audio_text_proj_2 = nn.Linear(config.bottleneck_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, 4, bias=False)
        self.layer_norm = nn.LayerNorm(config.bottleneck_dim)

        # transformer
        self.embeddings = SamAudioJudgeEmbeddings(config)
        self.layers = nn.ModuleList(
            [SamAudioJudgeEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = SamAudioJudgeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SamAudioJudgeRotaryEmbedding(config=config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.register_buffer("mean", torch.zeros(4))
        self.register_buffer("std", torch.ones(4))

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SamAudioJudgeOutput:
        audio_outputs: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
            **{**kwargs, "return_dict": True},
        )

        text_outputs: MaskedLMOutput = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{**kwargs, "return_dict": True},
            output_hidden_states=True,
        )

        audio_embeds, audio_hyp_embeds = audio_outputs.last_hidden_state.chunk(2, 0)
        audio_embeds = torch.cat([audio_hyp_embeds, audio_embeds], dim=2)
        audio_embeds = self.cat_audio_proj(audio_embeds)

        text_embeds = text_outputs.hidden_states[-1][:, 0]
        text_embeds = self.text_proj_1(text_embeds)
        text_embeds = self.text_proj_2(text_embeds)
        text_embeds = self.layer_norm(text_embeds)
        text_embeds = text_embeds[:, None, :].expand_as(audio_embeds)

        audio_text_embeds = torch.cat([audio_embeds, text_embeds], dim=2)
        audio_text_embeds = self.audio_text_proj_1(audio_text_embeds)
        audio_text_embeds = self.audio_text_proj_2(audio_text_embeds)

        output_mask = audio_outputs.output_mask
        inputs_embeds, attention_mask = self.embeddings(audio_text_embeds, padding_mask=output_mask)

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

        logits = self.output_proj(hidden_states[:, 1:])
        pooled_logits = torch.masked.mean(logits, mask=output_mask, dim=1)
        de_normalized_logits = pooled_logits * self.std + self.mean

        overall, recall, precision, faithfulness = de_normalized_logits.chunk(4, dim=1)

        return SamAudioJudgeOutput(
            overall=overall,
            recall=recall,
            precision=precision,
            faithfulness=faithfulness,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
            last_hidden_state=hidden_states[:, 1:],
        )


__all__ = ["SamAudioJudgeModel", "SamAudioJudgeConfig"]
