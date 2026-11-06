# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import json
import math
import re
from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...decision import DecisionConfig, DecisionMixin, Question
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessorMixin, Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, cached_file, can_return_tuple
from ..auto import AutoConfig, AutoModel, AutoTokenizer
from ..clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
from ..lfm2.configuration_lfm2 import Lfm2Config
from ..lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2DecoderLayer,
    Lfm2Model,
    Lfm2PreTrainedModel,
    Lfm2RotaryEmbedding,
    Lfm2ShortConv,
)
from ..lfm2_vl.image_processing_lfm2_vl import Lfm2VlImageProcessor
from ..lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector
from ..parakeet.configuration_parakeet import ParakeetEncoderConfig
from ..parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from ..parakeet.modeling_parakeet import (
    ParakeetEncoder,
    ParakeetEncoderAttention,
    ParakeetEncoderRelPositionalEncoding,
    ParakeetPreTrainedModel,
)


# The publisher uses this finite sentinel, then zeros fully masked attention rows.
D1_OMNI_MASK_VALUE = -10000.0


@auto_docstring(checkpoint="LiquidAI/d1-omni-600M")
@strict
class D1OmniTextConfig(Lfm2Config):
    model_type = "d1_omni_text"
    hidden_size: int = 1024
    intermediate_size: int = 6656
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    use_cache: bool = False

    def __post_init__(self, **kwargs):
        if self.layer_types is None and self.full_attn_idxs is None and self.num_hidden_layers == 16:
            self.full_attn_idxs = [2, 5, 8, 10, 12, 14]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="LiquidAI/d1-omni-600M")
@strict
class D1OmniAudioEncoderConfig(ParakeetEncoderConfig):
    model_type = "d1_omni_audio_encoder"
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 17
    num_mel_bins: int = 128
    scale_input: bool = False
    dropout: float = 0.0
    dropout_positions: float = 0.0
    layerdrop: float = 0.0
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0


@auto_docstring(checkpoint="LiquidAI/d1-omni-600M")
@strict
class D1OmniConfig(PreTrainedConfig):
    r"""
    head_layers (`int`, *optional*, defaults to 2):
        Number of bidirectional decision-head layers.
    max_length (`int`, *optional*, defaults to 16384):
        Maximum combined media-prefix and text length.
    image_text_length (`int`, *optional*, defaults to 896):
        Maximum text length for image requests.
    audio_text_length (`int`, *optional*, defaults to 15360):
        Maximum text length for audio requests.
    audio_residual_width (`int`, *optional*, defaults to 512):
        Bottleneck width of the residual audio projector.
    temperatures (`dict`, *optional*):
        Released per-type and option-count temperatures, applied only to text-only inference.
    downsample_factor (`int`, *optional*, defaults to 2):
        Spatial downsampling factor of the vision projector.
    projector_use_layernorm (`bool`, *optional*, defaults to `False`):
        Whether the vision projector applies a layer norm before its MLP.
    """

    model_type = "d1_omni"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="d1_omni_text"),
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="siglip2_vision_model"),
        "audio_config": SubConfigSpec(config_class=AutoConfig, model_type="d1_omni_audio_encoder"),
    }
    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None
    projector_hidden_size: int = 2048
    head_layers: int = 2
    max_length: int = 16384
    image_text_length: int = 896
    audio_text_length: int = 15360
    audio_residual_width: int = 512
    temperatures: dict | None = None
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    # LFM2-VL's released projector without its optional layer norm.
    downsample_factor: int = 2
    projector_bias: bool = True
    projector_use_layernorm: bool = False
    projector_hidden_act: str = "gelu"

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = {
                "hidden_size": 1024,
                "intermediate_size": 6656,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "full_attn_idxs": [2, 5, 8, 10, 12, 14],
            }
        if self.vision_config is None:
            self.vision_config = {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "num_patches": 256,
                "patch_size": 16,
                "vision_use_head": False,
            }
        audio = dict(self.audio_config or {}) if not isinstance(self.audio_config, PreTrainedConfig) else None
        if audio is not None and (not audio or "d_model" in audio):
            self.audio_residual_width = audio.pop("residual_width", self.audio_residual_width)
            width = audio.pop("d_model", 512)
            self.audio_config = dict(
                hidden_size=width,
                num_hidden_layers=audio.pop("n_layers", 17),
                intermediate_size=width * audio.pop("ff_expansion_factor", 4),
                num_attention_heads=audio.pop("n_heads", 8),
                num_mel_bins=audio.pop("feat_in", 128),
                scale_input=False,
                dropout=0.0,
                dropout_positions=0.0,
                layerdrop=0.0,
                activation_dropout=0.0,
                attention_dropout=0.0,
                **audio,
            )
        self.temperatures = self.temperatures or {}
        super().__post_init__(**kwargs)
        self.text_config.use_cache = False
        if self.text_config.conv_L_cache != 3 or self.text_config.conv_bias:
            raise ValueError("D1 Omni requires three-tap, bias-free text convolutions.")
        if self.text_config.hidden_size % 64:
            raise ValueError("The decision-head width must be a multiple of 64.")


class D1OmniTextAttention(Lfm2Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.is_causal = False


class D1OmniTextShortConv(Lfm2ShortConv):
    def forward(self, hidden_states, attention_mask, media_lengths, **kwargs):
        gates = self.in_proj(hidden_states * attention_mask.unsqueeze(-1)).transpose(1, 2)
        input_gate, output_gate, values = gates.chunk(3, dim=1)
        neighbors = F.pad(input_gate * values, (1, 1)).unfold(-1, 3, 1)
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        # The last media position cannot read the following text position.
        keep_right = (positions != media_lengths[:, None] - 1).to(hidden_states.dtype)
        weights = self.conv.weight[:, 0]
        mixed = neighbors[..., 0] * weights[:, 0, None] + neighbors[..., 1] * weights[:, 1, None]
        mixed = mixed + neighbors[..., 2] * keep_right[:, None] * weights[:, 2, None]
        return self.out_proj((output_gate * mixed).transpose(1, 2))


class D1OmniTextDecoderLayer(Lfm2DecoderLayer):
    def forward(self, hidden_states, position_embeddings, attention_mask, valid_tokens, media_lengths, **kwargs):
        normalized = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            mixed, _ = self.self_attn(normalized, position_embeddings, attention_mask, **kwargs)
        else:
            mixed = self.conv(normalized, valid_tokens, media_lengths)
        hidden_states = hidden_states + mixed
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class D1OmniTextPreTrainedModel(Lfm2PreTrainedModel):
    config_class = D1OmniTextConfig


class D1OmniTextRotaryEmbedding(Lfm2RotaryEmbedding):
    @torch.no_grad()
    def forward(self, x, position_ids):
        width = self.config.hidden_size // self.config.num_attention_heads
        exponent = torch.arange(0, width, 2, device="cpu").float() / width
        theta = self.config.rope_parameters["rope_theta"]
        inverse = (1.0 / theta**exponent).to(x.device)
        with torch.autocast(device_type=x.device.type, enabled=False):
            frequencies = (inverse[None, :, None] @ position_ids[:, None].float()).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(x.dtype), embedding.sin().to(x.dtype)


class D1OmniTextModel(Lfm2Model):
    def __init__(self, config: D1OmniTextConfig):
        super().__init__(config)
        self.embed_tokens.padding_idx = None

    def forward(self, input_ids=None, attention_mask=None, media_lengths=None, inputs_embeds=None, **kwargs):
        r"""
        media_lengths (`torch.LongTensor` of shape `(batch_size,)`):
            Prefix length per row; media positions cannot attend to text.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        if media_lengths is None:
            media_lengths = torch.zeros(inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device)
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).expand(inputs_embeds.shape[:2])
        media_queries = positions < media_lengths[:, None]
        text_keys = positions >= media_lengths[:, None]
        forbidden = ~attention_mask[:, None, None, :].bool()
        forbidden = forbidden | (media_queries[:, None, :, None] & text_keys[:, None, None, :])
        mask = inputs_embeds.new_zeros((inputs_embeds.shape[0], 1, inputs_embeds.shape[1], inputs_embeds.shape[1]))
        mask = mask.masked_fill(forbidden, max(-1e9, torch.finfo(inputs_embeds.dtype).min))
        rotary = self.rotary_emb(inputs_embeds, positions)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary, mask, attention_mask, media_lengths, **kwargs)
        return BaseModelOutput(last_hidden_state=self.embedding_norm(hidden_states))


class D1OmniHeadAttention(CLIPAttention):
    def __init__(self, config: PreTrainedConfig):
        super().__init__(config)


class D1OmniHeadLayer(CLIPEncoderLayer):
    def __init__(self, config: PreTrainedConfig):
        super().__init__(config)


class D1OmniScorer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layer_norm = nn.LayerNorm(width)
        self.dense = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, 1)

    def forward(self, hidden_states):
        return self.out_proj(F.gelu(self.dense(self.layer_norm(hidden_states))))


class D1OmniDecisionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.text_config.hidden_size
        head_config = PreTrainedConfig(
            attention_dropout=0.0,
            hidden_size=width,
            intermediate_size=4 * width,
            num_attention_heads=width // 64,
            hidden_act="relu",
            layer_norm_eps=1e-5,
        )
        head_config._attn_implementation = config._attn_implementation
        self.type_emb = nn.Embedding(3, width)
        self.layers = nn.ModuleList(D1OmniHeadLayer(head_config) for _ in range(config.head_layers))
        self.scorer = D1OmniScorer(width)

    def forward(self, hidden_states, attention_mask, marker_positions, marker_mask, question_type_ids):
        hidden_states = hidden_states + self.type_emb(question_type_ids)[:, None]
        mask = hidden_states.new_zeros((hidden_states.shape[0], 1, 1, hidden_states.shape[1]))
        mask = mask.masked_fill(~attention_mask[:, None, None, :].bool(), float("-inf"))
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        selected = hidden_states.gather(1, marker_positions[:, :, None].expand(-1, -1, hidden_states.shape[-1]))
        return self.scorer(selected).squeeze(-1).float().masked_fill(~marker_mask.bool(), D1_OMNI_MASK_VALUE)


class D1OmniVisionProjector(Lfm2VlMultiModalProjector):
    def __init__(self, config: D1OmniConfig):
        super().__init__(config)


class D1OmniVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tower = AutoModel.from_config(config.vision_config)
        self.projector = D1OmniVisionProjector(config)

    def forward(self, pixel_values, spatial_shapes, pixel_attention_mask):
        hidden = self.tower(
            pixel_values=pixel_values, spatial_shapes=spatial_shapes, pixel_attention_mask=pixel_attention_mask
        ).last_hidden_state
        return [
            self.projector(row[: h * w].reshape(1, h, w, -1)).flatten(1, 2)
            for row, (h, w) in zip(hidden, spatial_shapes)
        ]


class D1OmniAudioEncoderRelPositionalEncoding(ParakeetEncoderRelPositionalEncoding):
    def forward(self, hidden_states):
        # The publisher builds frequencies and sinusoids on CPU in float32.
        width = self.config.hidden_size
        length = hidden_states.shape[1]
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device="cpu")[:, None]
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32, device="cpu") * -(math.log(10000.0) / width)
        )
        values = positions * frequencies
        return (
            torch.stack((values.sin(), values.cos()), dim=-1)
            .flatten(-2)[None]
            .expand(hidden_states.shape[0], -1, -1)
            .to(hidden_states)
        )


class D1OmniAudioEncoderAttention(ParakeetEncoderAttention):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        batch, length, width = hidden_states.shape
        shape = (batch, length, self.config.num_attention_heads, self.head_dim)
        # Match the released relative-position attention's operation order and masked rows.
        query = self.q_proj(hidden_states).view(shape)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        relative = (
            self.relative_k_proj(position_embeddings)
            .view(batch, -1, self.config.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        content = (query + self.bias_u).transpose(1, 2) @ key.transpose(-2, -1)
        position = (query + self.bias_v).transpose(1, 2) @ relative.transpose(-2, -1)
        scores = (content + self._rel_shift(position)[..., :length]) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask, D1_OMNI_MASK_VALUE)
        probabilities = scores.softmax(-1).masked_fill(~attention_mask, 0.0)
        return self.o_proj((probabilities @ value).transpose(1, 2).reshape(batch, length, width)), None


class D1OmniAudioPreTrainedModel(ParakeetPreTrainedModel):
    config: D1OmniAudioEncoderConfig


class D1OmniAudioEncoder(ParakeetEncoder):
    pass


class D1OmniAudioAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.text_config.hidden_size
        self.norm = nn.LayerNorm(config.audio_config.hidden_size)
        self.linear_1 = nn.Linear(config.audio_config.hidden_size, width)
        self.linear_2 = nn.Linear(width, width)

    def forward(self, hidden_states):
        return self.linear_2(F.gelu(self.linear_1(self.norm(hidden_states))))


class D1OmniAudioResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.text_config.hidden_size
        self.ln = nn.LayerNorm(width)
        self.down = nn.Linear(width, config.audio_residual_width)
        self.up = nn.Linear(config.audio_residual_width, width)

    def forward(self, hidden_states):
        return hidden_states + self.up(F.gelu(self.down(self.ln(hidden_states))))


class D1OmniAudio(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = D1OmniAudioEncoder(config.audio_config)
        self.adapter = D1OmniAudioAdapter(config)
        self.residual = D1OmniAudioResidual(config)

    def forward(self, input_features, feature_attention_mask):
        outputs = self.encoder(input_features, attention_mask=feature_attention_mask, return_dict=True)
        hidden = self.residual(self.adapter(outputs.last_hidden_state))
        return [row[mask.bool()][None] for row, mask in zip(hidden, outputs.attention_mask)]


@auto_docstring
@dataclass
class D1OmniModelOutput(ModelOutput):
    r"""
    attention_mask (`torch.Tensor` of shape `(batch_size, text_length)`):
        Valid text positions after removing media-prefix states.
    """

    last_hidden_state: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


@auto_docstring
@dataclass
class D1OmniDecisionOutput(ModelOutput):
    r"""
    loss (`torch.Tensor` of shape `()`, *optional*):
        Mean cross entropy over labeled questions, when labels are supplied.
    logits (`torch.Tensor` of shape `(batch_size, num_options)`):
        Uncalibrated option scores in the publisher's internal order; Noul uses false, true.
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


@auto_docstring
class D1OmniPreTrainedModel(PreTrainedModel):
    config: D1OmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["D1OmniTextDecoderLayer", "D1OmniHeadLayer", "D1OmniAudioEncoderBlock"]


@auto_docstring
class D1OmniModel(D1OmniPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = D1OmniTextModel(config.text_config)
        self.vision = D1OmniVision(config)
        self.audio = D1OmniAudio(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        media_prefix: torch.Tensor | None = None,
        media_lengths: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        spatial_shapes: torch.LongTensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        image_counts: torch.LongTensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> D1OmniModelOutput:
        r"""
        media_prefix (`torch.Tensor` of shape `(batch_size, prefix_length, hidden_size)`, *optional*):
            Precomputed media embeddings, as an alternative to images or audio features.
        media_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of real media embeddings in each supplied prefix.
        spatial_shapes (`torch.LongTensor` of shape `(num_crops, 2)`, *optional*):
            Patch-grid height and width of each image crop.
        pixel_attention_mask (`torch.Tensor` of shape `(num_crops, num_patches)`, *optional*):
            Valid image patches.
        image_counts (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of image crops belonging to each question.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Valid audio feature frames before subsampling.
        """
        attention_mask = (
            torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        )
        if torch.any(attention_mask.sum(-1) == 0):
            raise ValueError("Each question requires at least one text token.")
        if not torch.equal(
            attention_mask,
            torch.arange(input_ids.shape[1], device=input_ids.device)[None] < attention_mask.sum(-1)[:, None],
        ):
            raise ValueError("D1 Omni requires right-padded text.")
        if sum(value is not None for value in (media_prefix, pixel_values, input_features)) > 1:
            raise ValueError("Supply only one of images, audio features, or precomputed media prefixes.")
        if pixel_values is not None:
            crops = self.vision(pixel_values, spatial_shapes, pixel_attention_mask)
            if image_counts is None or len(image_counts) != len(input_ids) or image_counts.sum() != len(crops):
                raise ValueError("image_counts must give the number of image crops for each question.")
            groups = torch.tensor_split(torch.arange(len(crops)), image_counts.cumsum(0)[:-1].cpu())
            prefixes = [torch.cat([crops[i] for i in group], dim=1)[0] for group in groups]
            media_lengths = input_ids.new_tensor([len(prefix) for prefix in prefixes])
            media_prefix = nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
        elif input_features is not None:
            if feature_attention_mask is None:
                raise ValueError("Audio requires feature_attention_mask.")
            prefixes = self.audio(input_features, feature_attention_mask)
            media_lengths = input_ids.new_tensor([prefix.shape[1] for prefix in prefixes])
            media_prefix = nn.utils.rnn.pad_sequence([prefix[0] for prefix in prefixes], batch_first=True)
        embeddings = self.encoder.embed_tokens(input_ids)
        lengths = attention_mask.sum(-1)
        if media_prefix is None:
            media_lengths = torch.zeros_like(lengths)
            sequences = [row[:length] for row, length in zip(embeddings, lengths)]
        else:
            if media_lengths is None or media_lengths.shape != lengths.shape:
                raise ValueError("media_prefix requires one media length per question.")
            if (
                media_prefix.shape[0] != len(input_ids)
                or torch.any(media_lengths < 0)
                or torch.any(media_lengths > media_prefix.shape[1])
            ):
                raise ValueError("media_lengths must index the supplied media prefixes.")
            sequences = [
                torch.cat((prefix[:offset].to(row), row[:length]))
                for prefix, offset, row, length in zip(media_prefix, media_lengths, embeddings, lengths)
            ]
        if torch.any(lengths + media_lengths > self.config.max_length):
            raise ValueError("Combined media and text exceed max_length.")
        packed = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        valid = torch.arange(packed.shape[1], device=input_ids.device)[None] < (lengths + media_lengths)[:, None]
        hidden = self.encoder(
            inputs_embeds=packed, attention_mask=valid, media_lengths=media_lengths, **kwargs
        ).last_hidden_state
        text = nn.utils.rnn.pad_sequence(
            [row[offset : offset + length] for row, offset, length in zip(hidden, media_lengths, lengths)],
            batch_first=True,
        )
        text_mask = torch.arange(text.shape[1], device=input_ids.device)[None] < lengths[:, None]
        return D1OmniModelOutput(last_hidden_state=text, attention_mask=text_mask)


@auto_docstring
class D1OmniForDecision(DecisionMixin, D1OmniModel):
    # The released model has no training objective. This is AutoModel's plain-CE adaptation.
    accepts_loss_kwargs = True

    def __init__(self, config):
        super().__init__(config)
        self.head = D1OmniDecisionHead(config)
        self.post_init()

    @property
    def decision_config(self):
        if getattr(self, "_decision_config", None) is None:
            self._decision_config = DecisionConfig.from_pretrained(self.name_or_path)
            self._decision_config.chat_template = "{{ state }}{{ question }}"
            self._decision_config.labels = dict.fromkeys(Question.questions_types)
            self._decision_config.noul_options = ["false", "true"]
        return self._decision_config

    @torch.no_grad()
    def decide(self, input_ids, decision_questions, decision_config=None, **kwargs):
        """Return option scores in the decision API's question order."""
        if any(question.allow_null for question in decision_questions):
            raise ValueError("D1 does not support null answers.")
        has_media = any(kwargs.get(key) is not None for key in ("pixel_values", "input_features", "media_prefix"))
        output = self(input_ids=input_ids, return_dict=True, **kwargs)
        scores = []
        for question, row in zip(decision_questions, output.logits):
            row = row[: len(question.options)]
            if not has_media:
                count = len(question.options)
                bucket = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
                row = row / self.config.temperatures.get(
                    f"{question.type}:{bucket}", self.config.temperatures.get(question.type, 1.0)
                )
            scores.append(row)
        return scores

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        marker_positions: torch.LongTensor,
        marker_mask: torch.Tensor,
        question_type_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        media_prefix: torch.Tensor | None = None,
        media_lengths: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        spatial_shapes: torch.LongTensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        image_counts: torch.LongTensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> D1OmniDecisionOutput:
        r"""
        marker_positions (`torch.LongTensor` of shape `(batch_size, num_options)`):
            Option-marker positions relative to each text sequence.
        marker_mask (`torch.Tensor` of shape `(batch_size, num_options)`):
            Valid options; padded options are excluded from the loss.
        question_type_ids (`torch.LongTensor` of shape `(batch_size,)`):
            Publisher question types: choice=0, score=1, noul=2.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Correct option indices in internal order, with -100 for ignored questions.
        num_items_in_batch (`int` or `torch.Tensor`, *optional*):
            Labeled-question count across accumulated microbatches.

        media_prefix (`torch.Tensor` of shape `(batch_size, prefix_length, hidden_size)`, *optional*):
            Precomputed media embeddings, as an alternative to images or audio features.
        media_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of real media embeddings in each supplied prefix.
        spatial_shapes (`torch.LongTensor` of shape `(num_crops, 2)`, *optional*):
            Patch-grid height and width of each image crop.
        pixel_attention_mask (`torch.Tensor` of shape `(num_crops, num_patches)`, *optional*):
            Valid image patches.
        image_counts (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of image crops belonging to each question.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Valid audio feature frames before subsampling.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            media_prefix=media_prefix,
            media_lengths=media_lengths,
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            image_counts=image_counts,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            return_dict=True,
            **kwargs,
        )
        if torch.any(~marker_mask.bool().any(-1)):
            raise ValueError("Each question requires at least one option.")
        if torch.any(marker_positions[marker_mask.bool()] < 0) or torch.any(
            marker_positions[marker_mask.bool()] >= outputs.last_hidden_state.shape[1]
        ):
            raise ValueError("Option markers must index text tokens.")
        positions = marker_positions.masked_fill(~marker_mask.bool(), 0)
        if torch.any(~outputs.attention_mask.gather(1, positions)[marker_mask.bool()]):
            raise ValueError("Option markers cannot index padded text.")
        logits = self.head(
            outputs.last_hidden_state, outputs.attention_mask, positions, marker_mask, question_type_ids
        )
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, option_mask=marker_mask, num_items_in_batch=num_items_in_batch
            )
        return D1OmniDecisionOutput(loss=loss, logits=logits)


@auto_docstring
class D1OmniImageProcessor(Lfm2VlImageProcessor):
    pass


class D1OmniFeatureExtractor(ParakeetFeatureExtractor):
    r"""Construct the released D1 log-mel frontend.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            Number of mel-frequency features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Input audio sampling rate in Hz.
        hop_length (`int`, *optional*, defaults to 160):
            STFT hop length in samples.
        n_fft (`int`, *optional*, defaults to 512):
            Fourier transform size.
        win_length (`int`, *optional*, defaults to 400):
            Hann window length in samples.
        preemphasis (`float`, *optional*, defaults to 0.97):
            Preemphasis coefficient.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to pad audio samples.
    """

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        preemphasis=0.97,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            preemphasis=preemphasis,
            padding_value=padding_value,
            **kwargs,
        )


@auto_docstring
class D1OmniProcessor(ProcessorMixin):
    """Prepare D1's per-question text, tiled images, or mono 16 kHz audio.

    The prompt's option budget, reserved-token escaping, and modality-specific
    option descriptions follow the released publisher implementation.

    Args:
        image_text_length (`int`, *optional*, defaults to 896):
            Maximum number of text tokens in image requests.
        audio_text_length (`int`, *optional*, defaults to 15360):
            Maximum number of text tokens in audio requests.
    """

    def __init__(
        self,
        image_processor,
        feature_extractor,
        tokenizer,
        chat_template=None,
        max_length=16384,
        image_text_length=896,
        audio_text_length=15360,
        **kwargs,
    ):
        r"""
        image_text_length (`int`, *optional*, defaults to 896):
            Maximum number of text tokens in image requests.
        audio_text_length (`int`, *optional*, defaults to 15360):
            Maximum number of text tokens in audio requests.
        """
        self.max_length = max_length
        self.image_text_length = image_text_length
        self.audio_text_length = audio_text_length
        super().__init__(
            image_processor,
            feature_extractor,
            tokenizer,
            chat_template=chat_template or "{{ state }}{{ question }}",
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # The publisher checkpoint predates the native processor config.
        hub_kwargs = {
            key: kwargs[key] for key in ("revision", "cache_dir", "token", "local_files_only") if key in kwargs
        }
        path = cached_file(
            pretrained_model_name_or_path,
            "processor_config.json",
            _raise_exceptions_for_missing_entries=False,
            **hub_kwargs,
        )
        if path is not None:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config = D1OmniConfig.from_pretrained(pretrained_model_name_or_path, **hub_kwargs)
        return cls(
            D1OmniImageProcessor(),
            D1OmniFeatureExtractor(feature_size=config.audio_config.num_mel_bins),
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **hub_kwargs),
            max_length=config.max_length,
            image_text_length=config.image_text_length,
            audio_text_length=config.audio_text_length,
        )

    def _encode_question(self, state, question, max_length, has_images=False, has_audio=False):
        tokenizer = self.tokenizer
        kind = question["type"]
        if kind not in ("choice", "score", "noul"):
            raise ValueError("A D1 question must have type choice, score, or noul.")
        if question.get("allow_null", False):
            raise ValueError("D1 does not support null answers.")
        criteria = question.get("criteria")
        if "options" in question:
            criteria = (
                dict(zip(question["options"], question["descriptions"]))
                if kind != "score"
                else question["descriptions"]
            )

        def text(value):
            return (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)
            )

        if kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError("A choice requires at least two named options.")
            options = [
                f"option_{i:03d}: {text(key if value is None or value == '' else value)}"
                if has_audio
                else key
                if value is None or value == ""
                else f"{key}: {text(value)}"
                for i, (key, value) in enumerate(criteria.items())
            ]
        elif kind == "score":
            if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10:
                raise ValueError("A score requires two to ten ordered levels.")
            options = [f"level {i}: {text(value)}" for i, value in enumerate(criteria)]
        else:
            criteria = criteria or ({"false": "no", "true": "yes"} if has_images else {})
            false = criteria.get("false", criteria.get("no"))
            true = criteria.get("true", criteria.get("yes"))
            options = (
                ["false: no", "true: yes"]
                if has_audio
                else [
                    "false: " + (text(false) if false not in (None, "") else "no, the statement does not hold"),
                    "true: " + (text(true) if true not in (None, "") else "yes, the statement holds"),
                ]
            )

        def encode(value):
            escaped = re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", value)
            return tokenizer(escaped, add_special_tokens=False)["input_ids"]

        token_id = tokenizer.convert_tokens_to_ids
        budget = max(96, min(len(options) * 24 + 32, max_length // 2))
        per_option = max(2, (budget - 3 * len(options)) // len(options))
        instructions = question.get("instructions", "")
        question_ids = ([token_id("<|reserved_8|>")] + encode(str(instructions)))[: max(16, budget)]
        markers = []
        for option in options:
            markers.append(len(question_ids) + 1)
            question_ids += (
                [token_id("<|reserved_9|>"), token_id("<|mask|>")]
                + encode(" " + option)[:per_option]
                + [token_id("<|reserved_10|>")]
            )
        question_ids.append(token_id("<|reserved_11|>"))
        room = max(0, max_length - len(question_ids) - 2)
        state_ids = [token_id("<|reserved_7|>")] + encode(text(state))[:room]
        ids = ([tokenizer.bos_token_id] + state_ids + question_ids)[:max_length]
        markers = [position + 1 + len(state_ids) for position in markers]
        if markers[-1] >= max_length:
            raise ValueError("The options do not fit in the context.")
        return ids, markers, {"choice": 0, "score": 1, "noul": 2}[kind]

    @auto_docstring
    def __call__(
        self,
        text=None,
        images=None,
        audio=None,
        questions=None,
        state=None,
        sampling_rate=16000,
        return_tensors="pt",
        **kwargs,
    ):
        r"""Encode questions over one shared state; each question is an independent batch row.

        Args:
            questions (`list[dict]`):
                Typed question dictionaries, each containing a type, instructions, and optional criteria.
            state (`str` or JSON value, *optional*):
                Shared context for the questions. Defaults to `text` when omitted.
            sampling_rate (`int`, *optional*, defaults to 16000):
                Sampling rate of the supplied mono audio. Must be 16000.
        """
        if return_tensors != "pt":
            raise ValueError("D1OmniProcessor currently returns PyTorch tensors.")
        if images is not None and audio is not None:
            raise ValueError("A request can contain images or audio, but not both.")
        if not questions:
            raise ValueError("Supply a nonempty list of questions.")
        state = text if state is None else state
        state = {} if state is None and audio is not None else "" if state is None else state
        count = len(questions)
        inputs = {}
        prefix_length = 0
        if images is not None:
            pixels = self.image_processor(images, return_tensors="pt")
            inputs = {
                key: pixels[key].repeat(count, *([1] * (pixels[key].ndim - 1)))
                for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes")
            }
            inputs["image_counts"] = torch.full((count,), pixels["pixel_values"].shape[0], dtype=torch.long)
            prefix_length = int(pixels["spatial_shapes"].prod(-1).sum() // 4)
        elif audio is not None:
            if sampling_rate != 16000:
                raise ValueError("D1 audio must be sampled at 16000 Hz.")
            samples = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
            if samples.ndim != 1:
                raise ValueError("D1 expects one mono audio clip.")
            samples = samples[:480000]
            samples = samples.astype(np.float32) / 32768.0 if samples.dtype == np.int16 else samples.astype(np.float32)
            samples = np.pad(samples, (0, max(0, 8000 - len(samples))))
            features = self.feature_extractor(samples, sampling_rate=16000, return_tensors="pt")
            inputs["input_features"] = features["input_features"].repeat(count, 1, 1)
            inputs["feature_attention_mask"] = features["attention_mask"].repeat(count, 1)
            prefix_length = int((features["attention_mask"].sum() + 7) // 8)
        limit = (
            self.image_text_length
            if images is not None
            else self.audio_text_length
            if audio is not None
            else self.max_length
        )
        limit = min(limit, self.max_length - prefix_length)
        if limit < 64:
            raise ValueError("The media leave fewer than 64 text positions.")
        encoded = [self._encode_question(state, q, limit, images is not None, audio is not None) for q in questions]
        ids = [torch.tensor(row[0], dtype=torch.long) for row in encoded]
        markers = [torch.tensor(row[1], dtype=torch.long) for row in encoded]
        inputs["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            ids, batch_first=True, padding_value=self.tokenizer.pad_token_id or 0
        )
        inputs["attention_mask"] = (
            torch.arange(inputs["input_ids"].shape[1])[None] < torch.tensor([len(row) for row in ids])[:, None]
        )
        inputs["marker_positions"] = torch.nn.utils.rnn.pad_sequence(markers, batch_first=True)
        inputs["marker_mask"] = (
            torch.arange(inputs["marker_positions"].shape[1])[None]
            < torch.tensor([len(row) for row in markers])[:, None]
        )
        inputs["question_type_ids"] = torch.tensor([row[2] for row in encoded])
        return BatchFeature(inputs)

    def apply_chat_template(
        self,
        conversation=None,
        *,
        state=None,
        question=None,
        tokenize=False,
        return_dict=False,
        processor_kwargs=None,
        **kwargs,
    ):
        """Prepare the publisher prompt used by the decision pipeline."""
        if question is None:
            return super().apply_chat_template(conversation, tokenize=tokenize, return_dict=return_dict, **kwargs)
        inputs = self(state=state, questions=[question], **(processor_kwargs or {}))
        if not tokenize:
            return self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        return inputs if return_dict else inputs["input_ids"]


__all__ = [
    "D1OmniProcessor",
    "D1OmniImageProcessor",
    "D1OmniFeatureExtractor",
    "D1OmniConfig",
    "D1OmniTextConfig",
    "D1OmniAudioEncoderConfig",
    "D1OmniModel",
    "D1OmniTextModel",
    "D1OmniAudioEncoder",
    "D1OmniPreTrainedModel",
    "D1OmniTextPreTrainedModel",
    "D1OmniAudioPreTrainedModel",
    "D1OmniForDecision",
]
