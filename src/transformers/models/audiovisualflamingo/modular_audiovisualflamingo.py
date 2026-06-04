# Copyright 2026 The HuggingFace Team and NVIDIA CORPORATION. All rights reserved.
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

import copy
import math
import warnings
from collections import defaultdict, deque
from math import pi
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import broadcast_tensors, einsum

from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast
from ...utils import ModelOutput
from ..audioflamingo3.modeling_audioflamingo3 import AudioFlamingo3MultiModalProjector
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel, AutoModelForCausalLM
from ..llava_next.modeling_llava_next import LlavaNextMultiModalProjector
from ..perceiver.modeling_perceiver import space_to_depth
from ..voxtral.modeling_voxtral import VoxtralPreTrainedModel


IGNORE_INDEX = -100

MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
    "sound": "<sound>",
}

MM_BOS_EOS_TOKENS = {
    "image": ["<|image_bos|>", "<|image_eos|>"],
    "video": ["<|video_bos|>", "<|video_eos|>"],
    "sound": ["<|sound_bos|>", "<|sound_eos|>"],
}


@strict
class AudioVisualFlamingoConfig(PreTrainedConfig):
    model_type = "audiovisualflamingo"
    keys_to_ignore_at_inference = ["past_key_values"]
    media_tokens = MEDIA_TOKENS
    mm_bos_eos_tokens = MM_BOS_EOS_TOKENS
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": AutoConfig,
        "audio_config": AutoConfig,
    }

    @staticmethod
    def _build_sub_config(config, default_model_type: str):
        if isinstance(config, PreTrainedConfig):
            return copy.deepcopy(config)
        if config is None:
            return CONFIG_MAPPING[default_model_type]()
        if isinstance(config, dict):
            model_type = config.get("model_type", default_model_type)
            config_kwargs = {k: v for k, v in config.items() if k != "model_type"}
            return CONFIG_MAPPING[model_type](**config_kwargs)
        raise TypeError(f"Unsupported config payload type: {type(config)!r}")

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        dynamic_s2=None,
        s2_scales=None,
        s2_max_split_size=None,
        s2_resize_output_to_scale_idx=0,
        image_encoder=None,
        video_encoder=None,
        sound_encoder=None,
        projector_bias=True,
        multimodal_projector_bias=True,
        load_audio_in_video=True,
        interleaved_vis_aud_in_video=True,
        **kwargs,
    ):
        legacy_config_aliases = {
            "llm_cfg": "text_config",
            "vision_tower_cfg": "vision_config",
            "sound_tower_cfg": "audio_config",
        }
        used_legacy_aliases = [key for key in legacy_config_aliases if key in kwargs]
        if used_legacy_aliases:
            formatted_aliases = ", ".join(
                f"`{key}` -> `{legacy_config_aliases[key]}`" for key in sorted(used_legacy_aliases)
            )
            raise TypeError(
                "AudioVisualFlamingoConfig only accepts canonical sub-config names. "
                f"Replace legacy aliases: {formatted_aliases}."
            )

        self.text_config = self._build_sub_config(text_config, "qwen2")
        self.vision_config = self._build_sub_config(vision_config, "siglip_vision_model")
        self.audio_config = self._build_sub_config(audio_config, "qwen2_audio_encoder")

        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.dynamic_s2 = dynamic_s2
        self.s2_scales = list(s2_scales) if s2_scales is not None else None
        self.s2_max_split_size = s2_max_split_size
        self.s2_resize_output_to_scale_idx = s2_resize_output_to_scale_idx

        self.image_encoder = copy.deepcopy(image_encoder or {"_target_": "BasicImageEncoder"})
        self.video_encoder = copy.deepcopy(video_encoder or {"_target_": "TSPVideoEncoder"})
        self.sound_encoder = copy.deepcopy(sound_encoder or {"_target_": "BasicSoundEncoder"})
        self.load_audio_in_video = load_audio_in_video
        self.interleaved_vis_aud_in_video = interleaved_vis_aud_in_video

        self.projector_bias = projector_bias
        self.multimodal_projector_bias = multimodal_projector_bias

        super().__init__(**kwargs)


def pool(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    if x.shape[dim] % size != 0:
        remainder = x.shape[dim] % size
        pad_len = size - remainder
        last_elements = x.narrow(dim, x.shape[dim] - remainder, remainder)
        mean_value = last_elements.mean()
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        padding = torch.ones(pad_shape, device=x.device, dtype=x.dtype) * mean_value
        x = torch.cat([x, padding], dim=dim)

    shape_before = x.shape[:dim]
    shape_after = x.shape[dim + 1 :]
    new_shape = shape_before + (-1, size) + shape_after
    return x.view(new_shape).mean(dim + 1)


def _tokens_to_channel_first(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected tensor of shape (batch, tokens, channels), got {tuple(x.shape)}")
    batch_size, num_tokens, channels = x.shape
    if num_tokens != height * width:
        raise ValueError(f"Token count {num_tokens} does not match spatial shape ({height}, {width})")
    return x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()


def _channel_first_to_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected tensor of shape (batch, channels, height, width), got {tuple(x.shape)}")
    batch_size, channels, height, width = x.shape
    return x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels).contiguous()


def _rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    device_type = t.device.type if t.device.type in {"cpu", "cuda"} else "cuda"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        original_dtype = t.dtype
        t = t.to(torch.float64)
        freqs = freqs.to(t)

        if t.ndim == 3:
            seq_len = t.shape[seq_dim]
            freqs = freqs[-seq_len:]

        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], (
            f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )

        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]
        t_middle = (t_middle * freqs.cos() * scale) + (_rotate_half(t_middle) * freqs.sin() * scale)
        out = torch.cat((t_left, t_middle, t_right), dim=-1)
    return out.to(original_dtype)


class MaxTimeContinuousTimeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_time, period_mode="longest"):
        super().__init__()
        if period_mode not in {"longest", "shortest"}:
            raise ValueError(f"period_mode should be 'longest' or 'shortest', got {period_mode!r}")
        self.period_mode = period_mode
        self.max_time = max_time

        if dim % 4 != 0:
            raise ValueError(f"MTCT rotary embedding requires `dim` divisible by 4, got {dim}")
        self.dim = dim
        bands = torch.arange(1, dim // 4 + 1, dtype=torch.float32)
        self.register_buffer("bands", bands, persistent=False)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        if times.ndim == 1:
            times = times.unsqueeze(0)

        times = times.float()
        batch_size, seq_len = times.shape
        times = times.clamp_min(0.0)
        max_time = times.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        if self.max_time is not None:
            max_time = max_time.clamp_max(float(self.max_time))

        if self.period_mode == "longest":
            denominator = max_time
        else:
            nonzero = times.masked_fill(times <= 0, float("inf")).min(dim=-1, keepdim=True).values
            nonzero = torch.where(torch.isfinite(nonzero), nonzero, max_time)
            denominator = nonzero.clamp_min(1e-6)

        angles = times.unsqueeze(-1) / denominator.unsqueeze(-1) * (2 * pi * self.bands)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.reshape(batch_size, seq_len, self.dim // 2)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        max_time=None,
    ):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.learned_freq = learned_freq
        self.max_time = max_time
        if max_time is not None and freqs_for == "lang":
            theta = max_time / (2 * pi)
        self.theta = theta

        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.register_buffer("cached_freqs", None, persistent=False)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def forward(self, t: torch.Tensor, seq_len=None, offset=0):
        should_cache = not self.learned_freq and seq_len is not None and self.freqs_for != "pixel"
        if should_cache and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs
        if self.max_time is not None:
            t = t / self.max_time * (2 * pi)

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        if should_cache:
            self.cached_freqs = freqs.detach()
        return freqs

    def get_axial_freqs(self, *dims):
        colon = slice(None)
        all_freqs = []
        dtype = self.freqs.dtype if torch.is_floating_point(self.freqs) else torch.float32
        for index, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = torch.linspace(-1, 1, steps=dim, device=self.device, dtype=dtype)
            else:
                pos = torch.arange(dim, device=self.device, dtype=dtype)

            freqs = self.forward(pos, seq_len=dim)
            all_axis = [None] * len(dims)
            all_axis[index] = colon
            all_freqs.append(freqs[(Ellipsis, *all_axis, colon)])

        return torch.cat(broadcast_tensors(*all_freqs), dim=-1)


def _move_rotary_module_to_device(module: nn.Module, device: torch.device) -> nn.Module:
    module_device = None
    on_meta = False
    for param in module.parameters(recurse=False):
        module_device = param.device
        on_meta = param.is_meta
        break
    if module_device is None:
        for buffer in module.buffers(recurse=False):
            module_device = buffer.device
            on_meta = buffer.is_meta
            break
    if module_device == device and not on_meta:
        return module
    if on_meta:
        if isinstance(module, RotaryEmbedding):
            return RotaryEmbedding(
                dim=module.dim,
                freqs_for=module.freqs_for,
                theta=module.theta,
                max_freq=module.max_freq,
                num_freqs=module.num_freqs,
                learned_freq=module.learned_freq,
                max_time=module.max_time,
            ).to(device=device)
        if isinstance(module, MaxTimeContinuousTimeRotaryEmbedding):
            return MaxTimeContinuousTimeRotaryEmbedding(
                dim=module.dim,
                max_time=module.max_time,
                period_mode=module.period_mode,
            ).to(device=device)
        return module.to_empty(device=device)
    return module.to(device=device)


class MultimodalProjector(LlavaNextMultiModalProjector):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int, bias: bool):
        nn.Module.__init__(self)
        self.downsample_rate = 2
        self.layers = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(vision_hidden_size * 4),
            nn.Linear(vision_hidden_size * 4, text_hidden_size, bias=bias),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size, bias=bias),
        )

    def forward(self, x, *args, **kwargs):
        _ = (args, kwargs)
        bsz, num_tokens, channels = x.shape
        h = w = int(num_tokens**0.5)
        x = x.reshape(bsz, h, w, channels).permute(0, 3, 1, 2).contiguous()
        if h % self.downsample_rate != 0 or w % self.downsample_rate != 0:
            x = F.pad(
                x,
                (0, w % self.downsample_rate, 0, h % self.downsample_rate),
                mode="constant",
                value=0,
            )
        x = space_to_depth(x, spatial_block_size=self.downsample_rate).reshape(bsz, -1, channels * 4)
        return self.layers(x)


class SoundMultimodalProjector(AudioFlamingo3MultiModalProjector):
    def __init__(self, audio_hidden_size: int, text_hidden_size: int, bias: bool):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(audio_hidden_size, text_hidden_size, bias=bias),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size, bias=bias),
        )

    def forward(self, x, *args, **kwargs):
        _ = (args, kwargs)
        return self.layers(x)


class SiglipVisionTowerDynamicS2(nn.Module):
    def __init__(self, config: AudioVisualFlamingoConfig) -> None:
        super().__init__()
        self.select_layer = config.mm_vision_select_layer
        self.select_feature = config.mm_vision_select_feature
        if config.s2_scales is None:
            raise ValueError("`config.s2_scales` must be provided when `dynamic_s2=True`.")
        self.scales = sorted(int(scale) for scale in config.s2_scales)
        self.max_split_size = config.s2_max_split_size
        self.resize_output_to_scale_idx = config.s2_resize_output_to_scale_idx

        vision_cfg = copy.deepcopy(config.vision_config)
        vision_cfg._attn_implementation = config._attn_implementation
        self.vision_tower = AutoModel.from_config(vision_cfg)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature != "cls_patch":
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        if isinstance(images, list):
            raise ValueError("VisionTowerDynamicS2 expects tensor input, not list.")
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        return self.feature_select(image_forward_outs).to(images.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.scales)


class AudioVisualFlamingoPretrainedModel(VoxtralPreTrainedModel):
    config_class = AudioVisualFlamingoConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["Qwen2DecoderLayer", "SiglipEncoderLayer"]

    @property
    def llm_model_embed_tokens(self):
        if self.llm is None:
            raise RuntimeError("LLM module is not initialized.")
        return self.llm.model.embed_tokens

    def _require_encoder_text_token_ids(self) -> dict[str, list[int]]:
        encoder_text_token_ids = getattr(self.config, "encoder_text_token_ids", None)
        if encoder_text_token_ids is None:
            raise ValueError("Missing `config.encoder_text_token_ids`.")
        return encoder_text_token_ids

    def embed_text_tokens(self, token_text: str | None) -> torch.Tensor | None:
        if token_text is None:
            return None
        token_ids = self._require_encoder_text_token_ids().get(token_text)
        if token_ids is None:
            raise ValueError(f"Missing token ids for encoder boundary text: {token_text!r}")
        token_ids = torch.tensor(token_ids, device=self.llm_model_embed_tokens.weight.device)
        return self.llm_model_embed_tokens(token_ids)

    def _require_media_token_ids(self) -> dict[str, int]:
        media_token_ids = getattr(self.config, "media_token_ids", None)
        if not media_token_ids:
            raise ValueError("Missing `config.media_token_ids`.")
        return media_token_ids

    def _init_media_encoders(self):
        def _parse_tokens(cfg, default_end="\n"):
            start = cfg.get("start_tokens")
            end = cfg.get("end_tokens", default_end)
            end = None if end == "None" else end
            sep = cfg.get("sep_tokens")
            return start, end, sep

        img_cfg = copy.deepcopy(self.config.image_encoder)
        vid_cfg = copy.deepcopy(self.config.video_encoder)
        snd_cfg = copy.deepcopy(self.config.sound_encoder)
        for dct in (img_cfg, vid_cfg, snd_cfg):
            dct.pop("_target_", None)

        self._image_start_tokens, self._image_end_tokens, _ = _parse_tokens(img_cfg)
        self._video_start_tokens, self._video_end_tokens, self._video_sep_tokens = _parse_tokens(vid_cfg)
        self._video_pool_sizes = vid_cfg.get("pool_sizes", [[1, 1, 1]])
        self._sound_start_tokens, self._sound_end_tokens, _ = _parse_tokens(snd_cfg)
        self._time_embeddings = {}

        self._video_embed_time = vid_cfg.get("embed_time", "False") in ("True", True)
        if self._video_embed_time:
            self._video_time_embed_type = vid_cfg.get("time_embed_type", "pixel")
            self._video_period_fix, self._video_max_time = self._create_time_embedding("video", vid_cfg)

        self._sound_embed_time = snd_cfg.get("embed_time", "False") in ("True", True)
        if self._sound_embed_time:
            self._sound_time_embed_type = snd_cfg.get("time_embed_type", "pixel")
            self._sound_period_fix, self._sound_max_time = self._create_time_embedding("sound", snd_cfg)

    def _create_time_embedding(self, key: str, cfg: dict):
        trope_dim = cfg.get("trope_dim", 128)
        trope_theta = cfg.get("trope_theta", 50000)
        max_time = cfg.get("max_time")
        time_embed_type = cfg.get("time_embed_type", "pixel")
        period_fix = cfg.get("period_fix", False)

        period_mode = None
        if isinstance(period_fix, str) and period_fix in ("shortest", "longest"):
            period_mode = period_fix
            period_fix = "MTCT"

        if period_fix == "MTCT":
            kwargs = {"dim": trope_dim, "max_time": max_time}
            if period_mode is not None:
                kwargs["period_mode"] = period_mode
            self._time_embeddings[key] = MaxTimeContinuousTimeRotaryEmbedding(**kwargs)
        elif key == "video":
            if time_embed_type == "lang":
                self._time_embeddings[key] = RotaryEmbedding(
                    dim=trope_dim, freqs_for="lang", theta=trope_theta, max_time=max_time
                )
            elif time_embed_type == "pixel":
                self._time_embeddings[key] = RotaryEmbedding(dim=trope_dim, freqs_for="pixel", max_freq=256)
        elif key == "sound":
            if time_embed_type in ("pixel", "lang"):
                self._time_embeddings[key] = RotaryEmbedding(
                    dim=trope_dim, freqs_for=time_embed_type, max_freq=256, max_time=max_time
                )
        return period_fix, max_time

    def _freeze_untrained_modules(self):
        if not self.training:
            return

        for module, flag_name in (
            (self.vision_tower, "tune_vision_tower"),
            (getattr(self, "sound_tower", None), "tune_sound_tower"),
            (self.mm_projector, "tune_mm_projector"),
            (getattr(self, "sound_mm_projector", None), "tune_sound_mm_projector"),
        ):
            if module is not None and not getattr(self.config, flag_name, False):
                module.eval()


class AudioVisualFlamingoForConditionalGeneration(AudioVisualFlamingoPretrainedModel, GenerationMixin):
    def __init__(self, config: AudioVisualFlamingoConfig, *args, **kwargs):
        super().__init__(config)
        _ = (args, kwargs)
        if not getattr(config, "dynamic_s2", False):
            raise NotImplementedError("Current AudioVisualFlamingo checkpoint requires `dynamic_s2=True`.")
        self.vision_tower = SiglipVisionTowerDynamicS2(config)
        audio_cfg = copy.deepcopy(config.audio_config)
        audio_cfg._attn_implementation = config._attn_implementation
        self.sound_tower = AutoModel.from_config(audio_cfg)

        text_cfg = copy.deepcopy(config.text_config)
        text_cfg._attn_implementation = config._attn_implementation
        model_max_length = getattr(config, "model_max_length", None)
        if model_max_length is not None:
            text_cfg.model_max_length = model_max_length
            orig_ctx_len = getattr(text_cfg, "max_position_embeddings", None)
            if orig_ctx_len is not None and model_max_length > orig_ctx_len:
                text_cfg.rope_scaling = {
                    "type": "linear",
                    "factor": float(math.ceil(model_max_length / orig_ctx_len)),
                }

        self.llm = AutoModelForCausalLM.from_config(text_cfg)
        self.mm_projector = MultimodalProjector(
            vision_hidden_size=self.vision_tower.hidden_size,
            text_hidden_size=self.llm.config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.sound_mm_projector = SoundMultimodalProjector(
            audio_hidden_size=self.sound_tower.config.d_model,
            text_hidden_size=self.llm.config.hidden_size,
            bias=config.projector_bias,
        )
        self.vocab_size = self.llm.config.vocab_size
        self._init_media_encoders()
        self.training = self.llm.training
        if self.training:
            self.train()
        else:
            self.eval()

        self.config.text_config = self.llm.config
        self.config.vision_config = self.vision_tower.config
        self.config.audio_config = self.sound_tower.config
        self.post_init()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.llm.set_decoder(decoder)

    def get_decoder(self):
        return self.llm.get_decoder()

    @property
    def language_model(self):
        return self.llm

    def _encode_visual_features(self, images: torch.Tensor, block_sizes: tuple[int, ...] | None = None):
        if not getattr(self.config, "dynamic_s2", False):
            raise NotImplementedError("Current AudioVisualFlamingo checkpoint requires `dynamic_s2=True`.")
        if len(images) == 0:
            return []

        if block_sizes is None:
            block_sizes = [None] * len(images)

        image_features = self.vision_tower(images)
        image_features, new_block_sizes = self.merge_features_for_dynamic_s2(image_features, block_sizes)
        image_features = [
            self.split_chessboard(feature, block_size[0], block_size[1])
            for feature, block_size in zip(image_features, new_block_sizes)
        ]
        image_features = torch.cat([_channel_first_to_tokens(feature) for feature in image_features], dim=0)
        image_features = self.mm_projector(image_features.to(self.device, self.dtype))
        image_features = list(
            image_features.split([block_size[0] * block_size[1] for block_size in new_block_sizes], dim=0)
        )
        image_features = [
            self.merge_chessboard(feature, block_size[0], block_size[1])
            for feature, block_size in zip(image_features, new_block_sizes)
        ]
        image_features = [_channel_first_to_tokens(feature)[0] for feature in image_features]
        if all(feature.shape[0] == image_features[0].shape[0] for feature in image_features):
            return torch.stack(image_features, dim=0)
        return image_features

    def merge_features_for_dynamic_s2(self, image_features, block_sizes):
        scales = self.vision_tower.scales
        resize_output_to_scale_idx = self.vision_tower.resize_output_to_scale_idx
        image_features_each_image = []
        new_block_sizes = []
        block_cnt = 0
        for block_size_each_image in block_sizes:
            if block_size_each_image is None:
                cur_features = image_features[block_cnt : block_cnt + 1]
                spatial_size = int(cur_features.shape[1] ** 0.5)
                cur_features = _tokens_to_channel_first(cur_features, spatial_size, spatial_size)
                cur_features = cur_features.repeat(1, len(scales), 1, 1)
                image_features_each_image.append(cur_features)
                new_block_sizes.append((1, 1))
                block_cnt += 1
                continue

            cur_features_each_scale = []
            for scale in scales[:-1]:
                num_blocks_this_scale = (scale // scales[0]) ** 2
                cur_features_each_scale.append(
                    self.merge_chessboard(
                        image_features[block_cnt : block_cnt + num_blocks_this_scale],
                        num_split_h=scale // scales[0],
                        num_split_w=scale // scales[0],
                    )
                )
                block_cnt += num_blocks_this_scale
            num_blocks_last_scale = block_size_each_image[0] * block_size_each_image[1]
            cur_features_each_scale.append(
                self.merge_chessboard(
                    image_features[block_cnt : block_cnt + num_blocks_last_scale],
                    num_split_h=block_size_each_image[0],
                    num_split_w=block_size_each_image[1],
                )
            )
            block_cnt += num_blocks_last_scale
            output_size = cur_features_each_scale[resize_output_to_scale_idx].shape[-2:]
            cur_features = torch.cat(
                [
                    F.interpolate(cur_features_each_scale[i].to(torch.float32), size=output_size, mode="area").to(
                        cur_features_each_scale[i].dtype
                    )
                    for i in range(len(cur_features_each_scale))
                ],
                dim=1,
            )
            image_features_each_image.append(cur_features)
            if resize_output_to_scale_idx == len(scales) - 1 or resize_output_to_scale_idx == -1:
                new_block_sizes.append(block_size_each_image)
            else:
                new_block_sizes.append(
                    (
                        scales[resize_output_to_scale_idx] // scales[0],
                        scales[resize_output_to_scale_idx] // scales[0],
                    )
                )
        assert block_cnt == len(image_features)
        return image_features_each_image, new_block_sizes

    @staticmethod
    def split_chessboard(x, num_split_h, num_split_w):
        bsz, channels, height, width = x.shape
        assert height % num_split_h == 0 and width % num_split_w == 0
        split_h, split_w = height // num_split_h, width // num_split_w
        return torch.cat(
            [
                x[:, :, i * split_h : (i + 1) * split_h, j * split_w : (j + 1) * split_w]
                for i in range(num_split_h)
                for j in range(num_split_w)
            ],
            dim=0,
        )

    @staticmethod
    def merge_chessboard(x, num_split_h, num_split_w):
        batch = x.shape[0]
        if x.dim() == 3:
            num_tokens = x.shape[1]
            spatial_size = int(num_tokens**0.5)
            x = _tokens_to_channel_first(x, spatial_size, spatial_size)
        assert batch % (num_split_h * num_split_w) == 0
        base_batch = batch // (num_split_h * num_split_w)
        return torch.cat(
            [
                torch.cat(
                    [
                        x[(i * num_split_w + j) * base_batch : (i * num_split_w + j + 1) * base_batch]
                        for j in range(num_split_w)
                    ],
                    dim=-1,
                )
                for i in range(num_split_h)
            ],
            dim=-2,
        )

    def encode_video(
        self,
        inp,
        block_sizes: tuple[int, ...] | None = None,
        mm_info: dict | None = None,
        num_frames: list[int] | None = None,
    ):
        _ = (mm_info, num_frames)
        if block_sizes is not None:
            raise ValueError(f"Video block sizes are not supported: {block_sizes}")
        if not inp:
            return []
        return self._encode_visual_features(torch.cat(inp, dim=0))

    def encode_images(
        self,
        images,
        block_sizes: tuple[int, ...] | None = None,
        mm_info: dict | None = None,
        num_frames: list[int] | None = None,
    ):
        _ = (mm_info, num_frames)
        return self._encode_visual_features(images, block_sizes=block_sizes)

    def _get_sound_chunk_length(self) -> int:
        return (
            self.sound_tower.config.max_source_positions
            * self.sound_tower.conv1.stride[0]
            * self.sound_tower.conv2.stride[0]
        )

    def _forward_sound_tower_batch(self, input_features: torch.Tensor) -> torch.Tensor:
        batch_size, n_mels, seq_len = input_features.shape
        chunk_length = self._get_sound_chunk_length()
        num_chunks = (seq_len + chunk_length - 1) // chunk_length

        padded_chunks = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_length
            end_idx = min(start_idx + chunk_length, seq_len)
            chunk = input_features[:, :, start_idx:end_idx]
            if chunk.shape[2] < chunk_length:
                chunk = F.pad(chunk, (0, chunk_length - chunk.shape[2]), mode="constant", value=0)
            padded_chunks.append(chunk)

        all_chunks = torch.cat(padded_chunks, dim=0).reshape(batch_size * num_chunks, n_mels, chunk_length)
        chunk_outputs = self.sound_tower(all_chunks, return_dict=True)
        hidden_states = chunk_outputs.last_hidden_state
        _, chunk_seq_len, hidden_size = hidden_states.shape
        return hidden_states.reshape(batch_size, num_chunks * chunk_seq_len, hidden_size)

    def encode_sound(self, sounds, mm_info: dict | None = None):
        _ = mm_info
        audio_features = []
        audio_output_lengths = []
        for sound in sounds:
            if hasattr(sound, "input_features") or (isinstance(sound, dict) and "input_features" in sound):
                sound = sound["input_features"]
            sound_dtype = sound.dtype
            sound = sound.to(device=self.sound_tower.device, dtype=self.sound_tower.dtype)
            sound_feature = self._forward_sound_tower_batch(sound).to(sound_dtype)
            audio_features.append(sound_feature)
            audio_output_lengths.append(sound_feature.shape[1])

        if not audio_features:
            return []

        audio_features = torch.cat(audio_features, dim=1).squeeze(0)
        projector_param = next(self.sound_mm_projector.parameters(), None)
        if projector_param is not None and audio_features.dtype != projector_param.dtype:
            audio_features = audio_features.to(projector_param.dtype)
        audio_features = self.sound_mm_projector(audio_features)

        split_audio_features = []
        start = 0
        for length in audio_output_lengths:
            split_audio_features.append(audio_features[start : start + length])
            start += length
        return split_audio_features

    def _embed_image_features(
        self, images: list[torch.Tensor], config: dict[str, Any], mm_info: dict
    ) -> list[torch.Tensor]:
        _ = mm_info
        features = self.encode_images(torch.stack(images, dim=0), block_sizes=config.get("block_sizes"))
        start_embeds = self.embed_text_tokens(self._image_start_tokens)
        end_embeds = self.embed_text_tokens(self._image_end_tokens)
        image_features = []
        for feature in features:
            if start_embeds is not None:
                feature = torch.cat([start_embeds, feature], dim=0)
            if end_embeds is not None:
                feature = torch.cat([feature, end_embeds], dim=0)
            image_features.append(feature)
        return image_features

    def _embed_video_features(
        self, videos: list[torch.Tensor], config: dict[str, Any], mm_info: dict
    ) -> list[torch.Tensor]:
        _ = config
        num_frames = [video.shape[0] for video in videos]
        features = self.encode_video(videos, mm_info=mm_info, num_frames=num_frames)
        features = torch.split(features, num_frames)
        start_embeds = self.embed_text_tokens(self._video_start_tokens)
        end_embeds = self.embed_text_tokens(self._video_end_tokens)
        sep_embeds = self.embed_text_tokens(self._video_sep_tokens)
        if not self._video_embed_time:
            return [self._tsp_process(feature, start_embeds, end_embeds, sep_embeds) for feature in features]

        batch_size = len(mm_info["video_info"])
        device = features[0].device
        new_time_embeds = None
        if self._video_time_embed_type == "learned_embed":
            times_list, video_idx = [], 0
            for i in range(batch_size):
                video_info = mm_info["video_info"][i]
                if video_info is None:
                    continue
                for j in range(len(video_info)):
                    feature = features[video_idx]
                    if video_info[j] == "dummy":
                        times = torch.zeros(feature.shape[0], device=device, dtype=feature.dtype)
                    else:
                        times = torch.tensor(video_info[j]["video_frame_times"]).to(device)
                    for pool_size in self._video_pool_sizes:
                        temporal_pool = pool_size[0]
                        if temporal_pool != 1:
                            if len(times) % temporal_pool != 0:
                                remainder = len(times) % temporal_pool
                                times = torch.cat([times, times[-remainder:].mean().expand(temporal_pool - remainder)])
                            times = pool(times, temporal_pool, 0)
                    times_list.append(times)
                    video_idx += 1
            original_lengths = [len(times) for times in times_list]
            max_length = max(original_lengths)
            for i in range(len(times_list)):
                if len(times_list[i]) < max_length:
                    times_list[i] = torch.cat(
                        [times_list[i], torch.zeros(max_length - len(times_list[i])).to(times_list[i].device)]
                    )
            times_tensor = torch.stack(times_list, dim=0)
            time_embeds_all = self._time_embeddings["video"](times_tensor, dtype=features[0].dtype)
            new_time_embeds = []
            for i in range(len(times_list)):
                new_time_embeds.append(
                    time_embeds_all[i][: original_lengths[i]].unsqueeze(1).expand(-1, features[0].shape[1], -1)
                )
            new_time_embeds[0] = new_time_embeds[0] + 0 * time_embeds_all.mean()

        new_features, video_idx = [], 0
        for i in range(batch_size):
            video_info = mm_info["video_info"][i]
            if video_info is None:
                continue
            for j in range(len(video_info)):
                feature = features[video_idx]
                if video_info[j] == "dummy":
                    times = torch.zeros(feature.shape[0], device=device, dtype=feature.dtype)
                else:
                    times = torch.tensor(video_info[j]["video_frame_times"]).to(device)
                if self._video_time_embed_type == "learned_embed":
                    feature = self._tsp_process(
                        feature, start_embeds, end_embeds, sep_embeds, time_embed=new_time_embeds[video_idx]
                    )
                else:
                    feature = self._tsp_process(feature, start_embeds, end_embeds, sep_embeds, times=times)
                new_features.append(feature)
                video_idx += 1
        assert video_idx == len(features)
        return new_features

    def _tsp_process(
        self,
        inputs: torch.Tensor,
        start_token_embeds: torch.Tensor | None,
        end_token_embeds: torch.Tensor | None,
        sep_token_embeds: torch.Tensor | None,
        times: torch.Tensor | None = None,
        time_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_frames, num_spatial_tokens = inputs.shape[:2]
        spatial_length = int(num_spatial_tokens**0.5)
        outputs = []
        for pool_size in self._video_pool_sizes:
            features = inputs.view(num_frames, spatial_length, spatial_length, -1)
            for dim, pool_factor in enumerate(pool_size):
                features = pool(features, pool_factor, dim=dim)
            features = features.flatten(1, 2)
            if self._video_embed_time:
                device = features.device
                if self._video_time_embed_type in ("pixel", "lang"):
                    temporal_pool = pool_size[0]
                    if temporal_pool != 1:
                        pooled_times = times
                        if len(pooled_times) % temporal_pool != 0:
                            remainder = len(pooled_times) % temporal_pool
                            pooled_times = torch.cat(
                                [pooled_times, pooled_times[-remainder:].mean().expand(temporal_pool - remainder)]
                            )
                        new_times = pool(pooled_times, temporal_pool, 0)
                    else:
                        new_times = times
                    pos_emb = _move_rotary_module_to_device(self._time_embeddings["video"], device)
                    self._time_embeddings["video"] = pos_emb
                    if self._video_period_fix == "True":
                        angle = (
                            new_times.to(device) / self._video_max_time * 2 * np.pi
                            if self._video_max_time is not None
                            else new_times.to(device)
                        )
                    elif self._video_period_fix == "MTCT":
                        time_values = new_times.unsqueeze(0) if new_times.ndim == 1 else new_times
                        freqs = pos_emb(time_values.float()).squeeze(0).unsqueeze(1)
                        features = apply_rotary_emb(freqs, features, seq_dim=0)
                    else:
                        angle = (-new_times * 2 * np.pi).to(device)
                    if self._video_period_fix != "MTCT":
                        freqs = pos_emb.get_axial_freqs(new_times.shape[0], features.shape[-2]).to(device)
                        angle_exp = (
                            angle.unsqueeze(1)
                            .unsqueeze(2)
                            .expand(new_times.shape[0], features.shape[-2], freqs.shape[-1])
                        )
                        features = apply_rotary_emb(freqs * angle_exp, features)
                elif self._video_time_embed_type == "learned_embed":
                    features = features + time_embed
            if start_token_embeds is not None:
                features = torch.cat(
                    [start_token_embeds.unsqueeze(0).expand(features.shape[0], -1, -1), features], dim=1
                )
            if end_token_embeds is not None:
                features = torch.cat(
                    [features, end_token_embeds.unsqueeze(0).expand(features.shape[0], -1, -1)], dim=1
                )
            features = features.flatten(0, 1)
            if sep_token_embeds is not None:
                features = torch.cat([features, sep_token_embeds], dim=0)
            outputs.append(features)
        return torch.cat(outputs, dim=0)

    def _embed_sound_features(
        self, sounds: list[torch.Tensor], config: dict[str, Any], mm_info: dict
    ) -> list[torch.Tensor]:
        _ = config
        features = self.encode_sound(sounds, mm_info=mm_info)
        start_embeds = self.embed_text_tokens(self._sound_start_tokens)
        end_embeds = self.embed_text_tokens(self._sound_end_tokens)
        if not self._sound_embed_time:
            return [self._process_sound_feature(feature, start_embeds, end_embeds) for feature in features]
        device = features[0].device
        feature_count = len(features)
        batch_size = len(mm_info["audio_info"])
        time_embeds_all = None
        if self._sound_time_embed_type == "learned_embed":
            times_list, audio_idx = [], 0
            for i in range(batch_size):
                audio_info = mm_info["audio_info"][i]
                if audio_info is None:
                    continue
                for j in range(len(audio_info)):
                    feature = features[audio_idx]
                    if audio_info[j] == "dummy":
                        times = torch.zeros(feature.shape[0], device=device, dtype=feature.dtype)
                    else:
                        chunk_length = audio_info[j]["new_audio_chunk_length"]
                        seconds_per_embed = chunk_length / feature.shape[0]
                        audio_start = audio_info[j]["audio_start_sec"]
                        times = torch.tensor(
                            [
                                audio_start + k * seconds_per_embed + seconds_per_embed / 2
                                for k in range(feature.shape[0])
                            ]
                        ).to(device)
                    times_list.append(times)
                    audio_idx += 1
            times_tensor = torch.stack(times_list, dim=0)
            time_embeds_all = self._time_embeddings["sound"](times_tensor, dtype=features[0].dtype)
        new_features, audio_idx = [], 0
        for i in range(batch_size):
            audio_info = mm_info["audio_info"][i]
            if audio_info is None:
                continue
            for j in range(len(audio_info)):
                feature = features[audio_idx]
                if audio_info[j] == "dummy":
                    times = torch.zeros(feature.shape[0], device=device, dtype=feature.dtype)
                else:
                    chunk_length = audio_info[j]["new_audio_chunk_length"]
                    seconds_per_embed = chunk_length / feature.shape[0]
                    audio_start = audio_info[j]["audio_start_sec"]
                    times = torch.tensor(
                        [audio_start + k * seconds_per_embed + seconds_per_embed / 2 for k in range(feature.shape[0])]
                    ).to(device)
                if self._sound_time_embed_type == "learned_embed":
                    feature = self._process_sound_feature(
                        feature, start_embeds, end_embeds, time_embed=time_embeds_all[audio_idx]
                    )
                else:
                    feature = self._process_sound_feature(feature, start_embeds, end_embeds, times=times)
                new_features.append(feature)
                audio_idx += 1
        assert audio_idx == feature_count
        return new_features

    def _process_sound_feature(
        self,
        features: torch.Tensor,
        start_token_embeds: torch.Tensor | None,
        end_token_embeds: torch.Tensor | None,
        times: torch.Tensor | None = None,
        time_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = features.to(self.device)
        device = features.device
        if self._sound_embed_time:
            if self._sound_time_embed_type in ("pixel", "lang"):
                new_times = times.unsqueeze(0)
                pos_emb = _move_rotary_module_to_device(self._time_embeddings["sound"], device)
                self._time_embeddings["sound"] = pos_emb
                if self._sound_period_fix == "True":
                    angle = (
                        new_times.to(device) / self._sound_max_time * 2 * np.pi
                        if self._sound_max_time is not None
                        else new_times.to(device)
                    )
                elif self._sound_period_fix == "MTCT":
                    freqs = pos_emb(new_times.float()).squeeze(0)
                    features = apply_rotary_emb(freqs, features)
                else:
                    angle = (-new_times * 2 * np.pi).to(device)
                if self._sound_period_fix != "MTCT":
                    freqs = pos_emb.get_axial_freqs(new_times.shape[0], features.shape[-2]).to(device)
                    angle_exp = angle.unsqueeze(2).expand(new_times.shape[0], features.shape[-2], freqs.shape[-1])
                    freqs = (freqs * angle_exp).squeeze(0)
                    features = apply_rotary_emb(freqs, features)
            elif self._sound_time_embed_type == "learned_embed":
                features = features + time_embed
        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def _embed(
        self,
        input_ids: torch.Tensor,
        media: dict[str, list[torch.Tensor]],
        media_config: dict[str, dict[str, Any]],
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        media = copy.deepcopy(media)
        media_config = copy.deepcopy(media_config)
        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        text_embeds = self.llm_model_embed_tokens(input_ids)
        mm_info = {}
        video_info = media.pop("video_info", None)
        audio_info = media.pop("audio_info", None)
        if video_info is not None:
            mm_info["video_info"] = video_info
        if audio_info is not None:
            mm_info["audio_info"] = audio_info
        media_embeds = self.__embed_media_tokens(media, media_config, mm_info) if media is not None else {}

        video_sound_embeds_idx = 0
        sep_embed = self.embed_text_tokens("\n")
        llm_embed_dtype = self.llm_model_embed_tokens.weight.dtype
        text_embeds = text_embeds.to(llm_embed_dtype)
        sep_embed = sep_embed.to(text_embeds.dtype)
        if video_info is not None and self.config.load_audio_in_video and self.config.interleaved_vis_aud_in_video:
            assert self._video_end_tokens is None, "end_tokens must be None for interleaved vis-aud in video"
            new_video_embeds = deque()
            video_embeds_idx = 0
            for k in range(len(video_info)):
                if video_info[k] is None:
                    continue
                for i in range(len(video_info[k])):
                    has_audio = video_info[k][i]["has_audio"]
                    if not has_audio:
                        new_video_embeds.append(media_embeds["video"][video_embeds_idx])
                        video_embeds_idx += 1
                        continue
                    if video_sound_embeds_idx >= len(media_embeds["sound"]):
                        raise ValueError(
                            f"Sound embeddings index {video_sound_embeds_idx} out of bounds for video_info[{k}][{i}]"
                        )
                    segment_aud_indices_list = video_info[k][i]["segment_aud_indices_list"]
                    segment_vis_indices_list = video_info[k][i]["segment_vis_indices_list"]
                    vis_fea_len_per_frame = (
                        media_embeds["video"][video_embeds_idx].shape[0] / video_info[k][i]["expected_frame_count"]
                    )
                    aud_fea_len_per_stft_frame = (
                        media_embeds["sound"][video_sound_embeds_idx].shape[0]
                        / audio_info[k][i]["new_audio_n_stft_frames"]
                    )
                    vis_end = 0
                    aud_end = 0
                    new_video_embed = []
                    for j in range(len(segment_vis_indices_list)):
                        vis_aud_fea = []
                        if len(segment_vis_indices_list[j]) > 0:
                            new_frames = [
                                int(np.ceil((frame + 1) * vis_fea_len_per_frame))
                                for frame in segment_vis_indices_list[j]
                            ]
                            vis_fea_end = min(new_frames[-1], media_embeds["video"][video_embeds_idx].shape[0])
                            vis_fea = media_embeds["video"][video_embeds_idx][vis_end:vis_fea_end]
                            vis_end = vis_fea_end
                            vis_aud_fea.append(vis_fea)
                        vis_aud_fea.append(sep_embed)
                        if len(segment_aud_indices_list[j]) > 0:
                            new_audio_indices = [
                                int(np.ceil(fea * aud_fea_len_per_stft_frame)) for fea in segment_aud_indices_list[j]
                            ]
                            aud_fea_end = min(
                                new_audio_indices[-1], media_embeds["sound"][video_sound_embeds_idx].shape[0]
                            )
                            aud_fea = media_embeds["sound"][video_sound_embeds_idx][aud_end:aud_fea_end]
                            vis_aud_fea.append(aud_fea)
                            aud_end = aud_fea_end
                        vis_aud_fea.append(sep_embed)
                        new_video_embed.append(torch.cat(vis_aud_fea, dim=0))
                    video_sound_embeds_idx += 1
                    new_video_embeds.append(torch.cat(new_video_embed, dim=0))
                    video_embeds_idx += 1
            assert len(new_video_embeds) == len(media_embeds["video"])
            media_embeds["video"] = new_video_embeds

        batch_size = labels.shape[0]
        text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]
        media_token_ids = self._require_media_token_ids()
        media_tokens = {token_id: name for name, token_id in media_token_ids.items()}
        inputs_m, labels_m = [], []
        sound_embeds_idx = 0
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    name = media_tokens[input_ids[k][pos].item()]
                    if input_ids[k][pos].item() == media_token_ids["sound"]:
                        if self.config.interleaved_vis_aud_in_video and sound_embeds_idx < video_sound_embeds_idx:
                            media_embeds[name].popleft()
                            sound_embeds_idx += 1
                            pos += 1
                            continue
                        sound_embeds_idx += 1
                    end = pos + 1
                    current_input = media_embeds[name].popleft()
                    current_label = torch.full(
                        [current_input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype
                    )
                else:
                    end = pos
                    while end < len(labels[k]) and input_ids[k][end].item() not in media_tokens:
                        end += 1
                    current_input = text_embeds[k][pos:end]
                    current_label = labels[k][pos:end]
                inputs_mk.append(current_input)
                labels_mk.append(current_label)
                pos = end
            inputs_m.append(torch.cat(inputs_mk, dim=0))
            labels_m.append(torch.cat(labels_mk, dim=0))
        inputs, labels = inputs_m, labels_m
        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed! Still {len(media_embeds[name])} left.")
        inputs, labels = self.__truncate_sequence(inputs, labels)
        return self.__batchify_sequence(inputs, labels)

    def __embed_media_tokens(
        self, media: dict[str, list[torch.Tensor]], media_config: dict[str, dict[str, Any]], mm_info
    ):
        embeds = defaultdict(deque)
        embed_fn = {
            "image": self._embed_image_features,
            "video": self._embed_video_features,
            "sound": self._embed_sound_features,
        }
        for name in media:
            if name == "sound":
                sound_media = media.get(name, [])
                if len(sound_media) == 0:
                    continue
                if not all(
                    hasattr(sound, "input_features") or (isinstance(sound, dict) and "input_features" in sound)
                    for sound in sound_media
                ):
                    raise ValueError("Expected pre-extracted sound features in `media['sound']`.")
            if len(media[name]) > 0:
                embeds[name] = deque(embed_fn[name](media[name], media_config[name], mm_info))
        return embeds

    def __truncate_sequence(self, inputs: list[torch.Tensor], labels: list[torch.Tensor]):
        model_max_length = getattr(self.config, "model_max_length", None)
        if model_max_length is None:
            model_max_length = getattr(self.llm.config, "model_max_length", 2048)
        model_max_length = int(model_max_length)
        if self.training and any(len(current_input) > model_max_length for current_input in inputs):
            warnings.warn(f"Truncating sequences to `model_max_length` ({model_max_length}).")
            inputs = [current_input[:model_max_length] for current_input in inputs]
            labels = [label[:model_max_length] for label in labels]
        return inputs, labels

    def __batchify_sequence(self, inputs: list[torch.Tensor], labels: list[torch.Tensor]):
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)
        padding_side = getattr(self.config, "padding_side", "left")
        inputs_p, labels_p = [], []
        for k in range(batch_size):
            pad_size = max_length - inputs[k].shape[0]
            input_padding = torch.zeros((pad_size, hidden_size), dtype=inputs[k].dtype, device=device)
            label_padding = torch.full((pad_size,), IGNORE_INDEX, dtype=labels[k].dtype, device=device)
            if padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                input_padding = torch.cat([inputs[k], input_padding], dim=0)
                label_padding = torch.cat([labels[k], label_padding], dim=0)
            else:
                labels[k] = labels[k].to(device)
                attention_mask[k, : -inputs[k].shape[0]] = False
                input_padding = torch.cat([input_padding, inputs[k]], dim=0)
                label_padding = torch.cat([label_padding, labels[k]], dim=0)
            inputs_p.append(input_padding)
            labels_p.append(label_padding)
        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    def repack_multimodal_data(self, inputs_embeds, attention_mask, position_ids, labels):
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        seqlens = [attention_mask[k].sum().item() for k in range(batch_size)]
        inputs_embeds_p = [inputs_embeds[k][attention_mask[k]] for k in range(batch_size)]
        attention_mask_p = [torch.ones(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        position_ids_p = [torch.arange(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        labels_p = [labels[k][attention_mask[k]] for k in range(batch_size)]
        inputs_embeds_p.append(torch.zeros(1, inputs_embeds.shape[-1], dtype=inputs_embeds.dtype, device=device))
        attention_mask_p.append(torch.tensor([0], dtype=torch.int, device=device))
        position_ids_p.append(torch.tensor([0], dtype=torch.int, device=device))
        labels_p.append(torch.tensor([IGNORE_INDEX], dtype=torch.int, device=device))
        for label in labels_p:
            label[0] = IGNORE_INDEX
        inputs_embeds_p = torch.cat(inputs_embeds_p, dim=0).unsqueeze(0)
        attention_mask_p = torch.cat(attention_mask_p, dim=0).unsqueeze(0)
        position_ids_p = torch.cat(position_ids_p, dim=0).unsqueeze(0)
        labels_p = torch.cat(labels_p, dim=0).unsqueeze(0)
        if hasattr(self, "pad_to_multiple_of"):
            batch_size, max_length, cur_length = labels_p.shape[0], labels_p.shape[1], labels_p.shape[1]
            hidden_size = inputs_embeds_p.shape[-1]
            if max_length % self.pad_to_multiple_of != 0:
                max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
                difference = max_length - cur_length
                inputs_embeds_p = torch.cat(
                    (
                        inputs_embeds_p,
                        torch.full((batch_size, difference, hidden_size), self.llm.pad_token_id).to(inputs_embeds_p),
                    ),
                    dim=1,
                )
                labels_p = torch.cat(
                    (labels_p, torch.full((batch_size, difference), IGNORE_INDEX).to(labels_p)), dim=1
                )
                attention_mask_p = torch.cat(
                    (attention_mask_p, torch.zeros((batch_size, difference), dtype=torch.bool).to(attention_mask_p)),
                    dim=1,
                )
                position_ids_p = torch.cat(
                    (position_ids_p, torch.full((batch_size, difference), -1).to(position_ids_p)), dim=1
                )
        return inputs_embeds_p, attention_mask_p, position_ids_p, labels_p

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        media: dict[str, list[torch.Tensor]] | None = None,
        media_config: list | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        packing: bool = True,
        force_packing: bool = False,
        seqlens_in_batch: torch.LongTensor | None = None,
        dpo_forward: bool = False,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        _ = (pixel_values, seqlens_in_batch)
        self._freeze_untrained_modules()
        if media_config is None:
            media_config = defaultdict(dict)
        if inputs_embeds is None:
            if media is None:
                if input_ids is None:
                    raise ValueError("Either `inputs_embeds` or `input_ids` must be provided.")
                inputs_embeds = self.llm_model_embed_tokens(input_ids)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                inputs_embeds, labels, attention_mask = self._embed(
                    input_ids, media, media_config, labels, attention_mask
                )
        if force_packing or (packing and self.training and not dpo_forward):
            inputs_embeds, attention_mask, position_ids, labels = self.repack_multimodal_data(
                inputs_embeds, attention_mask, position_ids, labels
            )
        llm_param = next(self.llm.parameters(), None)
        if llm_param is not None and inputs_embeds.dtype != llm_param.dtype:
            inputs_embeds = inputs_embeds.to(llm_param.dtype)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs,
        )
        if dpo_forward:
            return outputs.logits, labels
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        media=None,
        media_config=None,
        attention_mask=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        is_first_iteration = bool(kwargs.get("is_first_iteration", False))
        is_first_step = (
            is_first_iteration or past_key_values is None or (cache_position is not None and cache_position[0] == 0)
        )
        if is_first_step and inputs_embeds is None and media is not None:
            if media_config is None:
                media_config = defaultdict(dict)
            inputs_embeds, _, attention_mask = self._embed(input_ids, media, media_config, None, attention_mask)
        model_inputs = self.llm.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )
        if is_first_step and inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["attention_mask"] = attention_mask
            model_inputs["input_ids"] = None
            seq_len = attention_mask.shape[-1]
            cache_pos = model_inputs.get("cache_position")
            if cache_pos is None or cache_pos.shape[0] != seq_len:
                model_inputs["cache_position"] = torch.arange(seq_len, device=inputs_embeds.device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            model_inputs["position_ids"] = position_ids
        model_inputs["media"] = None
        model_inputs["media_config"] = None
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        attention_mask = model_kwargs.get("attention_mask")
        logits = getattr(outputs, "logits", None)
        if (
            model_kwargs.get("media") is not None
            and attention_mask is not None
            and logits is not None
            and attention_mask.shape[-1] != logits.shape[-2]
        ):
            batch_size = attention_mask.shape[0]
            seq_len = logits.shape[-2]
            model_kwargs["attention_mask"] = attention_mask.new_ones((batch_size, seq_len))
            model_kwargs["cache_position"] = torch.arange(seq_len, device=attention_mask.device)
            if model_kwargs.get("position_ids") is not None:
                position_ids = model_kwargs["attention_mask"].long().cumsum(-1) - 1
                position_ids.masked_fill_(model_kwargs["attention_mask"] == 0, 0)
                model_kwargs["position_ids"] = position_ids
            model_kwargs["media"] = None
            model_kwargs["media_config"] = None
        return super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens
        )


__all__ = [
    "AudioVisualFlamingoConfig",
    "AudioVisualFlamingoForConditionalGeneration",
    "AudioVisualFlamingoPretrainedModel",
]
