# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from math import pi
from typing import Any, Literal

import numpy as np
import torch
from beartype import beartype
from einops import rearrange, repeat
from torch import Tensor, broadcast_tensors, einsum, nn
from torch.nn import Module


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pool(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    # return x.view(x.shape[:dim] + (-1, size) + x.shape[dim + 1 :]).mean(dim + 1)
    # Reshape x to group elements along the specified dimension into chunks of 'size', then average over those chunks.

    # Check if the dimension is divisible by the pool size, if not pad with mean values
    if x.shape[dim] % size != 0:
        print(
            f"Warning: dimension {dim} with size {x.shape[dim]} is not divisible by pool size {size}, padding with mean values"
        )
        remainder = x.shape[dim] % size
        pad_len = size - remainder

        # Get the mean of the last few elements along the dimension to be pooled
        last_elements = x.narrow(dim, x.shape[dim] - remainder, remainder)
        mean_value = last_elements.mean()

        # Create padding tensor with the same shape as x except for the dimension being pooled
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        padding = torch.ones(pad_shape, device=x.device, dtype=x.dtype) * mean_value

        # Concatenate the original tensor with the padding along the specified dimension
        x = torch.cat([x, padding], dim=dim)

    shape_before = x.shape[:dim]
    shape_after = x.shape[dim + 1 :]
    new_shape = shape_before + (-1, size) + shape_after
    x_reshaped = x.view(new_shape)
    return x_reshaped.mean(dim + 1)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ori_dtype = t.dtype
        embed_dtype = torch.float64
        t = t.to(embed_dtype)
        if t.ndim == 3:
            seq_len = t.shape[seq_dim]
            freqs = freqs[-seq_len:].to(t)

        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim

        assert rot_dim <= t.shape[-1], (
            f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )

        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1).to(ori_dtype)


class MaxTimeContinuousTimeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_time, period_mode="shortest", device=None):
        super().__init__()
        assert dim % 2 == 0, "RoPE embedding dimension must be even"

        self.dim = dim
        self.max_time = max_time
        self.period_mode = period_mode

        # Set max period = max_time
        if period_mode == "shortest":  # shortest period is max_time
            base = 5
            inv_freq = 2 * math.pi / (max_time * (base ** (torch.arange(0, dim // 2).float() / (dim // 2))))
        elif period_mode == "longest":  # longest period is max_time ** ((dim // 2) / (dim // 2 - 1))
            theta = max_time ** ((dim // 2) / (dim // 2 - 1))
            inv_freq = 2 * math.pi / (theta ** (torch.arange(0, dim // 2).float() / (dim // 2)))
        else:
            raise ValueError(f"Invalid period mode: {period_mode}")
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, time_values: torch.Tensor):
        """
        time_values: [batch_size, seq_len], in seconds (or any continuous unit)
        Returns:
            cos, sin: [batch_size, seq_len, dim]
        """
        batch_size, seq_len = time_values.shape
        time_values_exp = time_values[:, None, :]  # [batch, 1, seq_len]
        freqs = (self.inv_freq[None, :, None] @ time_values_exp).transpose(1, 2)  # [batch, seq_len, dim//2]
        # emb = torch.cat([freqs, freqs], dim=-1)  # [batch, seq_len, dim]
        # return emb.cos(), emb.sin()
        return freqs

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)


class RotaryEmbedding(Module):
    @beartype
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
        max_time=None,
    ):
        super().__init__()

        self.dim = dim
        self.freqs_for = freqs_for
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.learned_freq = learned_freq
        self.use_xpos = use_xpos
        self.xpos_scale_base = xpos_scale_base
        self.interpolate_factor = interpolate_factor
        self.theta_rescale_factor = theta_rescale_factor
        self.cache_if_possible = cache_if_possible
        self.max_time = max_time

        self.tmp_store("cached_freqs", None)
        self.tmp_store("cached_scales", None)

        # Adjust theta to avoid angle wrapping after large times
        if exists(max_time) and freqs_for == "lang":
            # Make sure highest frequency completes 1 full rotation over max time
            # theta = base of exponent: higher theta → lower frequency range
            # max_time * (1/theta^(0)) = 2pi  =>  theta = max_time / (2pi)
            theta = max_time / (2 * pi)

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.theta = theta

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.tmp_store("dummy", torch.tensor(0))

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos
        if not use_xpos:
            self.tmp_store("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store("scale", scale)

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos, (
            "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"
        )

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        freqs = self.forward(
            self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset), seq_len=seq_len, offset=offset
        )

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, offset=k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, offset=offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    @beartype
    def get_scale(self, t: Tensor, seq_len: int | None = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len)

        if should_cache and exists(self.cached_scales) and (seq_len + offset) <= self.cached_scales.shape[0]:
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.tmp_store("cached_scales", scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible and not self.learned_freq and exists(seq_len) and self.freqs_for != "pixel"
        )

        if should_cache and exists(self.cached_freqs) and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        # Scale time to keep t * freq <= 2pi
        if hasattr(self, "max_time") and self.max_time is not None:
            t = t / self.max_time * (2 * pi)

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if should_cache:
            self.tmp_store("cached_freqs", freqs.detach())

        return freqs


def _move_rotary_module_to_device(module: nn.Module, device: torch.device) -> nn.Module:
    try:
        return module.to(device)
    except NotImplementedError as exc:
        if "meta tensor" not in str(exc).lower():
            raise

    if isinstance(module, MaxTimeContinuousTimeRotaryEmbedding):
        return MaxTimeContinuousTimeRotaryEmbedding(
            dim=module.dim,
            max_time=module.max_time,
            period_mode=module.period_mode,
        ).to(device)

    if isinstance(module, RotaryEmbedding):
        return RotaryEmbedding(
            dim=module.dim,
            freqs_for=module.freqs_for,
            theta=module.theta,
            max_freq=module.max_freq,
            num_freqs=module.num_freqs,
            learned_freq=module.learned_freq,
            use_xpos=module.use_xpos,
            xpos_scale_base=module.xpos_scale_base,
            interpolate_factor=module.interpolate_factor,
            theta_rescale_factor=1.0,
            seq_before_head_dim=module.seq_before_head_dim,
            cache_if_possible=module.cache_if_possible,
            max_time=module.max_time,
        ).to(device)

    raise TypeError(f"Unsupported rotary module type for meta materialization: {type(module)}")


class BaseEncoder(nn.Module):
    def __init__(self, parent: nn.Module) -> None:
        super().__init__()
        self._parent = [parent]

    @property
    def parent(self) -> nn.Module:
        return self._parent[0]

    def embed_tokens(self, tokens: str | None) -> torch.Tensor | None:
        return self.parent.embed_text_tokens(tokens)


class BasicImageEncoder(BaseEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: str | None = None,
        end_tokens: str | None = "\n",
    ) -> None:
        super().__init__(parent)
        end_tokens = None if end_tokens == "None" else end_tokens
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: torch.Tensor | None,
        end_token_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def forward(self, images: list[torch.Tensor], config: dict[str, Any], mm_info: dict) -> list[torch.Tensor]:
        images = torch.stack(images, dim=0)
        features = self.parent.encode_images(images, block_sizes=config.get("block_sizes"))
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        return [process_features(f) for f in features]


class BasicVideoEncoder(BaseEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: str | None = None,
        end_tokens: str | None = "\n",
    ) -> None:
        super().__init__(parent)
        end_tokens = None if end_tokens == "None" else end_tokens
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: torch.Tensor | None,
        end_token_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if start_token_embeds is not None:
            start_embeds = torch.stack([start_token_embeds] * features.shape[0], dim=0)
            features = torch.cat([start_embeds, features], dim=1)
        if end_token_embeds is not None:
            end_embeds = torch.stack([end_token_embeds] * features.shape[0], dim=0)
            features = torch.cat([features, end_embeds], dim=1)
        return features.flatten(0, 1)

    def forward(self, videos: list[torch.Tensor], config: dict[str, Any], mm_info: dict) -> list[torch.Tensor]:
        _ = mm_info
        num_frames = [video.shape[0] for video in videos]
        images = torch.cat(videos, dim=0)
        features = self.parent.encode_images(images)
        features = torch.split(features, num_frames)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        return [process_features(f) for f in features]


class BasicSoundEncoder(BaseEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: str | None = None,
        end_tokens: str | None = "\n",
        embed_time="True",
        trope_theta=50000,
        trope_dim=128,
        max_time=None,
        time_embed_type="pixel",
        period_fix=False,
    ) -> None:
        super().__init__(parent)
        end_tokens = None if end_tokens == "None" else end_tokens
        if embed_time == "True":
            embed_time = True
        elif embed_time == "False":
            embed_time = False
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

        if embed_time is False:
            self.embed_time = False
        else:
            self.embed_time = True
            self.time_embed_type = time_embed_type

            period_mode = None
            if isinstance(period_fix, str):
                if period_fix == "shortest":
                    period_fix = "MTCT"
                    period_mode = "shortest"
                elif period_fix == "longest":
                    period_fix = "MTCT"
                    period_mode = "longest"

            self.period_fix = period_fix
            self.max_time = max_time

            if period_fix == "MTCT":
                if period_mode is None:
                    self.pos_emb = MaxTimeContinuousTimeRotaryEmbedding(
                        dim=trope_dim,
                        max_time=max_time,
                    )
                else:
                    self.pos_emb = MaxTimeContinuousTimeRotaryEmbedding(
                        dim=trope_dim,
                        max_time=max_time,
                        period_mode=period_mode,
                    )

            elif time_embed_type in ["pixel", "lang"]:
                if trope_dim is None and max_time is None:
                    raise ValueError("trope_dim or max_time is required when embed_time is True")
                self.pos_emb = RotaryEmbedding(
                    dim=trope_dim,
                    freqs_for=time_embed_type,
                    max_freq=256,
                    max_time=max_time,
                )
            elif time_embed_type == "learned_embed":
                self.time_embed = parent.sound_mm_projector.time_embed
            else:
                raise ValueError(f"Invalid time_embed_type: {time_embed_type}")

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: torch.Tensor | None,
        end_token_embeds: torch.Tensor | None,
        times: torch.Tensor | None = None,
        time_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = features.to(self.parent.device)
        device = features.device

        if self.embed_time:
            device = features.device

            # Handle different embedding types
            if self.time_embed_type in ["pixel", "lang"]:
                times = times.unsqueeze(0)
                new_times = times
                self.pos_emb = _move_rotary_module_to_device(self.pos_emb, device)
                pos_emb = self.pos_emb
                if self.period_fix == "True":
                    if self.max_time is not None:
                        angle = new_times.to(device) / self.max_time * 2 * np.pi
                    else:
                        angle = new_times.to(device)
                elif self.period_fix == "MTCT":
                    freqs = self.pos_emb(new_times.float())
                    freqs = freqs.squeeze(0)
                    features = apply_rotary_emb(freqs, features)
                else:
                    angle = (-new_times * 2 * np.pi).to(device)

                if self.period_fix != "MTCT":
                    freqs = pos_emb.get_axial_freqs(new_times.shape[0], features.shape[-2]).to(device)
                    angle_expanded = angle.unsqueeze(2)
                    angle_expanded = angle_expanded.expand(new_times.shape[0], features.shape[-2], freqs.shape[-1])
                    freqs = freqs * angle_expanded
                    freqs = freqs.squeeze(0)
                    # ori_dtype = features.dtype
                    # embed_dtype = torch.float32
                    # features = features.to(embed_dtype)
                    features = apply_rotary_emb(freqs, features)
                    # features = features.to(ori_dtype)
            elif self.time_embed_type == "learned_embed":  # Learned embedding
                # Add time embeddings to features
                features = features + time_embed
            else:
                raise ValueError(f"Invalid time_embed_type: {self.time_embed_type}")

        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def forward(self, sounds: list[torch.Tensor], config: dict[str, Any], mm_info: dict) -> list[torch.Tensor]:
        # sounds = torch.stack(sounds, dim=0)
        features = self.parent.encode_sound(sounds, mm_info=mm_info)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )

        if self.embed_time:
            new_features = []
            device = features[0].device
            fea_count = len(features)
            aud_idx = 0
            bs = len(mm_info["audio_info"])

            if (
                self.time_embed_type == "learned_embed"
            ):  # Learned embedding, we need to first collect all times and only do time embedding once
                times_list = []
                for i in range(bs):
                    _audio_info = mm_info["audio_info"][i]
                    if _audio_info is not None:
                        for j in range(len(_audio_info)):
                            _feature = features[aud_idx]
                            if _audio_info[j] == "dummy":
                                times = torch.zeros(_feature.shape[0], device=device, dtype=_feature.dtype)
                            else:
                                audio_chunk_length = _audio_info[j]["new_audio_chunk_length"]
                                sec_per_embed = audio_chunk_length / _feature.shape[0]
                                audio_start_sec = _audio_info[j]["audio_start_sec"]
                                times = [
                                    audio_start_sec + i * sec_per_embed + sec_per_embed / 2
                                    for i in range(_feature.shape[0])
                                ]
                                times = torch.tensor(times).to(device)
                            times_list.append(times)
                            aud_idx += 1

                times = torch.stack(times_list, dim=0)
                time_embeds = self.time_embed(times, dtype=features[0].dtype)

            aud_idx = 0
            for i in range(bs):
                _audio_info = mm_info["audio_info"][i]
                if _audio_info is not None:
                    for j in range(len(_audio_info)):
                        try:
                            _feature = features[aud_idx]
                        except Exception as e:
                            print(
                                f"Error: {e}. Length of features: {len(features)}. Length of _audio_info: {len(_audio_info)}. Length of _feature: {_feature.shape[0]}"
                            )
                            raise e
                        if _audio_info[j] == "dummy":
                            times = torch.zeros(_feature.shape[0], device=device, dtype=_feature.dtype)
                        else:
                            audio_chunk_length = _audio_info[j]["new_audio_chunk_length"]
                            sec_per_embed = audio_chunk_length / _feature.shape[0]
                            audio_start_sec = _audio_info[j]["audio_start_sec"]
                            times = [
                                audio_start_sec + i * sec_per_embed + sec_per_embed / 2
                                for i in range(_feature.shape[0])
                            ]
                            times = torch.tensor(times).to(device)
                        if self.time_embed_type == "learned_embed":
                            _feature = process_features(_feature, time_embed=time_embeds[aud_idx])
                        else:
                            _feature = process_features(_feature, times=times)
                        new_features.append(_feature)
                        aud_idx += 1

            assert aud_idx == fea_count, f"aud_idx: {aud_idx}, fea_count: {fea_count}"
            features = new_features
        else:
            features = [process_features(f) for f in features]
        return features

        # return [process_features(f) for f in feature


class TSPVideoEncoder(BasicVideoEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        pool_sizes: list[tuple[int, int, int]],
        start_tokens: str | None = None,
        end_tokens: str | None = "\n",
        sep_tokens: str | None = None,
        embed_time: str = "False",
        trope_theta=50000,
        trope_dim=128,
        max_time=None,
        time_embed_type="pixel",
        period_fix=False,
    ) -> None:
        super().__init__(parent, start_tokens=start_tokens, end_tokens=end_tokens)
        self.pool_sizes = pool_sizes
        self.sep_tokens = sep_tokens

        if embed_time == "False":
            self.embed_time = False
        else:
            self.embed_time = True
            self.time_embed_type = time_embed_type

            period_mode = None
            if isinstance(period_fix, str):
                if period_fix == "shortest":
                    period_fix = "MTCT"
                    period_mode = "shortest"
                elif period_fix == "longest":
                    period_fix = "MTCT"
                    period_mode = "longest"

            self.period_fix = period_fix
            self.max_time = max_time

            if period_fix == "MTCT":
                if period_mode is None:
                    self.pos_emb = MaxTimeContinuousTimeRotaryEmbedding(
                        dim=trope_dim,
                        max_time=max_time,
                    )
                else:
                    self.pos_emb = MaxTimeContinuousTimeRotaryEmbedding(
                        dim=trope_dim,
                        max_time=max_time,
                        period_mode=period_mode,
                    )

            elif time_embed_type in ["pixel", "lang"]:
                if trope_dim is None and max_time is None:
                    raise ValueError("trope_dim or max_time is required when embed_time is True")

                if time_embed_type == "lang":
                    self.pos_emb = RotaryEmbedding(
                        dim=trope_dim,
                        freqs_for="lang",
                        theta=trope_theta,
                        max_time=max_time,
                    )
                elif time_embed_type == "pixel":
                    self.pos_emb = RotaryEmbedding(dim=trope_dim, freqs_for=time_embed_type, max_freq=256)
            elif time_embed_type == "learned_embed":
                self.time_embed = parent.mm_projector.time_embed
            else:
                raise ValueError(f"Invalid time_embed_type: {time_embed_type}")

    def _process_features(
        self,
        inputs: torch.Tensor,
        start_token_embeds: torch.Tensor | None,
        end_token_embeds: torch.Tensor | None,
        sep_token_embeds: torch.Tensor | None,
        times: torch.Tensor | None = None,
        time_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nt, ns = inputs.shape[:2]
        nl = int(ns**0.5)
        outputs = []
        for pool_size in self.pool_sizes:
            features = inputs.view(nt, nl, nl, -1)
            for dim, p in enumerate(pool_size):
                try:
                    features = pool(features, p, dim=dim)
                except Exception as e:
                    print(f"Error: Pooling failed: {e}")
                    print(
                        f"inputs.shape: {inputs.shape}, features.shape: {features.shape}, pool_size: {p}, dim: {dim}"
                    )
                    raise e
            features = features.flatten(1, 2)

            if self.embed_time:
                device = features.device
                if self.time_embed_type in ["pixel", "lang"]:
                    # consider the pooling in self.pool_sizes
                    temporal_pool_size = pool_size[0]
                    if temporal_pool_size != 1:
                        if len(times) % temporal_pool_size != 0:
                            # pad
                            print(
                                f"Warning: length of times: {len(times)} is not a multiple of temporal_pool_size: {temporal_pool_size}"
                            )
                            remainder = len(times) % temporal_pool_size
                            pad_len = temporal_pool_size - remainder
                            last_window_mean_times = times[-remainder:].mean()
                            times = torch.cat([times, torch.ones(pad_len).to(times.device) * last_window_mean_times])
                        new_times = pool(times, temporal_pool_size, 0)
                    else:
                        new_times = times

                    self.pos_emb = _move_rotary_module_to_device(self.pos_emb, device)
                    pos_emb = self.pos_emb
                    if self.period_fix == "True":
                        if self.max_time is not None:
                            angle = new_times.to(device) / self.max_time * 2 * np.pi
                        else:
                            angle = new_times.to(device)
                    elif self.period_fix == "MTCT":
                        if new_times.ndim == 1:
                            new_times = new_times.unsqueeze(0)
                        freqs = self.pos_emb(new_times.float())
                        freqs = freqs.squeeze(0)
                        freqs = freqs.unsqueeze(1)
                        features = apply_rotary_emb(freqs, features, seq_dim=0)
                    else:
                        angle = (-new_times * 2 * np.pi).to(device)

                    if self.period_fix != "MTCT":
                        freqs = pos_emb.get_axial_freqs(new_times.shape[0], features.shape[-2]).to(device)
                        angle_expanded = angle.unsqueeze(1).unsqueeze(2)
                        angle_expanded = angle_expanded.expand(new_times.shape[0], features.shape[-2], freqs.shape[-1])
                        freqs = freqs * angle_expanded
                        # ori_dtype = features.dtype
                        # embed_dtype = torch.float32
                        # features = features.to(embed_dtype)
                        features = apply_rotary_emb(freqs, features)
                        # features = features.to(ori_dtype)
                elif self.time_embed_type == "learned_embed":  # Learned embedding
                    # Add time embeddings to features
                    features = features + time_embed
                else:
                    raise ValueError(f"Invalid time_embed_type: {self.time_embed_type}")

            features = super()._process_features(
                features,
                start_token_embeds=start_token_embeds,
                end_token_embeds=end_token_embeds,
            )
            if sep_token_embeds is not None:
                features = torch.cat([features, sep_token_embeds], dim=0)
            outputs.append(features)
        return torch.cat(outputs, dim=0)

    def forward(self, videos: list[torch.Tensor], config: dict[str, Any], mm_info: dict) -> list[torch.Tensor]:
        num_frames = [_.shape[0] for _ in videos]

        features = self.parent.encode_video(videos, mm_info=mm_info, num_frames=num_frames)
        features = torch.split(features, num_frames)

        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
            sep_token_embeds=self.embed_tokens(self.sep_tokens),
        )

        if self.embed_time:
            bs = len(mm_info["video_info"])
            vid_idx = 0
            device = features[0].device

            if self.time_embed_type == "learned_embed":
                # Learned embedding, we need to first collect all times from all videos and only do time embedding once
                times_list = []
                for i in range(bs):
                    _video_info = mm_info["video_info"][i]
                    if _video_info is not None:
                        for j in range(len(_video_info)):
                            _feature = features[vid_idx]
                            if _video_info[j] == "dummy":
                                times = torch.zeros(_feature.shape[0], device=device, dtype=_feature.dtype)
                            else:
                                times = _video_info[j]["video_frame_times"]
                                times = torch.tensor(times).to(device)

                            for pool_size in self.pool_sizes:
                                temporal_pool_size = pool_size[0]
                                if temporal_pool_size != 1:
                                    if len(times) % temporal_pool_size != 0:
                                        # pad
                                        print(
                                            f"Warning: length of times: {len(times)} is not a multiple of temporal_pool_size: {temporal_pool_size}"
                                        )
                                        remainder = len(times) % temporal_pool_size
                                        pad_len = temporal_pool_size - remainder
                                        last_window_mean_times = times[-remainder:].mean()
                                        times = torch.cat(
                                            [times, torch.ones(pad_len).to(times.device) * last_window_mean_times]
                                        )
                                    times = pool(times, temporal_pool_size, 0)

                            times_list.append(times)
                            vid_idx += 1

                # pad the times to the same length
                ori_lens = [len(times) for times in times_list]
                max_len = max(ori_lens)
                for i in range(len(times_list)):
                    if len(times_list[i]) < max_len:
                        times_list[i] = torch.cat(
                            [times_list[i], torch.zeros(max_len - len(times_list[i])).to(times_list[i].device)]
                        )
                times = torch.stack(times_list, dim=0)
                time_embeds = self.time_embed(times, dtype=features[0].dtype)

                # remove the padding for each embed
                new_time_embeds = []
                for i in range(len(times_list)):
                    new_time_embeds.append(
                        time_embeds[i][: ori_lens[i]].unsqueeze(1).expand(-1, features[0].shape[1], -1)
                    )

                # add dummy embed to the first embed
                new_time_embeds[0] = new_time_embeds[0] + 0 * time_embeds.mean()

            new_features = []
            fea_count = len(features)
            vid_idx = 0
            for i in range(bs):
                _video_info = mm_info["video_info"][i]
                if _video_info is not None:
                    for j in range(len(_video_info)):
                        _feature = features[vid_idx]
                        if _video_info[j] == "dummy":
                            times = torch.zeros(_feature.shape[0], device=device, dtype=_feature.dtype)
                        else:
                            times = _video_info[j]["video_frame_times"]
                            times = torch.tensor(times).to(device)
                        if self.time_embed_type == "learned_embed":
                            _feature = process_features(_feature, time_embed=new_time_embeds[vid_idx])
                        else:
                            _feature = process_features(_feature, times=times)
                        new_features.append(_feature)
                        vid_idx += 1

            assert vid_idx == fea_count, f"vid_idx: {vid_idx}, fea_count: {fea_count}"
            features = new_features
        else:
            features = [process_features(f) for f in features]
        return features
