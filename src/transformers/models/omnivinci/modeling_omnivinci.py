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

import copy
import math
import warnings
from collections import defaultdict, deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    SiglipImageProcessor,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.perceiver.modeling_perceiver import space_to_depth
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from .configuration_omnivinci import IGNORE_INDEX, OmniVinciConfig
from .media_encoder import BasicImageEncoder, BasicSoundEncoder, TSPVideoEncoder


def context_length_extension(config):
    """Extend context length using RoPE scaling if needed."""
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    model_max_length = getattr(config, "model_max_length", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        print(f"Scaling RoPE from {orig_ctx_len} to {model_max_length}")
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    return config


class MultimodalProjector(nn.Module):
    """Multimodal projector for mapping vision features to LLM space."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.downsample_rate = 2
        self.layers = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(config.mm_hidden_size * 4),
            nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
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


class SoundMultimodalProjector(nn.Module):
    """Sound multimodal projector for mapping audio features to LLM space."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.sound_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


class Qwen2AudioTower(nn.Module):
    def __init__(self, sound_tower_cfg: dict[str, Any], config: PretrainedConfig):
        super().__init__()
        audio_cfg = Qwen2AudioEncoderConfig(**{k: v for k, v in sound_tower_cfg.items() if k != "model_type"})
        audio_cfg._attn_implementation = config._attn_implementation
        self.audio_tower = Qwen2AudioEncoder(audio_cfg)

        self.audio_chunk_unit_duration = 30
        self.audio_chunk_unit_length = 3000

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def config(self):
        return self.audio_tower.config

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    def forward(self, sounds):
        if isinstance(sounds, list):
            sound_features = []
            audio_output_lengths = []
            for sound in sounds:
                if hasattr(sound, "input_features") or (isinstance(sound, dict) and "input_features" in sound):
                    sound = sound["input_features"]
                sound = sound.to(device=self.device, dtype=self.dtype)

                sound_feature = self.forward_audio_tower_batch(sound)
                sound_feature = sound_feature.to(sound.dtype)
                sound_features.append(sound_feature)
                audio_output_lengths.append(sound_feature.shape[1])
            if len(sound_features) > 0:
                sound_features = torch.cat(sound_features, dim=1).squeeze(0)
        else:
            raise NotImplementedError("Not implemented for this encoder")

        return sound_features, audio_output_lengths

    def forward_audio_tower_batch(self, inp):
        """
        Process long audio input by splitting into fixed-size chunks (30 seconds),
        padding if needed, batching them together, and processing through the audio tower.

        Args:
            inp: Tensor of shape (batch_size, n_mels, seq_len)

        Returns:
            Tensor of shape (batch_size, num_chunks * chunk_seq_len, hidden_size)
        """
        batch_size, n_mels, seq_len = inp.shape
        chunk_length = self.audio_chunk_unit_length
        num_chunks = (seq_len + chunk_length - 1) // chunk_length  # Ceiling division

        padded_chunks = []

        for i in range(num_chunks):
            start_idx = i * chunk_length
            end_idx = min(start_idx + chunk_length, seq_len)

            # Extract and pad chunk if necessary
            chunk = inp[:, :, start_idx:end_idx]
            if chunk.shape[2] < chunk_length:
                pad_len = chunk_length - chunk.shape[2]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len), mode="constant", value=0)

            padded_chunks.append(chunk)

        # Stack chunks along batch dimension
        all_chunks = torch.cat(padded_chunks, dim=0).reshape(batch_size * num_chunks, n_mels, chunk_length)

        # Forward pass through the audio tower
        chunk_outputs = self.audio_tower(all_chunks)
        hidden_states = chunk_outputs.last_hidden_state

        # Reshape back to (batch_size, num_chunks * seq_len', hidden_size)
        _, chunk_seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, num_chunks * chunk_seq_len, hidden_size)

        return hidden_states


class SiglipVisionTowerDynamicS2(nn.Module):
    def __init__(self, vision_tower_cfg: dict[str, Any], config: PretrainedConfig) -> None:
        super().__init__()

        self.select_layer = getattr(config, "mm_vision_select_layer", -2)
        self.select_feature = getattr(config, "mm_vision_select_feature", "patch")
        self.scales = sorted(map(int, config.s2_scales.split(",")))
        self.max_split_size = config.s2_max_split_size
        self.resize_output_to_scale_idx = getattr(config, "s2_resize_output_to_scale_idx", 0)

        vision_cfg = SiglipVisionConfig(**{k: v for k, v in vision_tower_cfg.items() if k != "model_type"})
        vision_cfg._attn_implementation = config._attn_implementation
        self.vision_tower = SiglipVisionModel(vision_cfg)

        self.image_processor = SiglipImageProcessor()
        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[0]

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        if isinstance(images, list):
            raise ValueError("VisionTowerDynamicS2 expects tensor input, not list.")
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
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


class VILAPretrainedModel(PreTrainedModel):
    config_class = OmniVinciConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["Qwen2DecoderLayer", "SiglipEncoderLayer"]

    def __init__(self, config: OmniVinciConfig, *args, **kwargs):
        _ = (args, kwargs)
        super().__init__(config)
        self.config = config

    def _init_omnivinci_components(self, *args, **kwargs):
        _ = args
        config = self.config
        llm_spec = config.llm_cfg
        vision_tower_spec = config.vision_tower_cfg
        sound_tower_spec = config.sound_tower_cfg

        self.mm_projector = MultimodalProjector(config)

        if not getattr(config, "dynamic_s2", False):
            raise NotImplementedError("Current OmniVinci checkpoint requires `dynamic_s2=True`.")
        self.vision_tower = SiglipVisionTowerDynamicS2(vision_tower_spec, config)
        config.mm_hidden_size = self.vision_tower.hidden_size

        self.sound_tower = Qwen2AudioTower(sound_tower_spec, config)
        config.sound_hidden_size = 1280
        self.sound_mm_projector = SoundMultimodalProjector(config)

        llm_cfg = Qwen2Config(**{k: v for k, v in llm_spec.items() if k != "model_type"})
        llm_cfg._attn_implementation = config._attn_implementation
        model_max_length = getattr(config, "model_max_length", None)
        if model_max_length is not None:
            llm_cfg.model_max_length = model_max_length
            context_length_extension(llm_cfg)

        self.llm = Qwen2ForCausalLM(llm_cfg)
        config.hidden_size = self.llm.config.hidden_size

        self.vocab_size = self.llm.config.vocab_size
        self.update_vocab_size = lambda: setattr(self, "vocab_size", self.llm.config.vocab_size)

        image_encoder_config = dict(self.config.image_encoder)
        video_encoder_config = dict(self.config.video_encoder)
        sound_encoder_config = dict(self.config.sound_encoder)
        image_encoder_config.pop("_target_", None)
        video_encoder_config.pop("_target_", None)
        sound_encoder_config.pop("_target_", None)

        self.encoders = {
            "image": BasicImageEncoder(parent=self, **image_encoder_config),
            "video": TSPVideoEncoder(parent=self, **video_encoder_config),
            "sound": BasicSoundEncoder(parent=self, **sound_encoder_config),
        }

        self.post_config()

    @property
    def llm_model_embed_tokens(self):
        if self.llm is None:
            raise RuntimeError("LLM module is not initialized.")
        return self.llm.model.embed_tokens

    def _require_encoder_text_token_ids(self) -> dict[str, list[int]]:
        encoder_text_token_ids = getattr(self.config, "encoder_text_token_ids", None)
        if encoder_text_token_ids is None:
            raise ValueError(
                "Missing `config.encoder_text_token_ids`. Construct inputs with `OmniVinciProcessor` before calling "
                "generation so encoder boundary token ids are populated on the config."
            )
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
            raise ValueError(
                "Missing `config.media_token_ids`. Build inputs with `OmniVinciProcessor` so media token ids are "
                "populated on the config."
            )
        return media_token_ids

    def _get_padding_side(self) -> str:
        return getattr(self.config, "padding_side", "left")

    def _get_model_max_length(self) -> int:
        model_max_length = getattr(self.config, "model_max_length", None)
        if model_max_length is None and getattr(self, "llm", None) is not None:
            model_max_length = getattr(self.llm.config, "model_max_length", None)
        if model_max_length is None:
            model_max_length = 2048
        return int(model_max_length)

    def post_config(self):
        self.training = self.llm.training
        if self.training:
            self.train()
        else:
            self.eval()

        # configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
        # Transformers v5 generation/cache code resolves decoder metadata via config.get_text_config().
        # Expose the loaded LLM config so required fields (e.g. num_hidden_layers) are always available.
        self.config.text_config = self.llm.config
        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config
        if getattr(self.config, "mm_projector_cfg", None) is None:
            self.config.mm_projector_cfg = {"mm_projector_type": "mlp_downsample"}
        if getattr(self.config, "sound_tower_cfg", None) is None and hasattr(self, "sound_tower"):
            self.config.sound_tower_cfg = self.sound_tower.config
        if getattr(self.config, "sound_mm_projector_cfg", None) is None and hasattr(self, "sound_mm_projector"):
            self.config.sound_mm_projector_cfg = {"sound_mm_projector_type": "mlp"}

    def freezed_module_patch(self):
        """
        Huggingface will call model.train() at each training_step. To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        """
        if self.training:
            vision_tower = self.vision_tower
            sound_tower = getattr(self, "sound_tower", None)
            mm_projector = self.mm_projector
            sound_mm_projector = getattr(self, "sound_mm_projector", None)

            if vision_tower and not getattr(self.config, "tune_vision_tower", False):
                vision_tower.eval()
            if sound_tower and not getattr(self.config, "tune_sound_tower", False):
                sound_tower.eval()
            if mm_projector and not getattr(self.config, "tune_mm_projector", False):
                mm_projector.eval()
            if sound_mm_projector and not getattr(self.config, "tune_sound_mm_projector", False):
                sound_mm_projector.eval()


class OmniVinciForCausalLM(VILAPretrainedModel, GenerationMixin):
    def __init__(self, config: OmniVinciConfig, *args, **kwargs):
        super().__init__(config)
        self._init_omnivinci_components(*args, **kwargs)
        self.post_init()

    def merge_features_for_dynamic_s2(self, image_features, block_sizes):
        scales = self.vision_tower.scales
        resize_output_to_scale_idx = self.vision_tower.resize_output_to_scale_idx

        image_features_each_image = []
        new_block_sizes = []
        block_cnt = 0
        for block_size_each_image in block_sizes:
            if block_size_each_image is None:
                cur_features = image_features[block_cnt : block_cnt + 1]
                cur_features = rearrange(cur_features, "1 (h w) c -> 1 c h w", h=int(cur_features.shape[1] ** 0.5))
                cur_features = cur_features.repeat(1, len(scales), 1, 1)
                image_features_each_image.append(cur_features)
                new_block_sizes.append((1, 1))
                block_cnt += 1
            else:
                cur_features_each_scale = []
                for scale in scales[:-1]:
                    num_blocks_this_scale = (scale // scales[0]) ** 2
                    cur_features_each_scale.append(
                        self.merge_chessboard(
                            image_features[block_cnt : block_cnt + num_blocks_this_scale],
                            num_split_h=scale // scales[0],
                            num_split_w=scale // scales[0],
                        )
                    )  # 1 * C * H * W
                    block_cnt += num_blocks_this_scale
                num_blocks_last_scale = block_size_each_image[0] * block_size_each_image[1]
                cur_features_each_scale.append(
                    self.merge_chessboard(
                        image_features[block_cnt : block_cnt + num_blocks_last_scale],
                        num_split_h=block_size_each_image[0],
                        num_split_w=block_size_each_image[1],
                    )
                )  # 1 * C * H * W
                block_cnt += num_blocks_last_scale

                # resize and concat features from different scales
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
        """
        x: b * c * h * w
        out: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
        """
        B, C, H, W = x.shape
        assert H % num_split_h == 0 and W % num_split_w == 0
        h, w = H // num_split_h, W // num_split_w
        x_split = torch.cat(
            [
                x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w]
                for i in range(num_split_h)
                for j in range(num_split_w)
            ],
            dim=0,
        )
        return x_split

    @staticmethod
    def merge_chessboard(x, num_split_h, num_split_w):
        """
        x: b * n * c or b * h * w * c
        out: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        """
        B = x.shape[0]
        if x.dim() == 3:
            N = x.shape[1]
            x = rearrange(x, "b (h w) c -> b c h w", h=int(N**0.5), w=int(N**0.5))

        assert B % (num_split_h * num_split_w) == 0
        b = B // (num_split_h * num_split_w)

        x_merge = torch.cat(
            [
                torch.cat(
                    [x[(i * num_split_w + j) * b : (i * num_split_w + j + 1) * b] for j in range(num_split_w)], dim=-1
                )
                for i in range(num_split_h)
            ],
            dim=-2,
        )

        return x_merge

    def encode_video(
        self,
        inp,
        block_sizes: tuple[int, ...] | None = None,
        mm_info: dict | None = None,
        num_frames: list[int] | None = None,
    ):
        _ = (mm_info, num_frames)
        if not getattr(self.config, "dynamic_s2", False):
            raise NotImplementedError("Current OmniVinci checkpoint requires `dynamic_s2=True`.")

        inp_block_sizes = block_sizes
        if len(inp) > 0:
            images = torch.cat(inp, dim=0)
        else:
            images = []

        if block_sizes is None:
            block_sizes = [None] * len(images)

        if len(images) > 0:
            image_features = self.vision_tower(images)

            image_features, new_block_sizes = self.merge_features_for_dynamic_s2(image_features, block_sizes)

            image_features = [
                self.split_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes)
            ]  # list of B * C * H * W tensors
            image_features = torch.cat(
                [rearrange(x, "b c h w -> b (h w) c") for x in image_features], dim=0
            )  # B * N * C
        else:
            image_features = []

        if inp_block_sizes is None:
            new_block_sizes = [(1, 1)] * len(image_features)
        else:
            raise ValueError(f"inp_block_sizes is not None: {inp_block_sizes}")
        image_features = image_features.to(self.device, self.dtype)
        image_features = self.mm_projector(image_features)
        image_features = list(
            image_features.split([block_size[0] * block_size[1] for block_size in new_block_sizes], dim=0)
        )
        image_features = [
            self.merge_chessboard(x, block_size[0], block_size[1])
            for x, block_size in zip(image_features, new_block_sizes)
        ]  # list of 1 * C * H * W tensors
        image_features = [rearrange(x, "1 c h w -> (h w) c") for x in image_features]  # list of N * C tensors
        if all(feature.shape[0] == image_features[0].shape[0] for feature in image_features):
            image_features = torch.stack(image_features, dim=0)
        return image_features

    def encode_images(
        self,
        images,
        block_sizes: tuple[int, ...] | None = None,
        mm_info: dict | None = None,
        num_frames: list[int] | None = None,
    ):
        _ = (mm_info, num_frames)
        if not getattr(self.config, "dynamic_s2", False):
            raise NotImplementedError("Current OmniVinci checkpoint requires `dynamic_s2=True`.")

        if block_sizes is None:
            block_sizes = [None] * len(images)

        image_features = self.vision_tower(images)

        image_features, new_block_sizes = self.merge_features_for_dynamic_s2(image_features, block_sizes)

        image_features = [
            self.split_chessboard(x, block_size[0], block_size[1])
            for x, block_size in zip(image_features, new_block_sizes)
        ]  # list of B * C * H * W tensors
        image_features = torch.cat([rearrange(x, "b c h w -> b (h w) c") for x in image_features], dim=0)  # B * N * C

        image_features = self.mm_projector(image_features)
        image_features = list(
            image_features.split([block_size[0] * block_size[1] for block_size in new_block_sizes], dim=0)
        )
        image_features = [
            self.merge_chessboard(x, block_size[0], block_size[1])
            for x, block_size in zip(image_features, new_block_sizes)
        ]  # list of 1 * C * H * W tensors
        image_features = [rearrange(x, "1 c h w -> (h w) c") for x in image_features]  # list of N * C tensors
        if all(feature.shape[0] == image_features[0].shape[0] for feature in image_features):
            image_features = torch.stack(image_features, dim=0)
        return image_features

    def encode_sound(self, sounds, mm_info: dict | None = None):
        _ = mm_info
        sound_tower = getattr(self, "sound_tower", None)
        sound_mm_projector = getattr(self, "sound_mm_projector", None)
        if sound_tower is None or sound_mm_projector is None:
            raise ValueError("Sound inputs were provided, but sound modules are not initialized.")

        audio_features, audio_output_lengths = sound_tower(sounds)
        projector_param = next(sound_mm_projector.parameters(), None)
        if projector_param is not None and audio_features.dtype != projector_param.dtype:
            audio_features = audio_features.to(projector_param.dtype)
        audio_features = sound_mm_projector(audio_features)

        if audio_output_lengths is not None:
            # split the batch
            new_audio_features = []
            start = 0
            for length in audio_output_lengths:
                new_audio_features.append(audio_features[start : start + length])
                start += length
            audio_features = new_audio_features

        return audio_features

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

        # Extract text and media embeddings
        text_embeds = self.llm_model_embed_tokens(input_ids)

        mm_info = {}
        if "video_info" in media:
            video_info = media["video_info"]
            del media["video_info"]
            mm_info["video_info"] = video_info
        else:
            video_info = None

        if "audio_info" in media:
            audio_info = media["audio_info"]
            del media["audio_info"]
            mm_info["audio_info"] = audio_info
        else:
            audio_info = None

        if media is not None:
            media_embeds = self.__embed_media_tokens(media, media_config, mm_info)
        else:
            # no media was provided, so we just return an empty dict
            media_embeds = {}

        # Based on segment_aud_indices_list and segment_vis_indices_list, get interleaved vis-aud embeddings for video
        video_sound_embeds_idx = 0
        sep_embed = self.encoders["video"].embed_tokens("\n")
        llm_embed_dtype = self.llm_model_embed_tokens.weight.dtype
        text_embeds = text_embeds.to(llm_embed_dtype)
        sep_embed = sep_embed.to(text_embeds.dtype)

        if video_info is not None and self.config.load_audio_in_video and self.config.interleaved_vis_aud_in_video:
            assert self.encoders["video"].end_tokens is None, (
                "end_tokens must be None for interleaved vis-aud in video"
            )
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

                    # Check bounds for sound embeddings
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
                    _new_video_embed = []
                    for j in range(len(segment_vis_indices_list)):
                        _vis_aud_fea = []
                        if len(segment_vis_indices_list[j]) > 0:
                            _new_frames = [
                                int(np.ceil((_frame + 1) * vis_fea_len_per_frame))
                                for _frame in segment_vis_indices_list[j]
                            ]
                            _vis_fea_end = _new_frames[-1]
                            # Ensure we don't exceed the available features
                            _vis_fea_end = min(_vis_fea_end, media_embeds["video"][video_embeds_idx].shape[0])
                            if (
                                j == len(segment_vis_indices_list) - 1
                                and i == len(video_info) - 1
                                and k == len(video_info[i]) - 1
                                and _vis_fea_end != media_embeds["video"][video_embeds_idx].shape[0]
                            ):
                                print(
                                    f"Warning: The number of last interleaved video features does not match the video feature length. Expected: {media_embeds['video'][video_embeds_idx].shape[0]}, Got: {_vis_fea_end}"
                                )
                                _vis_fea_end = media_embeds["video"][video_embeds_idx].shape[0]
                            _vis_fea = media_embeds["video"][video_embeds_idx][vis_end:_vis_fea_end]
                            vis_end = _vis_fea_end
                            _vis_aud_fea.append(_vis_fea)
                        _vis_aud_fea.append(sep_embed)
                        if len(segment_aud_indices_list[j]) > 0:
                            _new_audio_indices = [
                                int(np.ceil(_fea * aud_fea_len_per_stft_frame)) for _fea in segment_aud_indices_list[j]
                            ]
                            _aud_fea_end = _new_audio_indices[-1]
                            # Ensure we don't exceed the available features
                            _aud_fea_end = min(_aud_fea_end, media_embeds["sound"][video_sound_embeds_idx].shape[0])
                            _aud_fea = media_embeds["sound"][video_sound_embeds_idx][aud_end:_aud_fea_end]
                            _vis_aud_fea.append(_aud_fea)
                            aud_end = _aud_fea_end
                        _vis_aud_fea.append(sep_embed)
                        _new_video_embed.append(torch.cat(_vis_aud_fea, dim=0))
                    video_sound_embeds_idx += 1
                    new_video_embeds.append(torch.cat(_new_video_embed, dim=0))
                    video_embeds_idx += 1

            assert len(new_video_embeds) == len(media_embeds["video"]), (
                "The number of new video embeddings does not match the number of original video embeddings."
            )
            media_embeds["video"] = new_video_embeds
        # Remove padding
        batch_size = labels.shape[0]
        text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]
        # Build inverse mapping from token ID to media name
        media_token_ids = self._require_media_token_ids()
        media_tokens = {token_id: name for name, token_id in media_token_ids.items()}

        # Fuse text and media embeddings
        inputs_m, labels_m = [], []
        sound_embeds_idx = 0
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    name = media_tokens[input_ids[k][pos].item()]
                    if input_ids[k][pos].item() == media_token_ids["sound"]:
                        if self.config.interleaved_vis_aud_in_video:
                            if sound_embeds_idx < video_sound_embeds_idx:
                                media_embeds[name].popleft()
                                sound_embeds_idx += 1
                                pos += 1
                                continue
                        sound_embeds_idx += 1

                    end = pos + 1
                    input = media_embeds[name].popleft()
                    label = torch.full([input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype)
                else:
                    end = pos
                    while end < len(labels[k]) and input_ids[k][end].item() not in media_tokens:
                        end += 1
                    input = text_embeds[k][pos:end]
                    label = labels[k][pos:end]

                inputs_mk.append(input)
                labels_mk.append(label)
                pos = end
            inputs_m.append(torch.cat(inputs_mk, dim=0))
            labels_m.append(torch.cat(labels_mk, dim=0))
        inputs, labels = inputs_m, labels_m

        # Check if all media embeddings are consumed

        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed! Still {len(media_embeds[name])} left.")

        # Truncate sequences to `model_max_length` as media embeddings are inserted
        inputs, labels = self.__truncate_sequence(inputs, labels)

        # Pad sequences to the longest one in the batch
        return self.__batchify_sequence(inputs, labels)

    def __embed_media_tokens(
        self,
        media: dict[str, list[torch.Tensor]],
        media_config: dict[str, dict[str, Any]],
        mm_info,
    ) -> dict[str, list[torch.Tensor]]:
        embeds = defaultdict(deque)

        for name in media:
            _encoder = self.encoders[name]

            if name == "sound":
                sound_media = media.get(name, [])
                if len(sound_media) == 0:
                    continue
                if not all(
                    hasattr(sound, "input_features") or (isinstance(sound, dict) and "input_features" in sound)
                    for sound in sound_media
                ):
                    raise ValueError(
                        "Expected pre-extracted sound features in `media['sound']`. "
                        "Run audio preprocessing through `OmniVinciProcessor`."
                    )

            if len(media[name]) > 0:
                embeds[name] = deque(_encoder(media[name], media_config[name], mm_info))
        return embeds

    def __truncate_sequence(
        self, inputs: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_max_length = self._get_model_max_length()
        if self.training and any(len(input) > model_max_length for input in inputs):
            warnings.warn(f"Truncating sequences to `model_max_length` ({model_max_length}).")
            inputs = [input[:model_max_length] for input in inputs]
            labels = [label[:model_max_length] for label in labels]
        return inputs, labels

    def __batchify_sequence(
        self, inputs: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros((size_pk, hidden_size), dtype=inputs[k].dtype, device=device)
            labels_pk = torch.full((size_pk,), IGNORE_INDEX, dtype=labels[k].dtype, device=device)
            if self._get_padding_side() == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                labels[k] = labels[k].to(device)
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    def repack_multimodal_data(self, inputs_embeds, attention_mask, position_ids, labels):
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        seqlens = [attention_mask[k].sum().item() for k in range(batch_size)]

        # Pack all sequences together
        inputs_embeds_p = [inputs_embeds[k][attention_mask[k]] for k in range(batch_size)]
        attention_mask_p = [torch.ones(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        position_ids_p = [torch.arange(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        labels_p = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Add one dummy token at the end of the packed sequence to ensure that `_get_unpacked_data` will be called
        inputs_embeds_p.append(torch.zeros(1, inputs_embeds.shape[-1], dtype=inputs_embeds.dtype, device=device))
        attention_mask_p.append(torch.tensor([0], dtype=torch.int, device=device))
        position_ids_p.append(torch.tensor([0], dtype=torch.int, device=device))
        labels_p.append(torch.tensor([IGNORE_INDEX], dtype=torch.int, device=device))

        # Mask the first token of each sequence to avoid contamination
        for label in labels_p:
            label[0] = IGNORE_INDEX

        # Batch the data
        inputs_embeds_p = torch.cat(inputs_embeds_p, dim=0).unsqueeze(0)
        attention_mask_p = torch.cat(attention_mask_p, dim=0).unsqueeze(0)
        position_ids_p = torch.cat(position_ids_p, dim=0).unsqueeze(0)
        labels_p = torch.cat(labels_p, dim=0).unsqueeze(0)

        if hasattr(
            self, "pad_to_multiple_of"
        ):  # related to quantization, please refer to ModelArguments for more information.
            assert len(labels_p.shape) == 2
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
                    (
                        attention_mask_p,
                        torch.zeros((batch_size, difference), dtype=torch.bool).to(attention_mask_p),
                    ),
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
        self.freezed_module_patch()

        if media_config is None:
            media_config = defaultdict(dict)

        if inputs_embeds is None:
            # During cached decoding steps, `media` is intentionally dropped and only the
            # newest text token is forwarded. In that case, skip multimodal embedding.
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
            (inputs_embeds, attention_mask, position_ids, labels) = self.repack_multimodal_data(
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
        is_first_step = past_key_values is None or (cache_position is not None and cache_position[0] == 0)

        # Build multimodal embeddings before delegating, so token/media alignment is preserved.
        if is_first_step and inputs_embeds is None and media is not None:
            if media_config is None:
                media_config = defaultdict(dict)
            inputs_embeds, _, attention_mask = self._embed(input_ids, media, media_config, None, attention_mask)

        # Delegate cache/input slicing details to the underlying LLM implementation.
        model_inputs = self.llm.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )

        if is_first_step:
            if inputs_embeds is not None:
                model_inputs["inputs_embeds"] = inputs_embeds
                model_inputs["attention_mask"] = attention_mask
                model_inputs["input_ids"] = None

        model_inputs["media"] = None
        model_inputs["media_config"] = None
        return model_inputs
