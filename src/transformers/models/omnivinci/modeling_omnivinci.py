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
import json
import math
import os
import os.path
import os.path as osp
import shutil
import warnings
from collections import OrderedDict, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from einops import rearrange

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Qwen2AudioEncoder,
    SiglipImageProcessor,
    WhisperFeatureExtractor,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.siglip import SiglipVisionModel
from transformers.utils.hub import has_file

from .configuration_omnivinci import IGNORE_INDEX, MEDIA_TOKENS, OmniVinciConfig
from .media_encoder import BasicImageEncoder, BasicSoundEncoder, CacheFeatures, TSPVideoEncoder
from .processing_omnivinci import infer_stop_tokens


def has_tokenizer(repo_id_or_path: str) -> bool:
    """Check if a tokenizer exists at the given path or repository."""
    try:
        return has_file(repo_id_or_path, "tokenizer_config.json")
    except (EnvironmentError, ValueError):
        return False


def context_length_extension(config):
    """Extend context length using RoPE scaling if needed."""
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    model_max_length = getattr(config, "model_max_length", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        print(f"Scaling RoPE from {orig_ctx_len} to {model_max_length}")
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    return config


def soft_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    soft_tokens: Optional[List[int]] = None,
    std: float = 1.0,
) -> torch.Tensor:
    """Fallback soft CE helper; preserves training path without affecting inference."""
    _ = (soft_tokens, std)
    if labels is None:
        return logits.new_zeros(())

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def _resolve_component_path(config: OmniVinciConfig, key: str) -> Optional[str]:
    value = getattr(config, key, None)
    if value in (None, "", {}):
        return None

    root_path = getattr(config, "_name_or_path", None) or getattr(config, "resume_path", None)
    if isinstance(value, (dict, PretrainedConfig)):
        if not root_path:
            raise ValueError(f"Cannot resolve '{key}': config root path is missing.")
        return os.path.join(root_path, key[:-4])
    if isinstance(value, str):
        return value

    raise TypeError(f"Unsupported config type for '{key}': {type(value)}")


def _get_attn_implementation(config: PretrainedConfig, default: str = "sdpa") -> str:
    attn_impl = getattr(config, "_attn_implementation", None)
    if not attn_impl:
        attn_impl = getattr(config, "_attn_implementation_internal", None)
    if attn_impl == "flash_attention_2":
        return default
    return attn_impl or default


def build_llm_and_tokenizer(
    model_name_or_path: str,
    config: PretrainedConfig,
    attn_implementation=None,
    model_max_length=None,
    *args,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Build language model and tokenizer from pretrained checkpoint."""
    llm_cfg = AutoConfig.from_pretrained(model_name_or_path)
    if attn_implementation is None:
        attn_implementation = _get_attn_implementation(config)
    llm_cfg._attn_implementation = attn_implementation
    llm_cfg.model_max_length = model_max_length
    if model_max_length is not None:
        context_length_extension(llm_cfg)

    if isinstance(config.model_dtype, str):
        model_dtype = eval(config.model_dtype)
    else:
        model_dtype = config.model_dtype

    llm = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=llm_cfg, torch_dtype=model_dtype, *args, **kwargs
    )
    print(f"Loaded model from {model_name_or_path} with dtype {model_dtype}")

    llm_path = model_name_or_path
    if not has_tokenizer(llm_path):
        llm_path = osp.join(llm_path, "llm")
    if not has_tokenizer(llm_path):
        raise ValueError(f"Cannot find tokenizer in {llm_path}.")

    tokenizer = AutoTokenizer.from_pretrained(llm_path, padding_side="right", use_fast=True, legacy=False)
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length

    if getattr(config, "chat_template", None) is not None:
        print(f"Using chat template: {config.chat_template}")
        fpath = os.path.join(os.path.dirname(__file__), "chat_templates", f"{config.chat_template}.jinja")
        if not os.path.exists(fpath):
            fpath = os.path.join(os.path.dirname(model_name_or_path), f"{config.chat_template}.jinja")
        with open(fpath) as fd:
            chat_template = fd.read()
        tokenizer.chat_template = chat_template.replace("    ", "").replace("\n", "")

    tokenizer.stop_tokens = infer_stop_tokens(tokenizer)
    tokenizer.stop_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.stop_tokens)

    tokenizer.media_tokens = MEDIA_TOKENS
    tokenizer.media_token_ids = {}
    for name, token in MEDIA_TOKENS.items():
        if config.sound_tower_cfg is None and name == "sound":
            continue
        tokenizer.add_tokens([token], special_tokens=True)
        tokenizer.media_token_ids[name] = tokenizer.convert_tokens_to_ids(token)
        tokenizer.media_tokens[name] = token

    config.hidden_size = llm.config.hidden_size
    return llm, tokenizer


class DownSampleBlock(nn.Module):
    """Downsample 2D feature maps by rearranging into 2x2 blocks."""

    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.contiguous()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class MultimodalProjectorConfig(PretrainedConfig):
    """Configuration for vision-to-language projector."""

    model_type = "v2l_projector"

    def __init__(self, mm_projector_type: str = None, **kwargs):
        super().__init__(**kwargs)
        self.mm_projector_type = mm_projector_type


class MultimodalProjector(PreTrainedModel):
    """Multimodal projector for mapping vision features to LLM space."""

    config_class = MultimodalProjectorConfig

    def __init__(self, mm_projector_cfg: MultimodalProjectorConfig, config: PretrainedConfig):
        super().__init__(mm_projector_cfg)
        mm_projector_type = mm_projector_cfg.mm_projector_type or "mlp_downsample"
        if mm_projector_type != "mlp_downsample":
            raise ValueError(
                f"Unsupported mm_projector_type '{mm_projector_type}'. "
                "Current OmniVinci checkpoint requires 'mlp_downsample'."
            )
        self.downsample_rate = 2
        self.layers = nn.Sequential(
            DownSampleBlock(),
            nn.LayerNorm(config.mm_hidden_size * 4),
            nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.post_init()

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


class SoundMultimodalProjectorConfig(PretrainedConfig):
    """Configuration for sound multimodal projector."""

    model_type = "sound_mm_projector"

    def __init__(self, sound_mm_projector_type: str = None, **kwargs):
        super().__init__(**kwargs)
        self.sound_mm_projector_type = sound_mm_projector_type


class SoundMultimodalProjector(PreTrainedModel):
    """Sound multimodal projector for mapping audio features to LLM space."""

    config_class = SoundMultimodalProjectorConfig

    def __init__(self, sound_mm_projector_cfg: SoundMultimodalProjectorConfig, config: PretrainedConfig):
        super().__init__(sound_mm_projector_cfg)
        if hasattr(config, "sound_mm_projector"):
            sound_mm_projector_type = config.sound_mm_projector
        else:
            sound_mm_projector_type = sound_mm_projector_cfg.sound_mm_projector_type
        self.sound_mm_projector_type = sound_mm_projector_type
        self.config.sound_mm_projector_type = sound_mm_projector_type

        if hasattr(config, "sound_mm_projector_cfg") and isinstance(config.sound_mm_projector_cfg, dict):
            config.sound_mm_projector_cfg["sound_mm_projector_type"] = sound_mm_projector_type

        if sound_mm_projector_type != "mlp":
            raise ValueError(
                f"Unsupported sound_mm_projector_type '{sound_mm_projector_type}'. "
                "Current OmniVinci checkpoint requires 'mlp'."
            )

        self.layers = nn.Sequential(
            nn.Linear(config.sound_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.post_init()

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


AutoConfig.register("sound_mm_projector", SoundMultimodalProjectorConfig)
AutoModel.register(SoundMultimodalProjectorConfig, SoundMultimodalProjector)


class AudioTower(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sounds):
        if isinstance(sounds, list):
            sound_features = []
            audio_output_lengths = []
            for sound in sounds:
                if hasattr(sound, "input_features"):
                    sound = sound["input_features"]
                sound_feature = self.audio_tower(sound)
                sound_feature = sound_feature.last_hidden_state
                sound_feature = sound_feature.to(sound.dtype)
                sound_features.append(sound_feature)
                audio_output_lengths.append(sound_feature.shape[1])
            sound_features = torch.cat(sound_features, dim=1).squeeze(0)
        else:
            raise NotImplementedError("Not implemented for this encoder")

        return sound_features, audio_output_lengths

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


class Qwen2AudioTower(AudioTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__()
        self.audio_tower = Qwen2AudioEncoder.from_pretrained(
            model_name_or_path, attn_implementation=_get_attn_implementation(config)
        )
        self.audio_chunk_unit_duration = 30
        self.audio_chunk_unit_length = 3000

    def forward(self, sounds):
        if isinstance(sounds, list):
            sound_features = []
            audio_output_lengths = []
            for sound in sounds:
                if hasattr(sound, "input_features") or (isinstance(sound, dict) and "input_features" in sound):
                    sound = sound["input_features"]

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


class VisionTower(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

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
            raise ValueError("VisionTower expects batched tensor input, not list.")
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
        return self.config.hidden_size



class VisionTowerDynamicS2(VisionTower):
    def __init__(self, args):
        super().__init__(args)

        self.scales = list(map(int, args.s2_scales.split(",")))
        self.scales.sort()
        self.max_split_size = args.s2_max_split_size
        self.resize_output_to_scale_idx = getattr(args, "s2_resize_output_to_scale_idx", 0)

    def forward(self, images):
        if isinstance(images, list):
            raise ValueError("VisionTowerDynamicS2 expects tensor input, not list.")
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        return self.feature_select(image_forward_outs).to(images.dtype)

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.scales)


class SiglipVisionTowerDynamicS2(VisionTowerDynamicS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__(config)
        if isinstance(config.model_dtype, str):
            model_dtype = eval(config.model_dtype)
        else:
            model_dtype = config.model_dtype

        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            attn_implementation=_get_attn_implementation(config),
            torch_dtype=model_dtype,
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[0]


def build_mm_projector(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    """Build multimodal projector from path or configuration."""
    if model_type_or_path is None:
        return None
    if config.resume_path:
        assert os.path.exists(model_type_or_path), f"Resume mm projector path {model_type_or_path} does not exist!"
        return MultimodalProjector.from_pretrained(model_type_or_path, config)
    else:
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config)
        return mm_projector


def build_sound_mm_projector(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    """Build sound multimodal projector from path or configuration."""
    if model_type_or_path is None:
        return None

    if isinstance(config.model_dtype, str):
        model_dtype = eval(config.model_dtype)
    else:
        model_dtype = config.model_dtype
    if config.resume_path:
        assert os.path.exists(
            model_type_or_path
        ), f"Resume sound mm projector path {model_type_or_path} does not exist!"
        _model = SoundMultimodalProjector.from_pretrained(model_type_or_path, config, torch_dtype=model_dtype)
        return _model
    else:
        sound_mm_projector_cfg = SoundMultimodalProjectorConfig(model_type_or_path)
        sound_mm_projector = SoundMultimodalProjector(sound_mm_projector_cfg, config).to(model_dtype)
        return sound_mm_projector


def build_vision_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    """Build vision tower from path or configuration."""
    if model_name_or_path is None:
        return None

    if config.resume_path and "radio" not in model_name_or_path:
        assert os.path.exists(model_name_or_path), f"Resume vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
        if "siglip" not in vision_tower_arch:
            raise NotImplementedError(f"Unknown vision tower architecture: {vision_tower_arch}")

    if not getattr(config, "dynamic_s2", False):
        raise NotImplementedError("Current OmniVinci checkpoint requires `dynamic_s2=True`.")

    vision_tower = SiglipVisionTowerDynamicS2(model_name_or_path, config)
    config.mm_hidden_size = vision_tower.hidden_size
    return vision_tower


def build_audio_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    """Build the audio tower used for sound."""
    if model_name_or_path is None:
        return None

    model = Qwen2AudioTower(model_name_or_path, config)
    config.sound_hidden_size = 1280
    return model


class VILAPretrainedModel(PreTrainedModel):
    config_class = OmniVinciConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["Qwen2DecoderLayer", "SiglipEncoderLayer"]

    def __init__(self, config: OmniVinciConfig, *args, **kwargs):
        super().__init__(config)
        self.config = config
        llm_cfg = _resolve_component_path(config, "llm_cfg")
        vision_tower_cfg = _resolve_component_path(config, "vision_tower_cfg")
        mm_projector_cfg = _resolve_component_path(config, "mm_projector_cfg")
        sound_tower_cfg = _resolve_component_path(config, "sound_tower_cfg")
        sound_mm_projector_cfg = _resolve_component_path(config, "sound_mm_projector_cfg")
        missing = [
            name
            for name, path in [
                ("llm_cfg", llm_cfg),
                ("vision_tower_cfg", vision_tower_cfg),
                ("mm_projector_cfg", mm_projector_cfg),
            ]
            if not path
        ]
        if missing:
            raise ValueError(f"Missing required OmniVinci components in config: {', '.join(missing)}")

        if bool(sound_tower_cfg) != bool(sound_mm_projector_cfg):
            raise ValueError("`sound_tower_cfg` and `sound_mm_projector_cfg` must be both set or both empty.")

        # loading on auto by default
        device_map = kwargs.get("device_map", "auto")
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)

        if sound_tower_cfg:
            self.sound_tower = build_audio_tower(sound_tower_cfg, config)
            self.sound_mm_projector = build_sound_mm_projector(sound_mm_projector_cfg, config)

        if device_map in ["auto", "cuda"]:
            self.mm_projector = self.mm_projector.cuda()
            self.vision_tower = self.vision_tower.cuda()
            self.sound_tower = self.sound_tower.cuda() if hasattr(self, "sound_tower") else None
            self.sound_mm_projector = self.sound_mm_projector.cuda() if hasattr(self, "sound_mm_projector") else None
        # set device_map auto can autoamtically shard llm to different devices
        self.llm, self.tokenizer = self.init_llm(llm_cfg, config, device_map=device_map)

        self.llm_model_embed_tokens = self.llm.model.embed_tokens

        self.tokenizer.padding_side = "left"

        self.vocab_size = len(self.tokenizer)
        self.update_vocab_size = lambda: setattr(self, "vocab_size", len(self.tokenizer))

        self.encoders = {}
        for name in ["image", "video", "sound"]:
            encoder_config = getattr(self.config, f"{name}_encoder")
            if isinstance(encoder_config, str):
                encoder_config = json.loads(encoder_config)
            if encoder_config.get("embed_time", False) == "True":
                if "trope_dim" not in encoder_config and encoder_config.get("time_embed_type", "") in [
                    "pixel",
                    "lang",
                ]:
                    encoder_config["trope_dim"] = self.config.hidden_size // 2
                    print(
                        f"Warning: trope_dim not found in config, defaulting to hidden_size // 2: {encoder_config['trope_dim']}"
                    )

            encoder_config.pop("_target_")
            if name == "video":
                self.encoders[name] = TSPVideoEncoder(parent=self, **encoder_config)
            elif name == "image":
                self.encoders[name] = BasicImageEncoder(self)
            else:
                self.encoders[name] = BasicSoundEncoder(parent=self, **encoder_config)

        self.post_config()

        self.llm_only_need_embed = kwargs.get("llm_only_need_embed", False)
        if self.llm_only_need_embed:
            print("We only need the embed_tokens in llm.")
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."

    @classmethod
    def copy_remote_py_files(cls, output_dir, copy=True):
        # copy .py and README for next loading
        current_file_path = os.path.abspath(__file__)
        current_folder = os.path.dirname(current_file_path)
        for file_name in os.listdir(current_folder):
            if file_name == "INSTRUCTIONS.md":
                src_fname = os.path.join(current_folder, file_name)
                dst_fname = os.path.join(output_dir, "README.md")
                if os.path.exists(dst_fname):
                    old_readme = open(dst_fname).read()
                else:
                    old_readme = ""
                with open(src_fname) as src, open(dst_fname, "w") as dst:
                    dst.write(src.read())
                    dst.write(old_readme)
                print("[HF] README", src_fname, "to", dst_fname)
            if file_name.endswith(".py") or file_name.endswith(".jinja"):
                full_file_name = os.path.join(current_folder, file_name)
                if os.path.isfile(full_file_name):
                    if copy:
                        shutil.copy(full_file_name, output_dir)
                        print("[HF] copying", full_file_name, "to", output_dir)
                    else:
                        # symlink to ease development
                        if os.path.exists(os.path.join(output_dir, file_name)):
                            os.remove(os.path.join(output_dir, file_name))
                        os.symlink(full_file_name, os.path.join(output_dir, file_name))
                        print("[HF] linking", full_file_name, "to", output_dir)

    def save_pretrained(self, output_dir, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.state_dict()

        if getattr(self, "tokenizer", None):
            self.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.llm:
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.llm.config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            self.llm.save_pretrained(os.path.join(output_dir, "llm"), state_dict=llm_state_dict)
            self.config.llm_cfg = self.llm.config

        if self.vision_tower:
            print(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.vision_tower.config._name_or_path = osp.join(output_dir, "vision_tower")
            vision_tower_state_dict = OrderedDict(
                {k.split("vision_tower.vision_tower.")[-1]: v for k, v in state_dict.items() if "vision_tower" in k}
            )
            self.vision_tower.vision_tower.save_pretrained(
                os.path.join(output_dir, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            self.vision_tower.image_processor.save_pretrained(os.path.join(output_dir, "vision_tower"))
            self.config.vision_tower_cfg = self.vision_tower.config
            if hasattr(self.config.vision_tower_cfg, "auto_map"):
                if "radio" not in self.vision_tower.__class__.__name__.lower():
                    delattr(self.config.vision_tower_cfg, "auto_map")
        if getattr(self, "sound_tower", None):
            print(f"saving sound_tower to {osp.join(output_dir, 'sound_tower')}")
            self.sound_tower.config._name_or_path = osp.join(output_dir, "sound_tower").replace(
                "tmp-checkpoint", "checkpoint"
            )

            sound_tower_state_dict = OrderedDict(
                {k.split("sound_tower.audio_tower.")[-1]: v for k, v in state_dict.items() if "sound_tower" in k}
            )

            self.sound_tower.audio_tower.save_pretrained(
                os.path.join(output_dir, "sound_tower"),
                state_dict=sound_tower_state_dict,
            )
            self.config.sound_tower_cfg = self.sound_tower.config

        if self.mm_projector:
            print(f"saving mm_projector to {osp.join(output_dir, 'mm_projector')}")
            self.mm_projector.config._name_or_path = osp.join(output_dir, "mm_projector")
            mm_projector_state_dict = OrderedDict(
                {k.split("mm_projector.")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
            )
            self.mm_projector.save_pretrained(
                os.path.join(output_dir, "mm_projector"),
                state_dict=mm_projector_state_dict,
            )
            self.config.mm_projector_cfg = self.mm_projector.config

        if getattr(self, "sound_mm_projector", None):
            print(f"saving sound_mm_projector to {osp.join(output_dir, 'sound_mm_projector')}")
            self.sound_mm_projector.config._name_or_path = osp.join(output_dir, "sound_mm_projector").replace(
                "tmp-checkpoint", "checkpoint"
            )

            sound_mm_projector_state_dict = OrderedDict(
                {k.split("sound_mm_projector.")[-1]: v for k, v in state_dict.items() if "sound_mm_projector" in k}
            )
            self.sound_mm_projector.save_pretrained(
                os.path.join(output_dir, "sound_mm_projector"),
                state_dict=sound_mm_projector_state_dict,
            )
            self.config.sound_mm_projector_cfg = self.sound_mm_projector.config

        # update and save top-level config
        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

        # copy .py and README for next loading
        self.copy_remote_py_files(output_dir)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        if not isinstance(config, PretrainedConfig):
            config = OmniVinciConfig.from_pretrained(pretrained_model_name_or_path)
        if pretrained_model_name_or_path is not None:
            config._name_or_path = str(pretrained_model_name_or_path)
            if getattr(config, "resume_path", None) is None or not osp.exists(str(config.resume_path)):
                config.resume_path = str(pretrained_model_name_or_path)
        if kwargs.get("torch_dtype", None) is not None:
            config.torch_dtype = kwargs.get("torch_dtype", None)
            config.model_dtype = kwargs.get("torch_dtype", None)
            if isinstance(kwargs.get("torch_dtype", None), str):
                kwargs["torch_dtype"] = eval(kwargs.get("torch_dtype", None))
            else:
                kwargs["torch_dtype"] = kwargs.get("torch_dtype", None)
        return cls._from_config(config, **kwargs)

    def init_llm(self, llm_config, config, *args, **kwargs):
        """Initialize language model and tokenizer."""
        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_config, config, *args, **kwargs)

        self.pad_token_list = (
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.tokenize("<|endoftext|>")[0],  # for Qwen
        )

        self.vocab_size = len(self.tokenizer)
        self.update_vocab_size = lambda: setattr(self, "vocab_size", len(self.tokenizer))
        # XGrammar tokenizer and grammar compiler
        # lazy init only when specified json output during inference
        self.grammar_compiler = None
        # self.llm.resize_token_embeddings(len(self.tokenizer))
        return self.llm, self.tokenizer

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
            self.config.mm_projector_cfg = self.mm_projector.config
        if getattr(self.config, "sound_tower_cfg", None) is None and hasattr(self, "sound_tower"):
            self.config.sound_tower_cfg = self.sound_tower.config
        if getattr(self.config, "sound_mm_projector_cfg", None) is None and hasattr(self, "sound_mm_projector"):
            self.config.sound_mm_projector_cfg = self.sound_mm_projector.config

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


class VILAForCausalLM(VILAPretrainedModel, GenerationMixin):
    def __init__(self, config: OmniVinciConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

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
        block_sizes: Optional[Optional[Tuple[int, ...]]] = None,
        mm_info: Optional[dict] = None,
        num_frames: Optional[List[int]] = None,
    ):
        _ = (mm_info, num_frames)
        if not getattr(self.config, "dynamic_s2", False):
            raise NotImplementedError("Current OmniVinci checkpoint requires `dynamic_s2=True`.")

        bs = len(inp)
        cache_feas = []
        cache_feas_index = []
        inp_block_sizes = block_sizes

        # handle cache features
        for _idx in range(len(inp)):
            if isinstance(inp[_idx], CacheFeatures):
                cache_feas.append(inp[_idx])
                cache_feas_index.append(_idx)
        raw_images = [_ for _ in inp if not isinstance(_, CacheFeatures)]

        raw_videos_num_frames = [_.shape[0] for _ in raw_images]
        if len(raw_images) > 0:
            images = torch.cat(raw_images, dim=0)
        else:
            images = []

        if block_sizes is None:
            block_sizes = [None] * len(images)

        def _load_video_features(image_features, cache_feas, cache_feas_index, raw_videos_num_frames):
            # load cache features
            if len(cache_feas) > 0:
                if len(image_features) > 0:
                    image_features = torch.split(image_features, raw_videos_num_frames)
                new_image_features = []
                cache_feas_idx = 0
                raw_fea_idx = 0
                for _idx in range(bs):
                    if _idx in cache_feas_index:
                        new_image_features.append(
                            cache_feas[cache_feas_idx].value["features"].to(self.device, self.dtype)
                        )
                        cache_feas_idx += 1
                    else:
                        new_image_features.append(image_features[raw_fea_idx])
                        raw_fea_idx += 1

                assert len(new_image_features) == bs
                image_features = new_image_features
                image_features = torch.cat(image_features, dim=0)
            return image_features

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

        # load cache features
        image_features = _load_video_features(image_features, cache_feas, cache_feas_index, raw_videos_num_frames)

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
        block_sizes: Optional[Optional[Tuple[int, ...]]] = None,
        mm_info: Optional[dict] = None,
        num_frames: Optional[List[int]] = None,
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

    def encode_sound(self, sounds, mm_info: Optional[dict] = None):
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
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            assert (
                self.encoders["video"].end_tokens is None
            ), "end_tokens must be None for interleaved vis-aud in video"
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
                                and not _vis_fea_end == media_embeds["video"][video_embeds_idx].shape[0]
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

            assert len(new_video_embeds) == len(
                media_embeds["video"]
            ), "The number of new video embeddings does not match the number of original video embeddings."
            media_embeds["video"] = new_video_embeds
        # Remove padding
        batch_size = labels.shape[0]
        text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]
        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        # Fuse text and media embeddings
        inputs_m, labels_m = [], []
        sound_embeds_idx = 0
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    name = media_tokens[input_ids[k][pos].item()]
                    if input_ids[k][pos].item() == self.tokenizer.media_token_ids["sound"]:
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
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        mm_info,
    ) -> Dict[str, List[torch.Tensor]]:
        embeds = defaultdict(deque)

        def _prepare_sound_media(sound_media: List[Any], max_audio_duration: int) -> List[Any]:
            cur_batch_max_audio_samples = max_audio_duration * self.config.audio_sampling_rate
            whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.config._name_or_path,
                chunk_length=max_audio_duration,
                sampling_rate=self.config.audio_sampling_rate,
                hop_length=self.config.audio_hop_length,
            )

            new_media = []
            aud_idx = 0
            audio_infos = mm_info.get("audio_info", [])
            for _batch_idx in range(len(audio_infos)):
                _audio_info = audio_infos[_batch_idx]
                if _audio_info is None:
                    continue
                for _mm_idx in range(len(_audio_info)):
                    if aud_idx >= len(sound_media):
                        raise ValueError("The number of audio info does not match the number of audio samples.")

                    _audio = sound_media[aud_idx]
                    if isinstance(_audio, torch.Tensor):
                        device = _audio.device
                        dtype = _audio.dtype
                        _audio = _audio.cpu().float()
                    else:
                        device = self.device
                        dtype = self.dtype

                    _audio = whisper.pad_or_trim(_audio, length=cur_batch_max_audio_samples)
                    aud_idx += 1
                    stft_features = whisper_feature_extractor(
                        _audio,
                        sampling_rate=self.config.audio_sampling_rate,
                        return_attention_mask=True,
                        padding="max_length",
                        return_tensors="pt",
                    ).to(device, dtype)

                    new_media.append(stft_features)
                    if _audio_info[_mm_idx] != "dummy":
                        _audio_info[_mm_idx]["new_audio_chunk_length"] = max_audio_duration
                        _audio_info[_mm_idx]["new_audio_n_samples"] = cur_batch_max_audio_samples
                        _audio_info[_mm_idx]["audio_end_sample_sec"] = (
                            _audio_info[_mm_idx]["audio_start_sec"] + max_audio_duration
                        )
                        _audio_info[_mm_idx]["new_audio_n_stft_frames"] = stft_features["input_features"].shape[-1]

            if aud_idx != len(sound_media):
                raise ValueError("The number of audio info does not match the number of audio samples.")
            return new_media

        for name in media:
            _encoder = self.encoders[name]

            if name == "sound":
                sound_media = media.get(name, [])
                if len(sound_media) == 0:
                    continue

                if self.training:
                    cur_batch_max_audio_samples = max(len(_audio) for _audio in sound_media)
                    cur_batch_max_audio_samples = int(
                        np.ceil(cur_batch_max_audio_samples / (self.config.audio_sampling_rate * 30))
                        * (self.config.audio_sampling_rate * 30)
                    )  # should be multiple of 30 seconds
                    cur_batch_max_audio_samples = min(
                        cur_batch_max_audio_samples,
                        self.config.audio_chunk_length * self.config.audio_sampling_rate,
                    )
                    cur_batch_max_audio_duration = cur_batch_max_audio_samples // self.config.audio_sampling_rate
                else:
                    all_audio_chunk_lengths = []
                    audio_infos = mm_info.get("audio_info", [])
                    for _audio_info in audio_infos:
                        if _audio_info is None:
                            continue
                        for _mm_idx in range(len(_audio_info)):
                            all_audio_chunk_lengths.append(_audio_info[_mm_idx]["new_audio_chunk_length"])
                    if not all_audio_chunk_lengths:
                        continue
                    cur_batch_max_audio_duration = max(all_audio_chunk_lengths)

                media[name] = _prepare_sound_media(sound_media, cur_batch_max_audio_duration)

            if len(media[name]) > 0:
                embeds[name] = deque(_encoder(media[name], media_config[name], mm_info))
        return embeds

    def __truncate_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and any(len(input) > self.tokenizer.model_max_length for input in inputs):
            warnings.warn(f"Truncating sequences to `model_max_length` ({self.tokenizer.model_max_length}).")
            inputs = [input[: self.tokenizer.model_max_length] for input in inputs]
            labels = [label[: self.tokenizer.model_max_length] for label in labels]
        return inputs, labels

    def __batchify_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if self.tokenizer.padding_side == "right":
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
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        images: Optional[torch.FloatTensor] = None,
        media_config: Optional[List] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        packing: bool = True,
        force_packing: bool = False,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        if images is not None:
            if media is not None:
                raise ValueError("Both 'media' and 'images' are provided. Please provide only one.")
            print("The 'images' argument is deprecated. Please use 'media' instead.")
            media = {"image": images}

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

        if self.training and getattr(self.config, "time_token_ids", []):
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                labels,
                soft_tokens=self.config.time_token_ids,
                std=self.config.soft_ce_std,
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

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(self.generation_config or GenerationConfig())
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.eos_token_id
        return generation_config
