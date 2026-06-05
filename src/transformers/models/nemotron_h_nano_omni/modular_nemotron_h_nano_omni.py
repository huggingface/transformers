# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import transformers

from ...generation import GenerationConfig
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..nemotron_h import NemotronHForCausalLM
from ..radio import RADIOModel
from .audio_model import SoundEncoder, SoundProjection
from .configuration_nemotron_h_nano_omni import NemotronH_Nano_Omni_Reasoning_V3_Config
from .evs import EfficientVideoSampling


logger = logging.get_logger(__name__)


__all__ = ["NemotronH_Nano_Omni_Reasoning_V3"]


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


def version_cmp(v1, v2, op="eq"):
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class NemotronH_Nano_Omni_Reasoning_V3(PreTrainedModel):
    config_class = NemotronH_Nano_Omni_Reasoning_V3_Config
    main_input_name = "pixel_values"
    _supports_flash_attn_2 = True
    _supports_flash_attn = True
    _no_split_modules = ["NemotronHBlock"]

    def __init__(self, config: NemotronH_Nano_Omni_Reasoning_V3_Config):
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.36.2", "ge")
        image_size = config.force_image_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.image_tag_type = config.image_tag_type
        self.img_context_token_id = config.img_context_token_id
        self.video_context_token_id = config.video_context_token_id

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")

        self.language_model = NemotronHForCausalLM(config.llm_config)
        self.vision_model = RADIOModel(config.vision_config)
        # WAR for transformers issue 38358 — RADIO's `_init_weights` shadows the base
        # `_initialize_weights` lookup; alias them so the meta-init path resolves.
        self.vision_model.model._initialize_weights = self.vision_model.model._init_weights
        self.vision_model.radio_model.make_preprocessor_external()

        # 3D video patch projection. The RADIO ViT only ships a 2D `embedder`
        # `[embed_dim, C·P²]`; this checkpoint also carries a `video_embedder`
        # `[embed_dim, T·C·P²]` for temporally-packed video patches.
        self.video_temporal_patch_dim = config.video_temporal_patch_size
        pg = self.vision_model.radio_model.model.patch_generator
        pg.video_embedder = nn.Linear(
            in_features=self.video_temporal_patch_dim * 3 * pg.patch_size * pg.patch_size,
            out_features=pg.embed_dim,
            bias=False,
        )

        self.vision_model = self.vision_model.to(self.language_model.config.torch_dtype)

        self.drop_vision_class_token = True

        vit_hidden_size = config.vit_hidden_size
        vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.video_pruning_rate = config.video_pruning_rate

        self.mlp1 = nn.Sequential(
            RMSNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, eps=1e-5),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                vision_projection_hidden_size,
                bias=False,
            ),
            SquaredReLU(),
            nn.Linear(vision_projection_hidden_size, llm_hidden_size, bias=False),
        )
        self.mlp1 = self.mlp1.to(self.language_model.config.torch_dtype)

        self.sound_context_token_id = getattr(config, "sound_context_token_id", None)
        if config.sound_config is not None:
            sound_config = config.sound_config
            sound_hidden_size = sound_config.hidden_size
            sound_projection_hidden_size = sound_config.projection_hidden_size

            from ..parakeet import ParakeetFeatureExtractor

            sampling_rate = getattr(sound_config, "sampling_rate", 16000)
            feature_size = getattr(sound_config, "num_mel_bins", 128)
            self.sound_feature_extractor = ParakeetFeatureExtractor(
                sampling_rate=sampling_rate,
                feature_size=feature_size,
            )

            self.sound_encoder = SoundEncoder(config=sound_config)
            self.sound_encoder = self.sound_encoder.to(self.language_model.config.torch_dtype)

            self.sound_projection = SoundProjection(
                sound_hidden_size=sound_hidden_size,
                projection_hidden_size=sound_projection_hidden_size,
                llm_hidden_size=llm_hidden_size,
                bias=sound_config.projection_bias,
            )
            self.sound_projection = self.sound_projection.to(self.language_model.config.torch_dtype)
        else:
            self.sound_encoder = None
            self.sound_projection = None
            self.sound_feature_extractor = None

        self.all_tied_weights_keys = {}

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_flags: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        inputs_embeds=None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        image_flags = image_flags.squeeze(-1)

        B, N, C = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id

        vit_embeds = self.extract_feature(pixel_values)
        del pixel_values

        vit_embeds = vit_embeds[image_flags == 1]
        try:
            inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            logger.warning(
                f"warning: {e}, inputs_embeds[selected].shape={inputs_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds[:n_token]

        del vit_embeds
        inputs_embeds = inputs_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if isinstance(pixel_values, (list, tuple)):
            outs = [self._extract_feature_single(pv) for pv in pixel_values]
            return torch.cat(outs, dim=0)
        return self._extract_feature_single(pixel_values)

    def _extract_feature_single(self, pixel_values):
        pixel_values = pixel_values.to(dtype=self.vision_model.config.torch_dtype)
        vit_embeds = self.vision_model(pixel_values).features
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        patch_size = self.vision_model.radio_model.model.patch_generator.patch_size
        B, _, H, W = pixel_values.shape
        h = H // patch_size
        w = W // patch_size
        vit_embeds = vit_embeds.reshape(B, h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(B, -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def extract_video_feature(self, pixel_values_videos):
        pg = self.vision_model.radio_model.model.patch_generator
        T = self.video_temporal_patch_dim
        N, C, H, W = pixel_values_videos.shape

        if N % T != 0:
            pad = pixel_values_videos[-1:].expand(T - (N % T), -1, -1, -1)
            pixel_values_videos = torch.cat([pixel_values_videos, pad], dim=0)
            N = pixel_values_videos.shape[0]
        num_groups = N // T

        x = pixel_values_videos.reshape(num_groups, T * C, H, W)

        orig_embedder = pg.embedder
        pg.embedder = pg.video_embedder
        try:
            vit_embeds = self.vision_model(x).features
        finally:
            pg.embedder = orig_embedder

        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        patch_size = pg.patch_size
        h = H // patch_size
        w = W // patch_size
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def extract_sound_feature(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.sound_encoder is None:
            raise RuntimeError("Sound encoder not initialized. Check if sound_config is provided.")
        sound_embeds = self.sound_encoder(input_features, attention_mask)
        sound_embeds = sound_embeds.to(dtype=torch.bfloat16)
        sound_embeds = self.sound_projection(sound_embeds)
        return sound_embeds

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        sound_clips: torch.FloatTensor | None = None,
        sound_length: torch.Tensor | None = None,
        input_ids: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        generation_config: GenerationConfig | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None

        has_images = pixel_values is not None
        has_videos = pixel_values_videos is not None
        has_sound = sound_clips is not None and self.sound_encoder is not None

        if has_images or has_videos or has_sound:
            image_vit_embeds, video_vit_embeds, sound_embeds = None, None, None

            if has_images:
                pixel_values = pixel_values.to(dtype=self.vision_model.config.torch_dtype)
                image_vit_embeds = self.extract_feature(pixel_values)

            if has_videos:
                pixel_values_videos = pixel_values_videos.to(dtype=self.vision_model.config.torch_dtype)
                video_vit_embeds = self.extract_video_feature(pixel_values_videos)

            if has_sound:
                import numpy as np

                is_raw_waveform = False
                if isinstance(sound_clips, (list, tuple)):
                    is_raw_waveform = True
                    waveforms = sound_clips
                elif isinstance(sound_clips, np.ndarray):
                    is_raw_waveform = True
                    waveforms = [sound_clips.squeeze()] if sound_clips.ndim > 1 else [sound_clips]
                elif isinstance(sound_clips, torch.Tensor):
                    if sound_clips.dim() == 1:
                        is_raw_waveform = True
                        waveforms = [sound_clips.cpu().numpy()]
                    elif sound_clips.dim() == 2:
                        is_raw_waveform = True
                        waveforms = [clip.cpu().numpy() for clip in sound_clips]
                    else:
                        is_raw_waveform = False
                else:
                    is_raw_waveform = False

                if is_raw_waveform:
                    audio_inputs = self.sound_feature_extractor(
                        waveforms,
                        sampling_rate=self.sound_feature_extractor.sampling_rate,
                        return_tensors="pt",
                    )
                    sound_input_features = audio_inputs.input_features
                    sound_attention_mask = audio_inputs.get("attention_mask", None)
                else:
                    sound_input_features = sound_clips
                    sound_attention_mask = None

                target_device = self.sound_encoder.encoder.subsampling.linear.weight.device
                target_dtype = self.language_model.config.torch_dtype

                sound_input_features = sound_input_features.to(dtype=target_dtype, device=target_device)
                if sound_attention_mask is not None:
                    sound_attention_mask = sound_attention_mask.to(device=target_device)

                sound_embeds = self.extract_sound_feature(sound_input_features, sound_attention_mask)

            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            input_ids_copy = input_ids.reshape(B * N)

            if image_vit_embeds is not None:
                image_mask = input_ids_copy == self.img_context_token_id
                assert image_mask.sum() != 0, "No image tokens found in input_ids"
                inputs_embeds[image_mask] = image_vit_embeds.reshape(-1, C).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

            if video_vit_embeds is not None:
                if B > 1:
                    raise NotImplementedError("Video is not supported for batch size > 1")
                video_mask = input_ids_copy == self.img_context_token_id
                assert video_mask.sum() != 0, "No video tokens found in input_ids"
                inputs_embeds[video_mask] = video_vit_embeds.reshape(-1, C).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

            if sound_embeds is not None and self.sound_context_token_id is not None:
                sound_mask = input_ids_copy == self.sound_context_token_id
                assert sound_mask.sum() != 0, "No sound tokens found in input_ids"
                if sound_embeds.dim() == 3 and sound_embeds.shape[0] > 1 and sound_attention_mask is not None:
                    natural_input_lengths = sound_attention_mask.sum(-1) + 1
                    output_lengths = self.sound_encoder.encoder._get_subsampling_output_length(natural_input_lengths)
                    flat = torch.cat(
                        [sound_embeds[i, : int(n)] for i, n in enumerate(output_lengths.tolist())],
                        dim=0,
                    )
                else:
                    flat = sound_embeds.reshape(-1, C)
                assert sound_mask.sum().item() == flat.shape[0], (
                    f"sound token count ({sound_mask.sum().item()}) != encoder output count ({flat.shape[0]})"
                )
                inputs_embeds[sound_mask] = flat.to(inputs_embeds.device, inputs_embeds.dtype)

            if video_vit_embeds is not None and self.video_pruning_rate > 0:
                h = w = int(video_vit_embeds.shape[1] ** 0.5)
                evs_mask = EfficientVideoSampling.compute_retention_mask(
                    video_embeds=video_vit_embeds,
                    thw=(video_vit_embeds.shape[0], h, w),
                    spatial_merge_size=1,
                    q=self.video_pruning_rate,
                )
                retention_mask = torch.ones_like(input_ids_copy, dtype=torch.bool)
                retention_mask[video_mask] = evs_mask.view(-1)
                inputs_embeds = inputs_embeds[retention_mask].unsqueeze(0)
                if attention_mask is not None:
                    attention_mask = attention_mask[:, retention_mask].contiguous()
                if input_ids is not None:
                    input_ids = input_ids[:, retention_mask].contiguous()
            else:
                inputs_embeds = inputs_embeds.reshape(B, N, C)
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
