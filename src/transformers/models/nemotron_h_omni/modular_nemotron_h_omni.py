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

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from ..auto import AutoModel, AutoModelForCausalLM
from ..internvl.modeling_internvl import InternVLMultiModalProjector
from ..nemotron_h.modeling_nemotron_h import NemotronHRMSNorm
from .configuration_nemotron_h_omni import NemotronH_Omni_Reasoning_V3_Config


logger = logging.get_logger(__name__)


__all__ = ["NemotronH_Omni_Reasoning_V3PreTrainedModel", "NemotronH_Omni_Reasoning_V3"]


class NemotronH_Omni_RMSNorm(NemotronHRMSNorm):
    pass


def compute_retention_mask(
    *,
    video_embeds: torch.FloatTensor,
    thw: torch.LongTensor,
    spatial_merge_size: int,
    q: float,
):
    """
    Computes the retention mask for video embeddings based on the grid dimensions.

    Args:
        video_embeds (`torch.FloatTensor` of shape `(T * H * W, hidden_size)`):
            The video embeddings to compute the retention mask for.
        thw (`torch.LongTensor` of shape `(3)`):
            The temporal, height and width of feature shape of each video in LLM.
        spatial_merge_size (`int`): The spatial merge size of the video embeddings.
            If embeddings will be downsampled *later*, this should be the downsampling factor.
        q: (`float`): Pruning rate factor, indicating number of tokens to prune (remove)

    Returns:
        `torch.Tensor`: The retention mask for the video embeddings (T * H * W).
            1 for tokens to keep, 0 for tokens to prune.
    """
    T, H, W = thw

    # Use reshape instead of einops to avoid graph breaks
    video_embeds = video_embeds.reshape(T, H // spatial_merge_size, W // spatial_merge_size, video_embeds.size(-1))

    # Core EVS
    similarity = torch.nn.functional.cosine_similarity(video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1)
    dissimilarity = 1 - similarity

    # Always ensure we include all tokens from the first frame
    dissimilarity = torch.cat([255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0)
    dissimilarity_flat = dissimilarity.view(-1)

    min_num_tokens = (H // spatial_merge_size) * (W // spatial_merge_size)  # a single frame
    evs_num_tokens = int(T * min_num_tokens * (1 - q))
    num_tokens_to_keep = max(min_num_tokens, evs_num_tokens)

    order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)
    topk_indices = order[:num_tokens_to_keep]

    retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
    retention_mask[topk_indices] = True
    retention_mask = retention_mask.reshape(dissimilarity.size())

    mask = retention_mask.view(-1)  # "T H W -> (T H W)"
    return mask


class NemotronH_Omni_Reasoning_V3MultiModalProjector(InternVLMultiModalProjector):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, bias: bool = False, eps: float = 1e-5):
        nn.Module.__init__(self)
        self.layer_norm = NemotronH_Omni_RMSNorm(input_size, eps=eps)
        self.linear_1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.act = ACT2FN["relu2"]
        self.linear_2 = nn.Linear(hidden_size, output_size, bias=bias)


@auto_docstring
class NemotronH_Omni_Reasoning_V3PreTrainedModel(PreTrainedModel):
    config: NemotronH_Omni_Reasoning_V3_Config
    main_input_name = "input_ids"
    input_modalities = ("image", "video", "audio", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _is_stateful = True
    # mel extraction runs in the processor, so the checkpoint's in-encoder featurizer buffers are unused
    _keys_to_ignore_on_load_unexpected = [r"audio_tower\.feature_extractor\."]
    # checkpoints never store the MTP vision norm; it keeps its identity affine initialization
    _keys_to_ignore_on_load_missing = [r"vision_final_layernorm\."]


class NemotronH_Omni_Reasoning_V3(NemotronH_Omni_Reasoning_V3PreTrainedModel, GenerationMixin):
    def __init__(self, config: NemotronH_Omni_Reasoning_V3_Config):
        super().__init__(config)
        self.image_token_id = config.image_token_id
        self.audio_token_id = config.audio_token_id
        self.video_pruning_rate = config.video_pruning_rate
        self.video_temporal_patch_dim = config.video_temporal_patch_size

        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.vision_model = AutoModel.from_config(config.vision_config)
        self.vision_model.make_preprocessor_external()
        # Megatron-Core adds a final LayerNorm to every block built from the shared config when the language model
        # has MTP layers, including the vision tower, so its features are normalized before the projector.
        self.vision_final_layernorm = (
            # CODEPATH: only checkpoints whose language model carries MTP layers build this norm.
            nn.LayerNorm(config.vision_hidden_size, eps=config.vision_config.layer_norm_eps)
            if config.text_config.num_nextn_predict_layers > 0
            else None
        )
        self.multi_modal_projector = NemotronH_Omni_Reasoning_V3MultiModalProjector(
            config.vision_hidden_size * int(1 / config.downsample_ratio) ** 2,
            config.projector_hidden_size,
            config.text_config.hidden_size,
        )

        # CODEPATH: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` ships an `audio_config` and builds the
        # audio tower; configs without one have no audio branch.
        self.audio_tower = AutoModel.from_config(config.audio_config) if config.audio_config is not None else None
        self.embed_audio = (
            # CODEPATH: same split as `audio_tower` above.
            NemotronH_Omni_Reasoning_V3MultiModalProjector(
                config.audio_config.hidden_size,
                config.audio_config.projection_hidden_size,
                config.text_config.hidden_size,
                bias=config.audio_config.projection_bias,
            )
            if config.audio_config is not None
            else None
        )

        self.post_init()

    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        batch_size, width, height, channels = vision_features.size()
        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        vision_features = vision_features.transpose(1, 2).contiguous()
        vision_features = vision_features.view(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )
        return vision_features.transpose(1, 2).contiguous()

    def project_vision_features(self, vision_features: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Pixel-shuffle the `(num_images, height * width, vision_hidden_size)` tower features and project them."""
        if self.vision_final_layernorm is not None:
            vision_features = self.vision_final_layernorm(vision_features)
        vision_features = vision_features.reshape(vision_features.shape[0], height, width, -1)
        vision_features = self.pixel_shuffle(vision_features, scale_factor=self.config.downsample_ratio)
        vision_features = vision_features.reshape(vision_features.shape[0], -1, vision_features.shape[-1])
        return self.multi_modal_projector(vision_features)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_hw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(total_patches, num_channels * patch_size**2)`):
            Flattened patches of all images, concatenated.
        image_grid_hw (`torch.LongTensor` of shape `(num_images, 2)`):
            Patch grid `(height, width)` of each image.
        """
        torch_compilable_check(
            image_grid_hw.prod(-1).sum() == pixel_values.shape[0],
            lambda: f"`pixel_values` holds {pixel_values.shape[0]} patches but `image_grid_hw` describes "
            f"{int(image_grid_hw.prod(-1).sum())}",
        )
        pixel_values = pixel_values.to(dtype=self.vision_model.config.torch_dtype)
        vision_outputs = self.vision_model(pixel_values, image_grid_hw=image_grid_hw, **kwargs)

        image_features = vision_outputs.features
        if self.vision_final_layernorm is not None:
            image_features = self.vision_final_layernorm(image_features)
        image_features = image_features.split(image_grid_hw.prod(-1).tolist())
        image_features = torch.cat(
            [
                self.pixel_shuffle(
                    features.view(1, grid_height, grid_width, -1), scale_factor=self.config.downsample_ratio
                ).flatten(0, 2)
                for features, (grid_height, grid_width) in zip(image_features, image_grid_hw.tolist())
            ]
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=self.multi_modal_projector(image_features),
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        pixel_values_videos = pixel_values_videos.to(dtype=self.vision_model.config.torch_dtype)
        temporal_patch_dim = self.video_temporal_patch_dim
        num_frames, channels, height, width = pixel_values_videos.shape

        # Frames are consumed in groups of `temporal_patch_dim`; repeat the last frame to fill the
        # final group so the packed reshape below is exact.
        if num_frames % temporal_patch_dim != 0:
            padding = pixel_values_videos[-1:].expand(
                temporal_patch_dim - (num_frames % temporal_patch_dim), -1, -1, -1
            )
            pixel_values_videos = torch.cat([pixel_values_videos, padding], dim=0)
            num_frames = pixel_values_videos.shape[0]

        packed = pixel_values_videos.reshape(
            num_frames // temporal_patch_dim, temporal_patch_dim * channels, height, width
        )
        vision_outputs = self.vision_model(packed, **kwargs)
        patch_size = self.vision_model.patch_size
        return BaseModelOutputWithPooling(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=self.project_vision_features(
                vision_outputs.features, height // patch_size, width // patch_size
            ),
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @can_return_tuple
    @auto_docstring(
        custom_intro="Encodes mel features with the audio tower and projects them into language model space."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_frames, num_mel_bins)`):
            Mel features produced by the processor.
        input_features_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            Mask marking the real mel frames of each padded clip.
        """
        if self.audio_tower is None:
            raise ValueError("Audio features were requested, but the model was initialized without an `audio_config`.")

        input_features = input_features.to(self.audio_tower.device, self.audio_tower.dtype)
        if input_features_mask is not None:
            input_features_mask = input_features_mask.to(self.audio_tower.device)
        audio_outputs = self.audio_tower(
            input_features=input_features, attention_mask=input_features_mask, return_dict=True, **kwargs
        )
        audio_embeds = self.embed_audio(audio_outputs.last_hidden_state)

        # Clips are batch-padded; keep only each clip's real (subsampled) length before flattening.
        if input_features_mask is not None:
            lengths = self.audio_tower._get_subsampling_output_length(input_features_mask.sum(-1) + 1)
            positions = torch.arange(audio_embeds.shape[1], device=audio_embeds.device)
            audio_embeds = audio_embeds[positions[None, :] < lengths[:, None].to(audio_embeds.device)]
        else:
            audio_embeds = audio_embeds.reshape(-1, audio_embeds.shape[-1])

        return BaseModelOutputWithPooling(
            last_hidden_state=audio_outputs.last_hidden_state,
            pooler_output=audio_embeds,
            hidden_states=audio_outputs.hidden_states,
            attentions=audio_outputs.attentions,
        )

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        features: torch.FloatTensor,
        token_id: int,
    ) -> torch.BoolTensor:
        """Locates the placeholder tokens of one modality and checks that each receives one feature vector."""
        special_mask = input_ids == token_id
        num_tokens = special_mask.sum()
        torch_compilable_check(
            num_tokens * inputs_embeds.shape[-1] == features.numel(),
            lambda: f"Got {features.numel() // inputs_embeds.shape[-1]} feature vectors for {int(num_tokens)} "
            f"placeholder tokens (id {token_id})",
        )
        return special_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_hw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(total_patches, num_channels * patch_size**2)`, *optional*):
            Flattened patches of all images, concatenated, as returned by the image processor.
        image_grid_hw (`torch.LongTensor` of shape `(num_images, 2)`, *optional*):
            Patch grid `(height, width)` of each image in `pixel_values`.
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mel_bins)`, *optional*):
            Mel features produced by the processor, encoded and scattered onto the audio
            placeholder tokens.
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask marking the real mel frames of each padded clip.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        
        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )

        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )

        if mm_encoder_outputs.get("image") is not None:
            image_embeds = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_embeds, self.image_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if mm_encoder_outputs.get("video") is not None:
            video_embeds = torch.cat(mm_encoder_outputs["video"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_embeds, self.image_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if self.video_pruning_rate > 0:
                h = w = int(video_embeds.shape[1] ** 0.5)
                evs_mask = compute_retention_mask(
                    video_embeds=video_embeds,
                    thw=(video_embeds.shape[0], h, w),
                    spatial_merge_size=1,
                    q=self.video_pruning_rate,
                )
                retention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                retention_mask[input_ids == self.image_token_id] = evs_mask.view(-1)
                inputs_embeds = inputs_embeds[retention_mask].unsqueeze(0)
                if attention_mask is not None:
                    attention_mask = attention_mask[retention_mask].unsqueeze(0)
                input_ids = input_ids[retention_mask].unsqueeze(0)

        if input_features is not None and self.audio_tower is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_features, input_features_mask).pooler_output
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds, audio_embeds, self.audio_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
