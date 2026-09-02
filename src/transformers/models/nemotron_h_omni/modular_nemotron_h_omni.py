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
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple, logging
from ..auto import AutoModel, AutoModelForCausalLM
from ..nemotron_h.modeling_nemotron_h import NemotronHRMSNorm
from .configuration_nemotron_h_omni import NemotronH_Omni_Reasoning_V3_Config


logger = logging.get_logger(__name__)


__all__ = ["NemotronH_Omni_Reasoning_V3"]


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


class NemotronH_Omni_Reasoning_V3SoundProjector(nn.Module):
    """Encode pre-extracted mel features with Parakeet and project them into LLM space, returning
    flattened embeddings ready to scatter onto `<audio>` positions. Mel extraction itself lives in
    the processor.
    """

    def __init__(self, config, llm_hidden_size: int):
        super().__init__()
        self.sound_encoder = AutoModel.from_config(config)
        self.sound_projection = NemotronH_Omni_Reasoning_V3MLP(
            config.hidden_size,
            config.projection_hidden_size,
            llm_hidden_size,
            bias=config.projection_bias,
        )

    def forward(self, input_features, input_features_mask=None) -> torch.Tensor:
        weight = self.sound_encoder.subsampling.linear.weight
        input_features = input_features.to(device=weight.device, dtype=weight.dtype)
        attention_mask = input_features_mask.to(weight.device) if input_features_mask is not None else None

        encoder_states = self.sound_encoder(
            input_features=input_features, attention_mask=attention_mask
        ).last_hidden_state
        sound_embeds = self.sound_projection(encoder_states.to(torch.bfloat16))

        # Multiple clips are batch-padded; keep only each clip's real (subsampled) length before flattening.
        if sound_embeds.dim() == 3 and sound_embeds.shape[0] > 1 and attention_mask is not None:
            lengths = self.sound_encoder._get_subsampling_output_length(attention_mask.sum(-1) + 1)
            positions = torch.arange(sound_embeds.shape[1], device=sound_embeds.device)
            return sound_embeds[positions[None, :] < lengths[:, None]]
        return sound_embeds.reshape(-1, sound_embeds.shape[-1])


class NemotronH_Omni_Reasoning_V3MLP(nn.Module):
    """Projector MLP: RMSNorm -> linear1 -> ReLU² -> linear2 (NemotronHMLP-style).

    Used both for the vision-to-LLM projector (`mlp1`) and the sound-to-LLM projection. The
    `linear1`/`linear2` submodule names match the Megatron sound-projection checkpoint structure
    (`sound_projection.{norm,linear1,linear2}.weight`); pass `bias=True` for the sound projection.
    """

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, bias: bool = False, eps: float = 1e-5
    ):
        super().__init__()
        self.norm = NemotronH_Omni_RMSNorm(in_features, eps=eps)
        self.linear1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.linear2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.act_fn = ACT2FN["relu2"]

    def forward(self, x):
        return self.linear2(self.act_fn(self.linear1(self.norm(x))))


class NemotronH_Omni_Reasoning_V3VisionProjector(nn.Module):
    """Project vision-tower features into LLM space: pixel-shuffle downsampling then `mlp1`.

    Shared by the image and the temporally-packed video path, which differ only in how the tower
    is run; both hand their `(num_patches, height * width, vit_hidden_size)` features to `forward`.
    """

    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config.downsample_ratio
        # Mirror Megatron training behavior: when the language model has MTP,
        # Megatron-Core's TransformerBlock places a final LayerNorm with the last
        # layer of every block built from the (shared) config -- including the
        # vision tower, whose config inherits `mtp_num_layers` from the language
        # model. The vision tower output is therefore layer-normalized before the
        # projector. LayerNorm is applied per token, so normalizing the RADIO
        # features here is exactly equivalent to Megatron's block-internal
        # placement before class-token stripping.
        # CODEPATH: only checkpoints whose language model carries MTP layers build this norm.
        if (getattr(config.llm_config, "num_nextn_predict_layers", 0) or 0) > 0:
            self.vision_final_layernorm = nn.LayerNorm(config.vit_hidden_size, eps=config.vision_config.layer_norm_eps)
        else:
            self.vision_final_layernorm = None
        self.mlp1 = NemotronH_Omni_Reasoning_V3MLP(
            config.vit_hidden_size * int(1 / config.downsample_ratio) ** 2,
            config.projector_hidden_size,
            config.llm_config.hidden_size,
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
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, vit_embeds, height, width):
        if self.vision_final_layernorm is not None:
            vit_embeds = self.vision_final_layernorm(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], height, width, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return self.mlp1(vit_embeds)


class NemotronH_Omni_Reasoning_V3(PreTrainedModel, GenerationMixin):
    config_class = NemotronH_Omni_Reasoning_V3_Config
    main_input_name = "input_ids"
    _supports_flash_attn_2 = True
    _supports_flash_attn = True

    def __init__(self, config: NemotronH_Omni_Reasoning_V3_Config):
        super().__init__(config)

        self.img_context_token_id = config.img_context_token_id

        self.language_model = AutoModelForCausalLM.from_config(config.llm_config)
        self.vision_model = AutoModel.from_config(config.vision_config)
        self.vision_model.make_preprocessor_external()

        llm_hidden_size = config.llm_config.hidden_size

        self.video_pruning_rate = config.video_pruning_rate
        self.video_temporal_patch_dim = config.video_temporal_patch_size

        self.vision_projector = NemotronH_Omni_Reasoning_V3VisionProjector(config)

        self.sound_context_token_id = config.sound_context_token_id
        # CODEPATH: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` ships a `sound_config` and
        # takes the audio branch.
        if config.sound_config is not None:
            self.sound_projector = NemotronH_Omni_Reasoning_V3SoundProjector(config.sound_config, llm_hidden_size)
        else:
            self.sound_projector = None

        self.all_tied_weights_keys = {}

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_image_features(self, pixel_values, image_flags=None, **kwargs):
        r"""
        image_flags (`torch.LongTensor` of shape `(num_images, 1)`, *optional*):
            Marks which projected image tiles are real; tiles flagged with `0` are dropped before
            the features are scattered onto the placeholder tokens.
        """
        # `pixel_values` is a list when tiles of differing sizes are batched; each runs the tower
        # separately and their projected tokens are concatenated.
        tiles = list(pixel_values) if isinstance(pixel_values, (list, tuple)) else [pixel_values]
        last_hidden_states, image_embeds = [], []
        for tile in tiles:
            tile = tile.to(dtype=self.vision_model.config.torch_dtype)
            vision_outputs = self.vision_model(tile, **kwargs)
            height, width = tile.shape[-2:]
            patch_size = self.vision_model.patch_size
            last_hidden_states.append(vision_outputs.last_hidden_state)
            image_embeds.append(
                self.vision_projector(vision_outputs.features, height // patch_size, width // patch_size)
            )

        image_embeds = torch.cat(image_embeds, dim=0)
        if image_flags is not None:
            image_embeds = image_embeds[image_flags.squeeze(-1) == 1]
        return BaseModelOutputWithPooling(
            last_hidden_state=torch.cat(last_hidden_states, dim=0),
            pooler_output=image_embeds,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @can_return_tuple
    @auto_docstring
    def get_video_features(self, pixel_values_videos, **kwargs):
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
        vision_outputs = self.vision_model(packed, use_video_patch_projection=True, **kwargs)
        patch_size = self.vision_model.patch_size
        return BaseModelOutputWithPooling(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=self.vision_projector(vision_outputs.features, height // patch_size, width // patch_size),
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_flags: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        inputs_embeds=None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, num_frames)`, *optional*):
            Mel features produced by the processor, encoded and scattered onto the audio
            placeholder tokens.
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask marking the real mel frames of each padded clip.
        image_flags (`torch.LongTensor` of shape `(num_images, 1)`, *optional*):
            Marks which projected image tiles are real; tiles flagged with `0` are dropped before
            the features are scattered onto the placeholder tokens.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Multimodal merges need `input_ids` to locate the placeholder tokens; skip them on the
        # cached decode steps and the inputs-embeds-only path where `input_ids` is absent.
        if pixel_values is not None and input_ids is not None:
            image_embeds = self.get_image_features(pixel_values, image_flags=image_flags).pooler_output
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = (input_ids == self.img_context_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask.to(inputs_embeds.device), image_embeds)

        if pixel_values_videos is not None and input_ids is not None:
            video_embeds = self.get_video_features(pixel_values_videos).pooler_output
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = (input_ids == self.img_context_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask.to(inputs_embeds.device), video_embeds)

            if self.video_pruning_rate > 0:
                h = w = int(video_embeds.shape[1] ** 0.5)
                evs_mask = compute_retention_mask(
                    video_embeds=video_embeds,
                    thw=(video_embeds.shape[0], h, w),
                    spatial_merge_size=1,
                    q=self.video_pruning_rate,
                )
                retention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                retention_mask[input_ids == self.img_context_token_id] = evs_mask.view(-1)
                inputs_embeds = inputs_embeds[retention_mask].unsqueeze(0)
                if attention_mask is not None:
                    attention_mask = attention_mask[retention_mask].unsqueeze(0)
                input_ids = input_ids[retention_mask].unsqueeze(0)

        if input_features is not None and self.sound_projector is not None and input_ids is not None:
            sound_embeds = self.sound_projector(input_features, input_features_mask)
            sound_embeds = sound_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            sound_mask = (input_ids == self.sound_context_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(sound_mask.to(inputs_embeds.device), sound_embeds)

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
                logits=logits, labels=labels, vocab_size=self.config.llm_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        input_features_mask=None,
        image_flags=None,
        use_cache=True,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            input_features_mask=input_features_mask,
            image_flags=image_flags,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # The multimodal inputs are only merged on the prefill step; drop them once the cache is warm.
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["input_features"] = None
            model_inputs["input_features_mask"] = None
            model_inputs["image_flags"] = None

        return model_inputs
