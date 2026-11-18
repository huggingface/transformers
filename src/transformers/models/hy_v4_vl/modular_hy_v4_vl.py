# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
import itertools

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_utils import PILImageResampling, SizeDict
from ...masking_utils import create_causal_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import MultiModalData, Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import group_videos_by_shape, reorder_videos
from ...vision_utils import get_vision_pixel_shuffle_index
from ..ernie4_5.modeling_ernie4_5 import apply_rotary_pos_emb  # noqa: F401
from ..hunyuan_vl.configuration_hunyuan_vl import HunYuanVLVisionConfig
from ..hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor
from ..hunyuan_vl.image_processing_pil_hunyuan_vl import HunYuanVLImageProcessorPil
from ..hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLRMSNorm,
    HunYuanVLVisionAttention,
    HunYuanVLVisionPatchEmbed,
    HunYuanVLVisionPatchMerger,
    HunYuanVLVisionTransformer,
    get_vision_merger_layout_index,
)
from ..hy_v4.configuration_hy_v4 import HYV4Config
from ..hy_v4.modeling_hy_v4 import (
    HYV4Attention,
    HYV4Model,
    HYV4PreTrainedModel,
)
from ..mimo_v2_flash.modeling_mimo_v2_flash import eager_attention_forward  # noqa: F401
from ..qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from ..qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor, Qwen3VLVideoProcessorInitKwargs


@auto_docstring(checkpoint="tencent/Hy4")
@strict
class HYV4VLVisionConfig(HunYuanVLVisionConfig):
    r"""
    interpolate_mode (`str`, *optional*, defaults to `"bilinear"`):
        Interpolation mode for the learned patch positional embeddings.
    out_hidden_size (`int`, *optional*, defaults to 6144):
        Width of the visual features passed to the text backbone.
    max_image_size (`int`, *optional*, defaults to 2048):
        Image size used to define the learned positional embedding grid.
    """

    img_max_token_num = AttributeError()
    min_image_size = AttributeError()
    max_vit_seq_len = AttributeError()
    text_hidden_size = AttributeError()
    temporal_patch_size = AttributeError()

    model_type = "hy_v4_vl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 40
    out_hidden_size: int = 6144


@auto_docstring(checkpoint="tencent/Hy4")
@strict
class HYV4VLTextConfig(HYV4Config):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups for routing.
    topk_group (`int`, *optional*, defaults to 1):
        Number of expert groups kept during routing.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type, either `"dense"` or `"sparse"`. Defaults to one dense layer followed by sparse layers.
    index_topk (`int`, *optional*, defaults to 2048):
        Maximum number of key positions selected by each DSA query.
    index_head_dim (`int`, *optional*, defaults to 128):
        Hidden dimension of each DSA indexer head.
    index_n_heads (`int`, *optional*, defaults to 32):
        Number of DSA indexer heads.
    indexer_types (`list[str]`, *optional*):
        Per-layer DSA indexer type, either `"full"` or `"shared"` (reuses the previous full layer's selection).
    hc_mult (`int`, *optional*, defaults to 4):
        Number of hidden-state channels maintained by iHC.
    hc_magnitude (`float`, *optional*, defaults to 2.0):
        Scale applied to the iHC post-gating branch.
    hc_eps (`float`, *optional*, defaults to 1e-6):
        Numerical epsilon added to iHC sigmoid gates.
    learnable_sink_init (`float`, *optional*, defaults to 0.0):
        Initial value of each learned attention-sink logit.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Magnitude of the routed-expert SwiGLU clamp. Values at or below zero disable the clamp.
    """

    model_type = "hy_v4_vl_text"
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {"mrope_section"}

    hidden_size: int = 6144
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    q_lora_rank: int = 2048
    index_n_heads: int = 32
    output_router_logits: bool = False

    def convert_rope_params_to_dict(self, **kwargs):
        kwargs = PreTrainedConfig.convert_rope_params_to_dict(self, **kwargs)
        self.rope_parameters.setdefault("mrope_section", [12, 10, 10])
        return kwargs

    def validate_architecture(self):
        PreTrainedConfig.validate_architecture(self)

        mrope_section = self.rope_parameters["mrope_section"]
        if len(mrope_section) != 3 or sum(mrope_section) != self.qk_rope_head_dim // 2:
            raise ValueError(
                "`mrope_section` must have three entries (time, height, width) summing to `qk_rope_head_dim // 2`, "
                f"got {mrope_section} for `qk_rope_head_dim={self.qk_rope_head_dim}`."
            )


@auto_docstring(checkpoint="tencent/Hy4")
@strict
class HYV4VLConfig(PreTrainedConfig):
    model_type = "hy_v4_vl"
    sub_configs_defaults = {
        "vision_config": HYV4VLVisionConfig,
        "text_config": HYV4VLTextConfig,
    }
    keys_to_ignore_at_inference = ["cache"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 120120
    video_token_id: int = 120124
    tie_word_embeddings: bool = False


class HYV4VLAttention(HYV4Attention):
    def __init__(self, config: HYV4VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Latent norms use the checkpoint `rms_norm_eps`
        self.q_a_layernorm = HYV4VLRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = HYV4VLRMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)


@auto_docstring
class HYV4VLPreTrainedModel(HYV4PreTrainedModel):
    config: HYV4VLConfig
    input_modalities = ("image", "video", "text")
    _no_split_modules = ["HYV4VLDecoderLayer", "HYV4VLVisionBlock"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, HYV4VLVisionPatchMerger):
            init.normal_(module.image_newline, mean=0.0, std=module.config.out_hidden_size**-0.5)


class HYV4VLRMSNorm(HunYuanVLRMSNorm):
    pass


class HYV4VLVisionPatchEmbed(HunYuanVLVisionPatchEmbed):
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        patch_pos_embed = self.position_embedding.weight[1:, :].reshape(
            1, self.position_edge, self.position_edge, self.embed_dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2).float()
        # Key difference: an explicit `scale_factor` of `(size + 0.1) / edge`, which shifts the sampling grid
        scale_h = (int(height) + 0.1) / self.position_edge
        scale_w = (int(width) + 0.1) / self.position_edge
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(scale_h, scale_w),
            mode=self.config.interpolate_mode,
            align_corners=False,
        )
        return patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim).to(embeddings.dtype)


class HYV4VLVisionPatchMerger(HunYuanVLVisionPatchMerger):
    def __init__(self, config: HYV4VLVisionConfig):
        nn.Module.__init__(self)

        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.proj_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size * config.spatial_merge_size**2,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.proj_act = ACT2FN[config.hidden_act]
        # Key difference: `proj_out` projects directly to the text width. There is no `mlp`, `after_rms` or
        # image begin/end/sep embeddings.
        self.proj_out = nn.Conv2d(
            config.hidden_size * config.spatial_merge_size**2, config.out_hidden_size, kernel_size=1
        )
        self.image_newline = nn.Parameter(torch.empty(config.out_hidden_size))

        self.before_rms = HYV4VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.LongTensor, **kwargs) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        hidden_states = self.before_rms(hidden_states[0])
        dtype = hidden_states.dtype
        # Key difference: each video frame is merged on its own
        block_index = get_vision_pixel_shuffle_index(grid_thw, merge_size, kwargs=kwargs)
        blocks = hidden_states[block_index].view(-1, merge_size, merge_size, hidden_states.shape[-1])
        blocks = self.proj_out(self.proj_act(self.proj_conv(blocks.permute(0, 3, 1, 2)))).flatten(1)
        # Key difference: no begin and end tokens, nor projection to the text model
        tokens = torch.cat([blocks, self.image_newline[None].to(dtype)])
        layout_index = get_vision_merger_layout_index(grid_thw, merge_size, begin_end=False, kwargs=kwargs)
        return tokens[layout_index].unsqueeze(0)


class HYV4VLVisionAttention(HunYuanVLVisionAttention):
    def __init__(self, config: HYV4VLVisionConfig):
        super().__init__(config)
        # Both towers share `eager_attention_forward`, only the text attention has sinks
        self.sinks = None


class HYV4VLVisionModel(HunYuanVLVisionTransformer):
    config: HYV4VLVisionConfig
    # Only the text attention (sinks + DSA) is restricted to eager
    _supports_sdpa = True
    _supports_flash_attn = True


class HYV4VLRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    pass


class HYV4VLTextModel(HYV4Model):
    config: HYV4VLTextConfig
    input_modalities = ("text",)

    def __init__(self, config: HYV4VLTextConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # As in Qwen2-VL, an optional leading text axis is used to build the causal mask
        text_position_ids = None
        if position_ids.shape[0] == 4:
            text_position_ids, position_ids = position_ids[0], position_ids[1:]

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": text_position_ids,
                "allow_is_causal_skip": False,  # Always force creation to account for causality in the indexer
            }
            causal_mask_mapping = {"indexed_attention": create_causal_mask(**mask_kwargs)}

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        # Prepare HC connections
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        topk_indices = None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping["indexed_attention"],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache,
        )


@auto_docstring
class HYV4VLModel(Qwen2VLModel):
    def __init__(self, config: HYV4VLConfig):
        super().__init__(config)
        self.vision_tower = HYV4VLVisionModel(config.vision_config)
        del self.visual

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: float | torch.Tensor = 1.0,
        device: str | torch.device | None = None,
    ):
        """
        Compute 3D positional indices for vision tokens derived from a single image or video input.

        Each frame is laid out as `H` rows of `W + 1` tokens, as every merged row ends with a newline token. All three
        axes of a frame start at the same offset, and every frame starts a new block of `max(H, W + 1)` positions.
        The resulting positions are offset by `start_position`.

        Args:
            start_position (`int`):
                Offset added to all computed positional indices.
            grid_thw (`Sequence[int]` or `torch.Tensor` of shape `(3,)`):
                The (T, H, W) grid representing the feature layout of the current image or video after patch embedding.
            temp_merge_size (`int`, *optional*):
                Factor by which the temporal dimension is reduced in the backbone. The temporal grid size is divided
                by this value. Defaults to 1.
            spatial_merge_size (`int`, *optional*):
                Factor by which the spatial dimensions (H and W) are reduced in the backbone. Both H and W are divided
                by this value. Defaults to 1.
            time_interval (`float` or scalar `torch.Tensor`, *optional*, defaults to 1.0):
                Unused, kept for compatibility with Qwen2-VL: frames are spaced by the block size instead.
            device (`str` or `torch.device`, *optional*):
                Device on which the resulting tensor is allocated. If `None`, uses the current default device.

        Returns:
            torch.LongTensor of shape (3, T * H * (W + 1)):
                Positional indices for temporal, height, and width dimensions,
                flattened into sequence form and offset by `start_position`.
        """
        llm_grid_t, llm_grid_h, llm_grid_w = (
            grid_thw[0].item() // temp_merge_size,
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size,
        )

        # Key differences: every row ends with a newline token (`W + 1` columns) and every frame starts a new block,
        # so all three axes are offset by the frame index times the block size
        position_temporal = torch.arange(llm_grid_t, device=device) * max(llm_grid_h, llm_grid_w + 1)
        position_height = torch.arange(llm_grid_h, device=device)
        position_width = torch.arange(llm_grid_w + 1, device=device)

        T_grid, H_grid, W_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
        vision_position_ids = torch.stack([T_grid, H_grid + T_grid, W_grid + T_grid], dim=0).reshape(3, -1)
        vision_position_ids += start_position
        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's sizes. The utility expects a `vision + text`
        sequence and will error out otherwise. For pure text sequence, please rely on model's auto-inferred
        position ids. In a mixed vision + text sequence, vision tokens use 3D RoPE (temporal, height, width)
        while text tokens use standard 1D RoPE.

        Example:
            Temporal patches: 2; Height patches: 2; Width patches: 2 (after merging)
            Each frame has a newline token at the end of every row, so it is a 2 x 3 block of 6 tokens and the video
            has 2 x 6 = 12 tokens. With `start_position = 0`, each frame starts a new block of `max(2, 3) = 3`
            positions:

            Temporal IDs: `[0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3]`
            Height IDs:   `[0, 0, 0, 1, 1, 1, 3, 3, 3, 4, 4, 4]`
            Width IDs:    `[0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]`

            The next text token continues at `max(ids) + 1 = 6` and text positions grow with a step of `1`.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                # text == 0
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                # image == 1, video == 2
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    # Key difference: the next token starts after the last frame's block, including the newline column
                    current_pos = vision_position_ids.max().item() + 1
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    @can_return_tuple
    @auto_docstring
    def get_image_embeddings(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        vision_outputs = self.vision_tower(pixel_values.to(self.vision_tower.dtype), grid_thw=image_grid_thw, **kwargs)
        vision_outputs.pooler_output = vision_outputs.pooler_output.squeeze(0)
        merge_size = self.config.vision_config.spatial_merge_size
        frames, height, width = image_grid_thw.unbind(-1)
        vision_outputs.split_sizes = frames * (height // merge_size) * (width // merge_size + 1)
        return vision_outputs

    @can_return_tuple
    @auto_docstring
    def get_video_embeddings(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.get_image_embeddings(pixel_values_videos, video_grid_thw, return_dict=True, **kwargs)


@auto_docstring
class HYV4VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keep_in_fp32_modules_strict = ["lm_head"]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> import os
        >>> from transformers import AutoProcessor, DistributedConfig, HYV4VLForConditionalGeneration

        >>> model_path = "tencent/Hy4"  # Tentative release identifier.
        >>> world_size = int(os.environ["WORLD_SIZE"])
        >>> processor = AutoProcessor.from_pretrained(model_path)
        >>> model = HYV4VLForConditionalGeneration.from_pretrained(
        ...     model_path, distributed_config=DistributedConfig(tp_size=world_size, ep_size=world_size),
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": "path/to/your/image.jpg"},
        ...             {"type": "text", "text": "Describe this image."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, add_generation_prompt=True, reasoning_effort="no_think",
        ...     return_tensors="pt", return_dict=True,
        ... ).to(model.device)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        >>> generated_trimmed = generated_ids[0][inputs["input_ids"].shape[-1]:]
        >>> print(processor.decode(generated_trimmed, skip_special_tokens=True))
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            mm_encoder_outputs=mm_encoder_outputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        # Key difference: the LM head is kept in fp32
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


@auto_docstring
class HYV4VLImageProcessor(HunYuanVLImageProcessor):
    pass


@auto_docstring
class HYV4VLImageProcessorPil(HunYuanVLImageProcessorPil):
    pass


class HYV4VLVideoProcessorInitKwargs(Qwen3VLVideoProcessorInitKwargs, total=False):
    r"""
    size (`dict[str, int]`, *optional*):
        Total clip pixel budget, given by the required `longest_edge` key. A `shortest_edge` entry, which existing
        configs carry, is accepted but not used: frames are never upscaled.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    min_frames (`int`, *optional*, defaults to 2):
        The minimum number of frames to sample from a video.
    max_frames (`int`, *optional*, defaults to 1024):
        The maximum number of frames to sample from a video.
    max_image_size (`dict[str, int]`, *optional*):
        Per-frame pixel-area limit, specified by the `longest_edge` key.
    """

    cap_pixels_per_frame = AttributeError()
    max_video_tokens = AttributeError()
    max_image_size: dict[str, int]


@auto_docstring
class HYV4VLVideoProcessor(Qwen3VLVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 4096, "longest_edge": 33554432}
    max_image_size = {"longest_edge": 1024 * 1024}
    temporal_patch_size = 1
    min_frames = 2
    max_frames = 1024
    do_sample_frames = False
    cap_pixels_per_frame = AttributeError()
    max_video_tokens = AttributeError()
    valid_kwargs = HYV4VLVideoProcessorInitKwargs

    @staticmethod
    def smart_resize(
        frame_height: int,
        frame_width: int,
        num_frames: int,
        *,
        factor: int,
        max_pixels: int = 33554432,
        max_image_size: int = 1024 * 1024,
    ) -> tuple[int, int]:
        """Return the largest aligned ``(height, width)`` for a frame that fits every budget.

        ``factor`` is the merged patch edge in pixels (``patch_size * merge_size``). A frame of ``rows x cols``
        merged patches costs ``rows * (cols + 1)`` tokens, one newline token per row, which is the same accounting the
        processor uses. The result never upscales, keeps each frame within ``max_image_size`` pixels, and keeps
        ``num_frames * rows * (cols + 1)`` inside the visual token budget of ``max_pixels``. Calling it again on its
        own output returns the same size, so a frame that was already resized is left untouched.
        """
        if min(num_frames, frame_height, frame_width, factor, max_pixels, max_image_size) <= 0:
            raise ValueError("Video dimensions, resize factor, and pixel budgets must be positive.")
        if max(frame_height, frame_width) / min(frame_height, frame_width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(frame_height, frame_width) / min(frame_height, frame_width)}"
            )
        max_rows = max(1, frame_height // factor)
        max_cols = max(1, frame_width // factor)
        visual_budget = max_pixels // (factor * factor)
        for rows in range(max_rows, 0, -1):
            # On the ray scale * (height, width), rows == k spans cols floor(W*k/H) .. ceil(W*(k+1)/H) - 1;
            # rows clamp at 1, so the last step admits any narrower width.
            low_cols = 1 if rows == 1 else max(1, frame_width * rows // frame_height)
            high_cols = min(max_cols, max(1, -(-frame_width * (rows + 1) // frame_height) - 1))
            budget_cols = visual_budget // (num_frames * rows) - 1 if visual_budget > 0 else 0
            cap_cols = max_image_size // (factor * factor * rows)
            cols = min(high_cols, budget_cols, cap_cols)
            if cols >= low_cols:
                return rows * factor, cols * factor
        return factor, factor

    def resize(
        self,
        videos: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        factor: int,
        max_image_size: dict[str, int],
        **kwargs,
    ) -> "torch.Tensor":
        # Only the clip budget is needed: `shortest_edge` is not a lower bound, frames are never upscaled.
        if not size.longest_edge:
            raise ValueError(f"`size` dict must contain a 'longest_edge' key but got {size}.")
        height, width = self.smart_resize(
            frame_height=videos.shape[-2],
            frame_width=videos.shape[-1],
            num_frames=videos.shape[1],
            factor=factor,
            max_pixels=size.longest_edge,
            max_image_size=max_image_size["longest_edge"],
        )
        return BaseVideoProcessor.resize(
            self,
            image=videos,
            size=SizeDict(height=height, width=width),
            resample=resample,
        )

    def patchify(
        self,
        videos: "torch.Tensor",
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple["torch.Tensor", int, int, int]:
        batch_size, num_frames, channel, resized_height, resized_width = videos.shape
        if pad := -num_frames % temporal_patch_size:
            videos = torch.cat((videos, videos[:, -1:].expand(-1, pad, -1, -1, -1)), dim=1)
            num_frames += pad

        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = videos.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Key difference: row-major order within each merged block, as in the image processor
        patches = patches.permute(0, 1, 4, 5, 7, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches, grid_t, grid_h, grid_w

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_image_size: dict[str, int],
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_resize:
                stacked_videos = self.resize(
                    videos=stacked_videos,
                    size=size,
                    resample=resample,
                    factor=patch_size * merge_size,
                    max_image_size=max_image_size,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_t, grid_h, grid_w = self.patchify(
                stacked_videos,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            processed_videos_grouped[shape] = patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * len(stacked_videos)

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        return BatchFeature(
            data={
                "pixel_values_videos": torch.cat(processed_videos, dim=0),
                "video_grid_thw": torch.tensor(processed_grids),
            },
            tensor_type=return_tensors,
        )

    def get_num_of_video_patches(
        self, num_frames: int, height: int, width: int, videos_kwargs=None
    ) -> int | tuple[int, int, int, int]:
        """
        A utility that returns number of video patches a given video size.

        Args:
            num_frames (`int`):
                Number of frames in the input video.
            height (`int`):
                Height of the input video.
            width (`int`):
                Width of the input video.
            videos_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the video processor.
        Returns:
            `int`: Number of video patches per video, or `tuple[int, int, int, int]` with the number of patches and
            the `(grid_t, grid_h, grid_w)` grids.
        """
        videos_kwargs = videos_kwargs if videos_kwargs is not None else {}
        do_resize = videos_kwargs.get("do_resize") if videos_kwargs.get("do_resize") is not None else self.do_resize
        size = videos_kwargs.get("size", None) or self.size
        max_image_size = videos_kwargs.get("max_image_size", None) or self.max_image_size
        patch_size = videos_kwargs.get("patch_size", None) or self.patch_size
        merge_size = videos_kwargs.get("merge_size", None) or self.merge_size
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", None) or self.temporal_patch_size

        if do_resize:
            height, width = self.smart_resize(
                frame_height=height,
                frame_width=width,
                num_frames=num_frames,
                factor=patch_size * merge_size,
                max_pixels=size["longest_edge"],
                max_image_size=max_image_size["longest_edge"],
            )

        grid_t = (num_frames + -num_frames % temporal_patch_size) // temporal_patch_size
        grid_h, grid_w = height // patch_size, width // patch_size
        num_patches = grid_t * grid_h * grid_w

        return num_patches, grid_t, grid_h, grid_w


@auto_docstring
class HYV4VLProcessor(Qwen2VLProcessor):
    @staticmethod
    def _num_frame_tokens(grid_h: int, grid_w: int, merge_size: int) -> int:
        # One extra newline token per merged row
        return (grid_h // merge_size) * (grid_w // merge_size + 1)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        _, grid_h, grid_w = image_inputs["image_grid_thw"][image_idx]
        merge_size = kwargs.get("merge_size") or self.image_processor.merge_size
        return self.image_token * self._num_frame_tokens(grid_h, grid_w, merge_size)

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        grid_t, grid_h, grid_w = video_inputs["video_grid_thw"][video_idx]
        merge_size = kwargs.get("merge_size") or self.video_processor.merge_size
        return self.video_token * (grid_t * self._num_frame_tokens(grid_h, grid_w, merge_size))

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = {**kwargs, **kwargs.get("images_kwargs", {})}
            merge_size = images_kwargs.get("merge_size") or self.image_processor.merge_size
            grids = [
                self.image_processor.get_number_of_image_patches(height, width, images_kwargs)
                for height, width in image_sizes
            ]
            vision_data["num_image_patches"] = [grid_h * grid_w for grid_h, grid_w in grids]
            vision_data["num_image_tokens"] = [self._num_frame_tokens(*grid, merge_size) for grid in grids]
        if video_sizes is not None:
            videos_kwargs = {**kwargs, **kwargs.get("videos_kwargs", {})}
            merge_size = videos_kwargs.get("merge_size") or self.video_processor.merge_size
            num_video_tokens = []
            for num_frames, height, width in video_sizes:
                _, grid_t, grid_h, grid_w = self.video_processor.get_num_of_video_patches(
                    num_frames, height, width, videos_kwargs
                )
                num_video_tokens.append(grid_t * self._num_frame_tokens(grid_h, grid_w, merge_size))
            vision_data["num_video_tokens"] = num_video_tokens
        return MultiModalData(**vision_data)


__all__ = [
    "HYV4VLConfig",
    "HYV4VLVisionConfig",
    "HYV4VLTextConfig",
    "HYV4VLPreTrainedModel",
    "HYV4VLModel",
    "HYV4VLTextModel",
    "HYV4VLVisionModel",
    "HYV4VLForConditionalGeneration",
    "HYV4VLImageProcessor",
    "HYV4VLImageProcessorPil",
    "HYV4VLVideoProcessor",
    "HYV4VLProcessor",
]
