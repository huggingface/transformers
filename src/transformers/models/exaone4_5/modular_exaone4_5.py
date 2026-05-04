# Copyright 2026 The LG AI Research and HuggingFace Inc. team. All rights reserved.
#
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
"""PyTorch EXAONE 4.5 model."""

from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import is_flash_attention_requested
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..exaone4.modeling_exaone4 import Exaone4PreTrainedModel
from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMLP,
    Qwen2_5_VLModel,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
)
from ..qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from ..qwen2_vl.modeling_qwen2_vl import (
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)


@auto_docstring(checkpoint="LGAI-EXAONE/EXAONE-4.5-33B")
@strict
class Exaone4_5_VisionConfig(Qwen2_5_VLVisionConfig):
    model_type = "exaone4_5_vision"
    base_config_key = "vision_config"
    num_key_value_heads: int = 8


@auto_docstring(checkpoint="LGAI-EXAONE/EXAONE-4.5-33B")
@strict
class Exaone4_5_Config(PreTrainedConfig):
    model_type = "exaone4_5"
    sub_configs = {"vision_config": AutoConfig, "text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 67
    video_token_id: int = 68
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "exaone4_5_vision")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["exaone4_5_vision"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "exaone4")
            # BC: EXAONE 4.5 first released with the text model type as `exaone4_5_text`, now changed to `exaone4`
            if self.text_config["model_type"] == "exaone4_5_text":
                self.text_config["model_type"] = "exaone4"
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["exaone4"]()

        super().__post_init__(**kwargs)


class Exaone4_5_PatchEmbed(Qwen2_5_VisionPatchEmbed):
    pass


class Exaone4_5_VisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding):
    pass


class Exaone4_5_PatchMerger(Qwen2_5_VLPatchMerger):
    pass


class Exaone4_5_VisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, config: Exaone4_5_VisionConfig):
        self.num_key_value_heads = config.num_key_value_heads
        super().__init__(config)
        del self.qkv
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.qkv = nn.Linear(self.dim, self.q_dim + (self.kv_dim * 2), bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        hidden_shape = (seq_length, -1, self.head_dim)

        query_states, key_states, value_states = self.qkv(hidden_states).split(
            [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
        )

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Exaone4_5_MLP(Qwen2_5_VLMLP):
    pass


class Exaone4_5_VisionBlock(Qwen2_5_VLVisionBlock):
    pass


class Exaone4_5_PreTrainedModel(Exaone4PreTrainedModel):
    config_class = Exaone4_5_Config
    _no_split_modules = ["Exaone4_5_VisionBlock", "Exaone4DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"mtp.*"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, Exaone4_5_VisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Exaone4_5_VisionModel(Exaone4_5_PreTrainedModel, Qwen2_5_VisionTransformerPretrainedModel):
    config_class = Exaone4_5_VisionConfig

    def __init__(self, config: Exaone4_5_VisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.patch_embed = Exaone4_5_PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Exaone4_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Exaone4_5_VisionBlock(config) for _ in range(config.depth)])
        self.merger = Exaone4_5_PatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False
        self.post_init()


class Exaone4_5_Model(Exaone4_5_PreTrainedModel, Qwen2_5_VLModel):
    def __init__(self, config: Exaone4_5_Config):
        super().__init__(config)
        self.visual = Exaone4_5_VisionModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Differ from Qwen: EXAONE 4.5 vision encoder uses 2D rotary positional embeddings (2D-RoPE)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Exaone4_5_ForConditionalGeneration(Exaone4_5_PreTrainedModel, Qwen2_5_VLForConditionalGeneration):
    """
    Main EXAONE 4.5 conditional generation class.

    Note: Unlike Qwen2VL, the EXAONE 4.5 vision encoder uses 2D rotary positional embeddings (2D-RoPE)
    and adopts a Grouped Query Attention (GQA) structure throughout the multimodal stack.
    """

    def _get_image_nums_and_video_nums(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns per-sample counts of image and video placeholder tokens.

        If `inputs_embeds` are provided, placeholder positions are inferred by comparing against
        the embedding vectors of `image_token_id` and `video_token_id`. Otherwise, counts are
        computed directly from `input_ids`.
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        if inputs_embeds is not None:
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        image_nums = torch.sum(image_mask, dim=1)
        video_nums = torch.sum(video_mask, dim=1)

        return image_nums, video_nums

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Exaone4_5_ForConditionalGeneration
        >>> import torch

        >>> model = Exaone4_5_ForConditionalGeneration.from_pretrained("LGAI-EXAONE/EXAONE-4.5-33B")
        >>> processor = AutoProcessor.from_pretrained("LGAI-EXAONE/EXAONE-4.5-33B")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Describe the image."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ... )
        >>> inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        >>> generated_ids = model.generate(**inputs, max_new_tokens=64)
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        # Force recomputation of 2D-RoPE and ignore rope_deltas
        model_inputs["position_ids"] = None
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        return model_inputs


class Exaone4_5_ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Exaone4_5_Processor(Qwen2_5_VLProcessor):
    pass


__all__ = [
    "Exaone4_5_Config",
    "Exaone4_5_ForConditionalGeneration",
    "Exaone4_5_Model",
    "Exaone4_5_PreTrainedModel",
    "Exaone4_5_Processor",
    "Exaone4_5_VisionModel",
    "Exaone4_5_VisionConfig",
]
