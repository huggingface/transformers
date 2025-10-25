# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from ...cache_utils import Cache
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
)
from ..qwen2_vl.modeling_qwen2_vl import VisionMlp


logger = logging.get_logger(__name__)


# ------------------------- Configurations -------------------------


class LlavaOnevision1_5VisionConfig(Qwen2_5_VLVisionConfig):
    model_type = "llava_onevision1_5"

    def __init__(self, layer_norm_eps=1e-05, **super_kwargs):
        super().__init__(self, **super_kwargs)
        self.layer_norm_eps = layer_norm_eps


class LlavaOnevision1_5Config(Qwen2_5_VLConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaOnevision1_5Model`]. It is used to instantiate a
    LlavaOnevision1_5Model model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Llava-Onevision 1.5 [lmms-lab/LLaVA-OneVision-1.5-8B-Instruct](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3Config`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `LlavaOnevision1_5VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The token index to denote start of vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The token index to denote end of vision input.

    ```python
    >>> from transformers import LlavaOnevision1_5Model, LlavaOnevision1_5Config

    >>> # Initializing a LlavaOnevision1_5 style configuration
    >>> configuration = LlavaOnevision1_5Config()

    >>> # Initializing a model from the Llava-Onevision-1.5-8B style configuration
    >>> model = LlavaOnevision1_5Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_onevision1_5"
    sub_configs = {"vision_config": LlavaOnevision1_5VisionConfig, "text_config": AutoConfig}

    def __init__(self, text_config=None, **super_kwargs):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen3")
            self.sub_configs["text_config"] = CONFIG_MAPPING[text_config["model_type"]]
        elif text_config is None:
            self.sub_configs["text_config"] = CONFIG_MAPPING["qwen3"]
        super().__init__(self, **super_kwargs)


# ------------------------- Outputs -------------------------


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava-Onevision-1.5 outputs, with hidden states and attentions.
    """
)
class LlavaOnevision1_5ModelOutputWithPast(Qwen2_5_VLModelOutputWithPast):
    pass


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava-Onevision-1.5 causal language model (or autoregressive) outputs.
    """
)
class LlavaOnevision1_5CausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    pass


# ------------------------- Vision backbone (LlavaOnevision1_5Vision) -------------------------


class LlavaOnevision1_5VisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding):
    pass


class LlavaOnevision1_5VisionPatchEmbed(Qwen2_5_VisionPatchEmbed):
    def __init__(self, **super_kwargs):
        super().__init__(self, **super_kwargs)
        kernel_size = [self.patch_size, self.patch_size]
        self.proj = nn.Conv2d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class LlavaOnevision1_5VisionPatchMerger(Qwen2_5_VLPatchMerger):
    def __init__(self, context_dim: int, layer_norm_eps: float = 1e-05, **super_kwargs) -> None:
        super().__init__(self, **super_kwargs)
        self.ln_q = LayerNorm(context_dim, eps=layer_norm_eps)


class LlavaOnevision1_5VisionMlp(VisionMlp):
    pass


class LlavaOnevision1_5VisionAttention(Qwen2_5_VLVisionAttention):
    pass


class LlavaOnevision1_5VisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config: LlavaOnevision1_5VisionConfig) -> None:
        super().__init__(config)
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = LlavaOnevision1_5VisionAttention(config)
        self.mlp = LlavaOnevision1_5VisionMlp(
            dim=config.hidden_size, hidden_dim=config.intermediate_size, hidden_act=config.hidden_act
        )


class LlavaOnevision1_5PreTrainedModel(PreTrainedModel):
    config_class = LlavaOnevision1_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaOnevision1_5VisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, LlavaOnevision1_5VisionPretrainedModel):
            std_cls = float(module.config.hidden_size) ** -0.5
            torch.nn.init.normal_(module.class_embedding, mean=0.0, std=std_cls)
            torch.nn.init.normal_(module.class_pos_emb, mean=0.0, std=std_cls)


class LlavaOnevision1_5VisionPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    config_class = LlavaOnevision1_5VisionConfig

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        self.patch_embed = LlavaOnevision1_5VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = LlavaOnevision1_5VisionRotaryEmbedding(head_dim // 2)

        self.class_embedding = nn.Parameter(torch.ones(config.hidden_size))
        self.class_pos_emb = nn.Parameter(torch.ones(1, head_dim // 2))

        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.blocks = nn.ModuleList([LlavaOnevision1_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = LlavaOnevision1_5VisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu_long = cu_seqlens.to(torch.long)
        lengths = (cu_long[1:] - cu_long[:-1]).tolist()
        h_chunks = torch.split(hidden_states, lengths, dim=0)
        r_chunks = torch.split(rotary_pos_emb, lengths, dim=0)

        cls_h = self.class_embedding.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(0)
        cls_r = self.class_pos_emb.to(device=rotary_pos_emb.device, dtype=rotary_pos_emb.dtype)

        hidden_states = torch.cat([torch.cat([cls_h, h], dim=0) for h in h_chunks], dim=0)  # [N+S, Dh]
        rotary_pos_emb = torch.cat([torch.cat([cls_r, r], dim=0) for r in r_chunks], dim=0)  # [N+S, Dr]

        cu_long_with_cls = cu_long + torch.arange(cu_long.numel(), device=cu_long.device, dtype=cu_long.dtype)
        cu_seqlens = cu_long_with_cls.to(grid_thw.dtype if torch.jit.is_tracing() else torch.int32)

        hidden_states = self.pre_layernorm(hidden_states)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        lengths_with_cls = (cu_long_with_cls[1:] - cu_long_with_cls[:-1]).tolist()
        out_chunks = torch.split(hidden_states, lengths_with_cls, dim=0)
        hidden_states = torch.cat([c[1:] for c in out_chunks], dim=0)

        return self.merger(hidden_states)


# ------------------------- Top-level multi-modal Model -------------------------


@auto_docstring
class LlavaOnevision1_5Model(Qwen2_5_VLModel, PreTrainedModel):
    config: LlavaOnevision1_5Config
    _no_split_modules = ["LlavaOnevision1_5VisionBlock"]

    def __init__(self, config: LlavaOnevision1_5Config):
        PreTrainedModel.__init__(self, config)
        self.visual = LlavaOnevision1_5VisionPretrainedModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.rope_deltas = None
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaOnevision1_5ModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """
        if position_ids is not None:
            position_ids = position_ids[0] if len(position_ids.shape) == 3 else position_ids

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            if self.rope_deltas is None or cache_position is None or cache_position[0] == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)

        position_ids = position_ids[0] if len(position_ids.shape) == 3 else position_ids

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = LlavaOnevision1_5ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


@auto_docstring
class LlavaOnevision1_5ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaOnevision1_5ForConditionalGeneration

        >>> model = LlavaOnevision1_5ForConditionalGeneration.from_pretrained("Deep-VLM/LLaVA-OneVision-1.5-8B-Instruct-hf", trust_remote_code=True)
        >>> processor = AutoProcessor.from_pretrained("Deep-VLM/LLaVA-OneVision-1.5-8B-Instruct-hf", trust_remote_code=True)

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        super().forward(self, **super_kwargs)


__all__ = [
    "LlavaOnevision1_5Config",
    "LlavaOnevision1_5ForConditionalGeneration",
    "LlavaOnevision1_5Model",
    "LlavaOnevision1_5PreTrainedModel",
]
