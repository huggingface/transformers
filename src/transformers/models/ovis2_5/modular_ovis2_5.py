# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional

import torch
from torch import nn

from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, LlavaNextModelOutputWithPast
from ..ovis2.configuration_ovis2 import Ovis2Config
from ..ovis2.modeling_ovis2 import (
    Ovis2ForConditionalGeneration,
    Ovis2Model,
    Ovis2VisionModel,
    Ovis2VisualEmbeddingTable,
)
from ..qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..siglip2.modeling_siglip2 import Siglip2Encoder, Siglip2VisionEmbeddings, Siglip2VisionTransformer


class Ovis2_5ModelOutputWithPast(LlavaNextModelOutputWithPast):
    pass


class Ovis2_5CausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    pass


class Ovis2_5VisionConfig(PretrainedConfig):
    model_type = "ovis2_5"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        num_patches=-1,
        image_size=512,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_stride=2,
        window_size=112,
        fullatt_block_indexes=None,
        temporal_patch_size=1,
        preserve_original_pe=True,
        use_rope=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_stride = hidden_stride
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.temporal_patch_size = temporal_patch_size
        self.preserve_original_pe = preserve_original_pe
        self.use_rope = use_rope


class Ovis2_5TextConfig(Qwen3Config):
    model_type = "ovis2_5_text"


class Ovis2_5Config(Ovis2Config):
    model_type = "ovis2_5"
    sub_configs = {"vision_config": Ovis2_5VisionConfig, "text_config": Ovis2_5TextConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151665,
        visual_indicator_token_ids=[151666, 151667, 151668],
        vocab_size=151643,
        hidden_size=1536,
        **kwargs,
    ):
        super().__init__(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=image_token_id,
            visual_indicator_token_ids=visual_indicator_token_ids,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            **kwargs
        )
        # TODO: add video token id



class Ovis2_5VisionEmbeddings(Siglip2VisionEmbeddings):
    def __init__(self, config):
        self.image_size = config.image_size
        self.preserve_original_pe = config.preserve_original_pe
        self.hidden_stride = config.hidden_stride

        # siglip2 naflex
        if self.num_patches > 0:
            self.patch_embedding = nn.Linear(
                in_features=config.num_channels * self.patch_size * self.patch_size,
                out_features=self.embed_dim,
            )
            if self.preserve_original_pe:  # TODO: rename
                self.position_embedding_size = int(self.num_patches**0.5)
        else:
            self.patch_embedding = nn.Conv2d(
                in_channels=config.num_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                padding="valid",
            )
            if self.preserve_original_pe:
                self.num_patches = (self.image_size // self.patch_size) ** 2
                self.position_embedding_size = self.image_size // self.patch_size

        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor, grid_thws: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (num_patches, num_channels * temporal_patch_size * patch_size * patch_size)
            grid_thws: (`torch.LongTensor`):
                grid shape (num_patches, 3)
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        if isinstance(self.patch_embedding, nn.Linear):
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        elif isinstance(self.patch_embedding, nn.Conv2d):
            pixel_values = pixel_values.view(
                -1, self.config.num_channels * self.config.temporal_patch_size, self.patch_size, self.patch_size
            )
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            patch_embeds = patch_embeds.reshape(-1, self.embed_dim)

        if self.preserve_original_pe:
            assert grid_thws is not None
            pos_embed_new = torch.zeros_like(patch_embeds)
            ori_h = ori_w = self.position_embedding_size
            positional_embeddings = (
                self.position_embedding.weight.reshape(self.position_embedding_size, self.position_embedding_size, -1)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )
            # pos_embed = self.pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2)
            cnt = 0
            for t, h, w in grid_thws:
                thw = t * h * w
                pe = F.interpolate(positional_embeddings, size=(h, w), mode="bicubic", align_corners=False)
                pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
                pe = pe[0].repeat(t, 1)
                pe = pe.reshape(
                    t, h // self.hidden_stride, self.hidden_stride, w // self.hidden_stride, self.hidden_stride, -1
                )
                pe = pe.permute(0, 1, 3, 2, 4, 5).reshape(thw, -1)
                pos_embed_new[cnt : cnt + thw] = pe
                cnt += thw
            patch_embeds = patch_embeds + pos_embed_new

        return patch_embeds


class Ovis2_5VisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding):
    pass


class Ovis2_5VisionEncoder(Siglip2Encoder):
    def __init__(self, config):
        super().__init__()
        self.rotary_pos_emb = Ovis2_5VisionRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)
        self.patch_size = config.patch_size
        self.hidden_stride = config.hidden_stride
        self.window_size = config.window_size
        self.spatial_merge_unit = config.hidden_stride * config.hidden_stride
        self.fullatt_block_indexes = (
            None if config.fullatt_block_indexes is None else [int(i) for i in config.fullatt_block_indexes.split("|")]
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.hidden_stride // self.patch_size
        )  # patch (after merge) number in each window

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.hidden_stride,  # number of patch after merge
                grid_w // self.hidden_stride,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(
        self,
        inputs_embeds,
        grid_thws: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        rotary_pos_emb = self.rot_pos_emb(grid_thws)
        window_index, cu_window_seqlens = self.get_window_index(grid_thws)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=inputs_embeds.device,
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = inputs_embeds.size()
        inputs_embeds = inputs_embeds.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        inputs_embeds = inputs_embeds[window_index, :, :]
        inputs_embeds = inputs_embeds.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thws[:, 1] * grid_thws[:, 2], grid_thws[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        reverse_indices = torch.argsort(window_index)
        encoder_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        for index, block in enumerate(self.layers):
            if self.fullatt_block_indexes is None or index in self.fullatt_block_indexes:
                cu_seqlens_tmp = cu_seqlens
            else:
                cu_seqlens_tmp = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    block.__call__, hidden_states, cu_seqlens_tmp, position_embeddings
                )
            else:
                hidden_states = block(hidden_states, cu_seqlens_tmp, position_embeddings)
            if output_hidden_states:
                hidden_states_ = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
                encoder_states += (hidden_states_[reverse_indices, :].reshape(seq_len, -1),)
        # tokens = self.post_trunk_norm(tokens)
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :].reshape(seq_len, -1)

        return hidden_states, encoder_states


class Ovis2_5VisionTransformer(Siglip2VisionTransformer):
    def __init__(self, config):
        super().__init__()
        self.encoder = Ovis2_5VisionEncoder(config)
        del self.use_head


class Ovis2_5VisualEmbeddingTable(Ovis2VisualEmbeddingTable):
    pass


class Ovis2_5PreTrainedModel(PreTrainedModel):
    config: Ovis2_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Ovis2_5_VisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flex_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True


class Ovis2_5VisionModel(Ovis2VisionModel):
    def forward(self, pixel_values: torch.FloatTensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(pixel_values)
        last_hidden_state = outputs.last_hidden_state

        logits = self.head_linear(last_hidden_state)
        logits = self.head_norm(logits)

        prob_token = nn.functional.softmax(logits, dim=-1)

        return prob_token


class Ovis2_5Model(Ovis2Model):
    pass


@auto_docstring
class Ovis2_5ForConditionalGeneration(Ovis2ForConditionalGeneration, GenerationMixin):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_image_features(self, pixel_values: torch.FloatTensor):
        return self.model.get_image_features(pixel_values=pixel_values)


__all__ = ["Ovis2_5Config", "Ovis2_5PreTrainedModel", "Ovis2_5Model", "Ovis2_5ForConditionalGeneration"]
