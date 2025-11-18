# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from ...cache_utils import Cache
from ...processing_utils import Unpack
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ..internvl.configuration_internvl import InternVLConfig, InternVLVisionConfig
from ..internvl.modeling_internvl import (
    InternVLCausalLMOutputWithPast,
    InternVLModel,
    InternVLModelOutputWithPast,
    InternVLMultiModalProjector,
    InternVLPreTrainedModel,
    InternVLVisionAttention,
    InternVLVisionEmbeddings,
    InternVLVisionEncoder,
    InternVLVisionLayer,
    InternVLVisionMLP,
    InternVLVisionModel,
    InternVLVisionModelOutputWithPooling,
    InternVLVisionPatchEmbeddings,
    InternVLVisionPreTrainedModel,
    InternVLVisionRMSNorm
)
from ..llava.modeling_llava import (
    LlavaForConditionalGeneration,
)

import torch.nn as nn

class InternVLFlashMLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.dense_in = nn.Linear(in_dim, out_dim)
        self.act_fn = nn.GELU()
        self.dropout_in = nn.Dropout(dropout)
        self.dense_out = nn.Linear(out_dim, in_dim)
        self.dropout_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_dim)
        
    def forward(self, hidden_states):
        hidden_states = self.dense_in(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout_in(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout_out(hidden_states)
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

class InternVLFlashMLP2(nn.Module):
    def __init__(self, vit_hidden_size, llm_hidden_size, config):
        super().__init__()
        
        in_dim = vit_hidden_size * int(1 / config.downsample_ratio) ** 4
        mid_dim = llm_hidden_size * 2
        out_dim = llm_hidden_size
        self.norm = nn.LayerNorm(in_dim)
        self.dense1 = nn.Linear(in_dim, mid_dim)
        self.act_fn1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(mid_dim, mid_dim)
        self.act_fn2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.1)
        self.dense3 = nn.Linear(mid_dim, out_dim)
        
    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.act_fn1(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.act_fn2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = self.dense3(hidden_states)
        
        return hidden_states
    
class InternVLFlashGating(nn.Module):
    def __init__(self, hidden_size=2048, expansion_factor=4, dropout=0.1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        mid_dim = hidden_size * expansion_factor

        self.block1 = InternVLFlashMLP(hidden_size, mid_dim)
        self.block2 = InternVLFlashMLP(hidden_size, mid_dim)
        self.block3 = InternVLFlashMLP(hidden_size, mid_dim)
        self.block4 = InternVLFlashMLP(hidden_size, mid_dim)
        self.gate_norm = nn.LayerNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        x = x + self.block4(x)
        logits = self.gate_proj(self.gate_norm(x))
        probs = torch.softmax(logits, dim=-1)  # 每个 token 的 expert 选择概率
        return probs

class InternVLFlashCrossAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, dim))  # [1, D]
        self.attn1 = InternVLVisionAttention(InternVLVisionConfig)
        self.norm1 = nn.LayerNorm(dim)
        self.attn2 = InternVLVisionAttention(InternVLVisionConfig)
        self.norm2 = nn.LayerNorm(dim)
        self.attn3 = InternVLVisionAttention(InternVLVisionConfig)
        self.norm3 = nn.LayerNorm(dim)
        self.attn4 = InternVLVisionAttention(InternVLVisionConfig)
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, batched_tokens: list[torch.Tensor]):
        """
        batched_tokens: List of Tensors of shape [Ti, D], length = B
        """
        B = len(batched_tokens)
        if B == 0:
            return torch.empty(
                0, self.query_token.shape[-1], device=self.query_token.device, dtype=self.query_token.dtype
            )

        D = batched_tokens[0].shape[-1]
        device = batched_tokens[0].device
        # 1. Padding
        max_len = max(t.shape[0] for t in batched_tokens)
        dtype = self.query_token.dtype
        padded = torch.zeros(B, max_len, D, dtype=dtype, device=device)
        padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
        for i, t in enumerate(batched_tokens):
            L = t.shape[0]
            padded[i, :L] = t
            padding_mask[i, :L] = False
        # 2. Query token: [B, 1, D]
        query = self.query_token.unsqueeze(0).expand(B, -1, -1)  # learnable token for each sample
        # 3. Attention layers
        out1, _ = self.attn1(query, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out1 = self.norm1(out1)
        out2, _ = self.attn2(out1, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out2 = self.norm2(out2)
        out3, _ = self.attn3(out2, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out3 = self.norm3(out3)
        out4, _ = self.attn4(out3, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out4 = self.norm4(out4)
        return out4.squeeze(1)


class InternvlFlashVisionConfig(InternVLVisionConfig):
    pass


class InternvlFlashConfig(InternVLConfig):
    pass


@auto_docstring
class InternvlFlashVisionModel(InternVLVisionModel):
    pass



class InternvlFlashModel(InternVLModel):
    def __init__(self, config: InternvlFlashConfig):
        super().__init__(config)

        vit_hidden_size = config.vision_config.hidden_size
        self.pooling_before_gating = InternVLFlashCrossAttentionPooling(dim=vit_hidden_size)
        self.gating = InternVLFlashGating(hidden_size=vit_hidden_size)

        llm_hidden_size = config.text_config.hidden_size
        self.multi_modal_projector = InternVLMultiModalProjector(config)
        self.mlp2 = InternVLFlashMLP2(vit_hidden_size, llm_hidden_size, config)
        self.flash_relative_threshold = config.flash_relative_threshold
        self.flash_absolute_threshold = config.flash_absolute_threshold


    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        """Perform pixel shuffle downsampling on vision features.

        Args:
            vision_features (`torch.Tensor`):
                Input tensor of shape (batch_size, width, height, channels).
            scale_factor (`float`, *optional*, defaults to `0.5`):
                Factor by which to downsample. Default is 0.5, which halves the dimensions.

        Returns:
            vision_features (`torch.Tensor`):
                Downsampled tensor of shape (batch_size, height*scale_factor, width*scale_factor, channels/(scale_factor^2)).
        """
        batch_size, width, height, channels = vision_features.size()

        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError("Height and width must be divisible by scale_factor for proper downsampling.")

        # Reshape to allow downsampling
        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        # Permute dimensions to align downsampled axis correctly
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        # Reshape to achieve final downsampled dimensions
        vision_features = vision_features.view(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )

        # Swap height and width back for proper orientation
        vision_features = vision_features.permute(0, 2, 1, 3).contiguous()

        return vision_features

    def compress_visual_tokens_in_sentence(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        mask_idx: torch.Tensor,
        lengths: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        img_context_token_id: int,
        gate_result,
    ) -> tuple:
        N, C = input_embeds.shape

        keep_mask = torch.ones(N, dtype=torch.bool, device=input_embeds.device)

        delete_flags = torch.zeros(N, dtype=torch.int32, device=input_embeds.device)

        if (lengths % 256 != 0).any():
            raise ValueError(f"lengths % 256 != 0, lengths = {lengths}")
        
        block_counts = lengths // 256
        total_blocks = block_counts.sum()

        starts_expended = torch.repeat_interleave(starts, block_counts)
        global_range = torch.arange(total_blocks, device=input_embeds.device)
        cumsum_blocks = torch.cumsum(block_counts, dim=0)
        group_starts = torch.cat([torch.zeros(1, dtype=torch.long, device=lengths.device), cumsum_blocks[:-1]])
        local_block_indices = global_range - torch.repeat_interleave(group_starts, block_counts)

        all_block_starts = starts_expended + (local_block_indices * 256)

        compressed_starts = all_block_starts[gate_result]

        if compressed_starts.numel() > 0:
            offsets = torch.arange(64, 256, device=input_embeds.device)
            indices_to_remove = (compressed_starts.unsqueeze(1) + offsets.unsqueeze(0)).view(-1)

            keep_mask[indices_to_remove] = False
            delete_flags[indices_to_remove] = 1



        cumulative_deletes = torch.cumsum(delete_flags, dim=0)
        cumulative_deletes = torch.cat([cumulative_deletes, cumulative_deletes[-1:].clone()], dim=0)

        new_input_embeds = input_embeds[keep_mask.to(input_embeds.device), :]
        new_input_ids = input_ids[keep_mask.to(input_ids.device)]

        return new_input_embeds, new_input_ids, keep_mask

    def get_image_num_per_sample(
        self,
        input_ids: torch.Tensor,
    ):
        input_ids = input_ids.squeeze(0)  # (N,) #todo add batch size support
        selected = input_ids == self.config.image_token_id
        padded = torch.cat(
            [torch.tensor([0], device=selected.device), selected.int(), torch.tensor([0], device=selected.device)]
        )
        diff = torch.diff(padded)

        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]
        lengths = ends - starts

        return lengths, starts, ends

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        lengths: torch.Tensor,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`int` or `list[int]`):
                Layer index or list of layer indices to extract features from.
        Returns:
            vision_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`.
        """
        vit_embeds_1024 = self.vision_tower(pixel_values=pixel_values).last_hidden_state

        vit_embeds_1024 = vit_embeds_1024[:, 1:, :]
        h = w = int(vit_embeds_1024.shape[1] ** 0.5)
        vit_embeds_1024 = vit_embeds_1024.reshape(vit_embeds_1024.shape[0], h, w, -1)

        # begin moe
        lengths = [int(x) for x in lengths.tolist()]
        tile_splits = torch.split(vit_embeds_1024, lengths, dim=0)
        vit_embeds_1024_split_and_merge = [x.reshape(-1, x.shape[-1]) for x in tile_splits]

        gate = self.pooling_before_gating(vit_embeds_1024_split_and_merge)
        gate = self.gating(gate)

        vit_embeds_256 = vit_embeds_1024.clone()

        vit_embeds_64 = self.pixel_shuffle(vit_embeds_1024, scale_factor=self.config.downsample_ratio**2)
        vit_embeds_64 = vit_embeds_64.reshape(vit_embeds_64.shape[0], -1, vit_embeds_64.shape[-1])
        vit_embeds_64 = self.mlp2(vit_embeds_64)

        vit_embeds_256 = self.pixel_shuffle(vit_embeds_256, scale_factor=self.config.downsample_ratio)
        vit_embeds_256 = vit_embeds_256.reshape(vit_embeds_256.shape[0], -1, vit_embeds_256.shape[-1])
        vit_embeds_256 = self.multi_modal_projector(vit_embeds_256)

        relative_threshold_value = torch.quantile(
            gate_result[:, 0].to(torch.float32), self.flash_relative_threshold
        )
        gate_result = (gate_result[:, 0] > relative_threshold_value) & (
            gate_result[:, 0] >= self.flash_absolute_threshold
        )

        selected_embeds = []
        for i in range(gate_result.size(0)):
            if gate_result[i]:
                selected_embeds.append(vit_embeds_64[i])
            else:
                selected_embeds.append(vit_embeds_256[i])

        vit_embeds = torch.cat(selected_embeds, dim=0)

        return vit_embeds, gate_result

    def _scatter_image_embeddings(
            self,
            inputs_embeds: torch.Tensor,
            input_ids: torch.Tensor,
            vit_embeds: torch.Tensor,
        ) -> torch.Tensor:
            selected_mask = input_ids == self.config.image_token_id
            if selected_mask.sum() == 0:
                return inputs_embeds 
            inputs_embeds[selected_mask] = vit_embeds.to(inputs_embeds.device)
            return inputs_embeds

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, InternVLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        # image feature is vit embeds
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            lengths, starts, ends= self.get_image_num_per_sample(input_ids) / 256
            lengths_copy = lengths.clone()

            lengths_sum = torch.ones(int(lengths.sum().item()), dtype=torch.int64)
            lengths = lengths_sum.repeat_interleave(1)
            vit_embeds, gate_result = self.get_image_features(pixel_values, lengths)

            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)

            assert torch.all(attention_mask == 1)
            inputs_embeds, input_ids, keep_mask = self.compress_visual_tokens_in_sentence(
                input_embeds=inputs_embeds,
                input_ids=input_ids,
                img_context_token_id=self.config.image_token_id,
                gate_result=gate_result,
                lengths=lengths_copy,
                starts=starts,
                ends=ends,
            )

            attention_mask = attention_mask[:, keep_mask].to(inputs_embeds.device)

            inputs_embeds = self._scatter_image_embeddings(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                vit_embeds=vit_embeds,
            ).reshape(B, -1, C)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return InternVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=inputs_embeds if pixel_values is not None else None,
        )

class InternvlFlashForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: InternvlFlashConfig):
        super(LlavaForConditionalGeneration, self).__init__(config)
        self.model = InternvlFlashModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def forward(**super_kwargs):
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> torch_device = "cuda"
        >>> processor = AutoProcessor.from_pretrained("chenhaoguan/InternVL3_5-2B-Flash-hf")
        >>> model = AutoModelForImageTextToText.from_pretrained(
        ...     "chenhaoguan/InternVL3_5-2B-Flash-hf", dtype=torch.bfloat16, device_map=torch_device
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
        ...             },
        ...             {
        ...                 "type": "image",
        ...                 "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
        ...             },
        ...             {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)
        >>> generate_ids = model.generate(**inputs, max_new_tokens=200)
        >>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
        The images depict the Statue of Liberty and the Golden Gate Bridge.
        ```"""
        super().forward(**super_kwargs)


__all__ = [
    "InternvlFlashVisionConfig",
    "InternvlFlashConfig",
    "InternvlFlashVisionPreTrainedModel",
    "InternvlFlashVisionModel",
    "InternvlFlashPreTrainedModel",
    "InternvlFlashModel",
    "InternvlFlashForConditionalGeneration",
]
