# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen2.5Omni model (Audio, Image, Video)."""

import math
import operator
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import ConvTranspose1d, Parameter

from transformers.models.llama.modeling_llama import rotate_half
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLMLP,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionBlock,
)
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoderLayer
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding

from ...cache_utils import Cache, EncoderDecoderCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from ...utils.hub import cached_file
from .configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniBigVGANConfig,
    Qwen2_5OmniConfig,
    Qwen2_5OmniDiTConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniToken2WavConfig,
    Qwen2_5OmniVisionEncoderConfig,
)


if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb
else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


logger = logging.get_logger(__name__)

QWEN2_5OMNI_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`{config_class}`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# @add_start_docstrings(
#     "The bare Qwen2.5Omni Model outputting raw hidden-states without any specific head on top.",
#     QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniConfig"),
# )
class Qwen2_5OmniPreTrainedModel(PreTrainedModel):
    config_class = Qwen2_5OmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        # important: this ported version of Qwen2.5OmniThinker isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = self.config.init_std if hasattr(self.config, "init_std") else 0.02

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2_5OmniPreTrainedModelForConditionalGeneration(Qwen2_5OmniPreTrainedModel):
    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            min_dtype (`float`):
                The minimum value representable with the dtype `dtype`.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: List[int],
        grid_hs: List[int],
        grid_ws: List[int],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = (
            torch.arange(llm_grid_h, device=llm_grid_h.device)
            .view(1, -1, 1)
            .expand(len(t_index), -1, llm_grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(llm_grid_w, device=llm_grid_w.device)
            .view(1, 1, -1)
            .expand(len(t_index), llm_grid_h, -1)
            .flatten()
        )
        t_index = (
            torch.Tensor(t_index)
            .to(llm_grid_h.device)
            .view(-1, 1)
            .expand(-1, llm_grid_h * llm_grid_w)
            .flatten()
            .long()
        )
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)  # + 1 ) # 12.09 by malinhan
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def split_list_into_ranges(self, lst, interval):
        ranges = [[] for _ in range((max(lst) // interval) + 1)]
        for num in lst:
            index = num // interval
            ranges[index].append(num)
        return ranges

    def get_rope_index(
        self,
        raw_input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        raw_attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.LongTensor] = None,
        padding_side: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time corresponding to each temporal of feature of each video.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_index
        video_token_id = self.config.video_token_index
        audio_token_id = self.config.audio_token_index
        vision_end_token_id = self.config.vision_end_token_id
        audio_start_token_id = self.config.audio_start_token_id
        audio_end_token_id = self.config.audio_end_token_id
        position_id_per_seconds = self.config.position_id_per_seconds
        seconds_per_chunk = self.config.seconds_per_chunk

        if second_per_grids is None and video_grid_thw is not None:
            second_per_grids = torch.tensor(
                [1] * video_grid_thw.shape[0], dtype=torch.long, device=raw_input_ids.device
            )  # 每个gridt间隔1秒

        mrope_position_deltas = []
        position_ids = []
        input_ids = []
        attention_mask = []
        audio_idx = 0
        video_idx = 0
        image_idx = 0
        for i, src_item in enumerate(raw_input_ids):
            new_src_item = []
            new_attention_mask = []
            llm_pos_ids_list = []
            if padding_side is None:
                i_attention_mask = raw_attention_mask[i]
            else:
                i_attention_mask = raw_attention_mask[i][raw_attention_mask[i] == 1]
                src_item = src_item[raw_attention_mask[i] == 1]
            for idx in range(len(src_item)):
                start_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if src_item[idx] not in [audio_token_id, video_token_id, image_token_id]:
                    if src_item[idx] == vision_end_token_id and use_audio_in_video:
                        start_idx -= 1

                    new_src_item.append(src_item[idx])
                    new_attention_mask.append(i_attention_mask[idx])
                    llm_pos_ids = torch.LongTensor([start_idx]).to(raw_input_ids.device).expand(3, -1)
                    llm_pos_ids_list.append(llm_pos_ids)
                elif src_item[idx] == audio_token_id:
                    audio_seqlen = audio_seqlens[audio_idx]
                    place_num = ((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1
                    new_src_item.extend([audio_token_id] * place_num)
                    new_attention_mask.extend([i_attention_mask[idx]] * place_num)
                    llm_pos_ids = torch.arange(place_num, device=raw_input_ids.device).expand(3, -1) + start_idx
                    llm_pos_ids_list.append(llm_pos_ids)
                    audio_idx += 1
                elif src_item[idx] == image_token_id:
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                    llm_pos_ids = self.get_llm_pos_ids_for_vision(
                        start_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    llm_pos_ids_list.append(llm_pos_ids)
                    vision_seqlen = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    new_src_item.extend([image_token_id] * vision_seqlen)
                    new_attention_mask.extend([i_attention_mask[idx]] * vision_seqlen)
                    image_idx += 1
                elif src_item[idx] == video_token_id and not use_audio_in_video:
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t, device=raw_input_ids.device)
                        * second_per_grids[video_idx]
                        * position_id_per_seconds
                    ).long()
                    llm_pos_ids = self.get_llm_pos_ids_for_vision(
                        start_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    llm_pos_ids_list.append(llm_pos_ids)
                    vision_seqlen = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                    new_src_item.extend([video_token_id] * vision_seqlen)
                    new_attention_mask.extend([i_attention_mask[idx]] * vision_seqlen)
                    video_idx += 1
                else:
                    audio_seqlen = audio_seqlens[audio_idx]
                    vision_seqlen = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                    grid_t = video_grid_thw[video_idx][0]
                    grid_h = video_grid_thw[video_idx][1]
                    grid_w = video_grid_thw[video_idx][2]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                    t_index = (
                        torch.arange(grid_t).to(raw_input_ids.device)
                        * second_per_grids[video_idx]
                        * position_id_per_seconds
                    ).long()  # 每格对应的时间戳
                    t_index_split_chunk = self.split_list_into_ranges(
                        t_index, t_ntoken_per_chunk
                    )  # 每个chunk每格对应的时间戳

                    new_src_item.extend([audio_start_token_id])  # video的start已经加进去了
                    start_idx -= 1
                    llm_pos_ids_list.extend(
                        [torch.LongTensor([start_idx]).to(raw_input_ids.device).expand(3, -1)] * 1
                    )  # audio_bos
                    place_num = (((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1) + 2
                    pure_audio_len = place_num - 2
                    added_audio_len = 0
                    audio_llm_pos_ids_list = []
                    for t_chunk in t_index_split_chunk:
                        vision_ntoken_per_chunk = len(t_chunk) * grid_h * grid_w // (spatial_merge_size**2)
                        new_src_item.extend([video_token_id] * vision_ntoken_per_chunk)
                        vision_llm_pos_ids_list = self.get_llm_pos_ids_for_vision(
                            start_idx + 1,
                            video_idx,
                            spatial_merge_size,  ## + 1 : 12.09 by malinhan
                            t_chunk,
                            grid_hs,
                            grid_ws,
                        ).split(1, dim=1)
                        llm_pos_ids_list.extend(vision_llm_pos_ids_list)  # img_token
                        new_src_item.extend(
                            min(t_ntoken_per_chunk, pure_audio_len - added_audio_len) * [audio_token_id]
                        )
                        audio_start_idx = (
                            start_idx if len(audio_llm_pos_ids_list) == 0 else audio_llm_pos_ids_list[-1][0].item()
                        )
                        if min(t_ntoken_per_chunk, pure_audio_len - added_audio_len) > 0:
                            audio_llm_pos_ids_list = (
                                torch.arange(
                                    min(t_ntoken_per_chunk, pure_audio_len - added_audio_len),
                                    device=raw_input_ids.device,
                                ).expand(3, -1)
                                + audio_start_idx
                                + 1
                            ).split(1, dim=1)
                        else:
                            audio_llm_pos_ids_list = []
                        added_audio_len += min(t_ntoken_per_chunk, pure_audio_len - added_audio_len)
                        llm_pos_ids_list.extend(audio_llm_pos_ids_list)  # audio_token
                    if added_audio_len < pure_audio_len:
                        new_src_item.extend((pure_audio_len - added_audio_len) * [audio_token_id])
                        audio_llm_pos_ids_list = (
                            torch.arange(pure_audio_len - added_audio_len, device=raw_input_ids.device).expand(3, -1)
                            + llm_pos_ids_list[-1].max()
                            + 1
                        ).split(1, dim=1)
                        llm_pos_ids_list.extend(audio_llm_pos_ids_list)  # audio_token

                    llm_pos_ids_list.extend(
                        [torch.tensor([llm_pos_ids_list[-1].max() + 1] * 3, device=raw_input_ids.device).unsqueeze(1)]
                        * 1
                    )  # audio_eos
                    new_src_item.extend([audio_end_token_id])
                    new_attention_mask.extend([i_attention_mask[idx]] * (vision_seqlen + place_num))
                    audio_idx += 1
                    video_idx += 1

            position_ids.append(torch.cat(llm_pos_ids_list, dim=1))  # 3 b s
            input_ids.append(torch.tensor(new_src_item))  # b s
            attention_mask.append(torch.tensor(new_attention_mask))  # b s
            mrope_position_deltas.append(torch.cat(llm_pos_ids_list, dim=1).max() + 1 - len(src_item))
        if padding_side is not None:
            position_ids = self._pad_to_list_of_tensors_2d(position_ids, padding_value=1, padding_side=padding_side)
            input_ids = self._pad_to_list_of_tensors_1d(
                input_ids, padding_value=self.pad_token_id, padding_side=padding_side
            )
            attention_mask = self._pad_to_list_of_tensors_1d(
                attention_mask, padding_value=0, padding_side=padding_side
            )
        input_ids = torch.stack(input_ids, dim=0).to(raw_input_ids.device)
        attention_mask = torch.stack(attention_mask, dim=0).to(raw_input_ids.device)
        position_ids = torch.stack(position_ids, dim=1).to(raw_input_ids.device)
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

        return position_ids, mrope_position_deltas, input_ids, attention_mask

    def _pad_to_list_of_tensors_1d(self, tensor_list, padding_value=0, padding_side="left"):
        lengths = [len(tensor) for tensor in tensor_list]
        max_length = max(lengths)
        pad_len = [max_length - leng for leng in lengths]
        for idx in range(len(tensor_list)):
            if pad_len[idx] != 0:
                if padding_side == "left":
                    tensor_list[idx] = torch.cat(
                        [
                            torch.full(
                                size=[pad_len[idx]],
                                fill_value=padding_value,
                                dtype=tensor_list[idx].dtype,
                                device=tensor_list[idx].device,
                            ),
                            tensor_list[idx],
                        ],
                        dim=0,
                    )
                else:
                    tensor_list[idx] = torch.cat(
                        [
                            tensor_list[idx],
                            torch.full(
                                size=[pad_len[idx]],
                                fill_value=padding_value,
                                dtype=tensor_list[idx].dtype,
                                device=tensor_list[idx].device,
                            ),
                        ],
                        dim=0,
                    )
        return tensor_list

    def _pad_to_list_of_tensors_2d(self, tensor_list, padding_value=0, padding_side="left"):
        dim = tensor_list[0].shape[0]
        lengths = [tensor.shape[-1] for tensor in tensor_list]
        max_length = max(lengths)
        pad_len = [max_length - leng for leng in lengths]
        for idx in range(len(tensor_list)):
            if pad_len[idx] != 0:
                if padding_side == "left":
                    tensor_list[idx] = torch.cat(
                        [
                            torch.full(
                                size=[dim, pad_len[idx]],
                                fill_value=padding_value,
                                dtype=tensor_list[idx].dtype,
                                device=tensor_list[idx].device,
                            ),
                            tensor_list[idx],
                        ],
                        dim=-1,
                    )
                else:
                    tensor_list[idx] = torch.cat(
                        [
                            tensor_list[idx],
                            torch.full(
                                size=[dim, pad_len[idx]],
                                fill_value=padding_value,
                                dtype=tensor_list[idx].dtype,
                                device=tensor_list[idx].device,
                            ),
                        ],
                        dim=-1,
                    )
        return tensor_list


############################
#      Start Thinker       #
############################


@dataclass
class Qwen2_5OmniThinkerCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2.5OmniThinker causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        attention_mask (`torch.FloatTensor`, *optional*):
            Attentions mask, used to update attention mask and position_ids.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen2_5OmniAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[Qwen2_5OmniThinkerConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        seq_length, _ = hidden_states.size()

        # get query proj
        # query_states = self.q_proj(hidden_states)
        query_states = (hidden_states @ self.q_proj.weight.t()) + self.q_proj.bias

        query_states = query_states.reshape(seq_length, self.num_heads, -1)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states).reshape(seq_length, self.num_heads, -1)
            value_states = self.v_proj(current_states).reshape(seq_length, self.num_heads, -1)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)

        attention_mask = torch.full(
            [1, seq_length, key_states.shape[1]],
            torch.finfo(query_states.dtype).min,
            device=query_states.device,
            dtype=query_states.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_output = torch.matmul(attn_weights, value_states).transpose(0, 1).reshape(seq_length, self.embed_dim)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class Qwen2_5OmniAudioFlashAttention2(Qwen2_5OmniAudioAttention):
    """
    Qwen2.5OmniThinker flash attention module. This module inherits from `Qwen2_5OmniAudioAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "The `static` cache implementation is not compatible with `attn_implementation='flash_attention_2'`. "
                "Use `attn_implementation='sdpa'` in the meantime, and open an issue at https://github.com/huggingface/transformers"
            )
        # Qwen2.5OmniThinkerFlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError("Qwen2.5OmniThinkerFlashAttention2 attention does not support output_attentions")

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        seq_length, all_dim = hidden_states.size()
        query_states = (hidden_states @ self.q_proj.weight.t()) + (
            self.q_proj.bias if self.q_proj.bias is not None else 0
        )
        query_states = query_states.reshape(seq_length, self.num_heads, -1)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = (current_states @ self.k_proj.weight.t()) + (
                self.k_proj.bias if self.k_proj.bias is not None else 0
            )
            key_states = key_states.reshape(seq_length, self.num_heads, -1)
            value_states = (current_states @ self.v_proj.weight.t()) + (
                self.v_proj.bias if self.v_proj.bias is not None else 0
            )
            value_states = value_states.reshape(seq_length, self.num_heads, -1)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            query_states, key_states, value_states, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0
        )
        attn_output = attn_output.reshape(seq_length, all_dim)
        attn_output = (attn_output @ self.out_proj.weight.t()) + (
            self.out_proj.bias if self.out_proj.bias is not None else 0
        )

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2_5OmniAudioSdpaAttention(Qwen2_5OmniAudioAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions or layer_head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2_5OmniThinkerModel is using Qwen2_5OmniThinkerSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None. Falling back to the manual attention"
                ' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                cu_seqlens=cu_seqlens,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        seq_length, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states).reshape(seq_length, self.num_heads, -1)
            value_states = self.v_proj(current_states).reshape(seq_length, self.num_heads, -1)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        attention_mask = torch.zeros(
            [1, seq_length, key_states.shape[0]], device=query_states.device, dtype=torch.bool
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)

        # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
        # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(0, 1)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(seq_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value


QWEN2_5_OMNI_AUDIO_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniAudioAttention,
    "flash_attention_2": Qwen2_5OmniAudioFlashAttention2,
    "sdpa": Qwen2_5OmniAudioSdpaAttention,
}


class Qwen2_5OmniAudioEncoderLayer(Qwen2AudioEncoderLayer):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__(config)
        self.self_attn = QWEN2_5_OMNI_AUDIO_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = (hidden_states @ self.fc1.weight.t()) + (self.fc1.bias if self.fc1.bias is not None else 0)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = (hidden_states @ self.fc2.weight.t()) + (self.fc2.bias if self.fc2.bias is not None else 0)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OmniAudioEncoder(Qwen2_5OmniPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen2_5OmniAudioEncoderLayer`].

    Args:
        config: Qwen2_5OmniAudioEncoderConfig
    """

    config_class = Qwen2_5OmniAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen2_5OmniAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = nn.ModuleList([Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

            feature_lens: [B], torch.LongTensor , mel length

            aftercnn_lens : [B], torch.LongTensor , mel length after cnn

            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        feature_lens_accum = list(accumulate(feature_lens.tolist(), func=operator.add, initial=0))

        all_feature_chunks = []
        all_chunk_lengths = []
        for index_ in range(feature_lens.shape[0]):
            feature_length = feature_lens_accum[index_ + 1] - feature_lens_accum[index_]
            each_audio_split_list = input_features[
                :, feature_lens_accum[index_] : feature_lens_accum[index_ + 1]
            ].split(self.n_window * 2, dim=1)

            for each_audio_split in each_audio_split_list:
                each_split_embed = nn.functional.gelu(self.conv1(each_audio_split))
                each_split_embed = nn.functional.gelu(self.conv2(each_split_embed)).transpose_(0, 1)

                embed_pos = self.positional_embedding(each_split_embed.shape[0]).to(each_split_embed.dtype)
                all_feature_chunks.append(each_split_embed + embed_pos)
                all_chunk_lengths.append(each_split_embed.shape[0])
                feature_length -= self.n_window * 2

            if not (feature_length <= 0 and feature_length > -2 * self.n_window):
                raise ValueError()

        hidden_states = torch.cat(all_feature_chunks, dim=0)
        cu_seqlens = list(accumulate(all_chunk_lengths, func=operator.add, initial=0))
        cu_seqlens = torch.Tensor(cu_seqlens).to(device=hidden_states.device, dtype=torch.int32)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        tmp_hidden_states = []
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None and head_mask.size()[0] != (len(self.layers)):
            raise ValueError(
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        cu_seqlens,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        cu_seqlens,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                tmp_hidden_states.append(hidden_states)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        if output_hidden_states:
            encoder_states = encoder_states + (token_audio,)

        if not return_dict:
            return tuple(v for v in [token_audio, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=token_audio, hidden_states=encoder_states, attentions=all_attentions)

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class Qwen2_5OmniVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5OmniVisionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def _apply_rotary_pos_emb_flashatt(self, tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        tensor_ = tensor.float()
        cos = freqs.cos()  # .type_as(tensor_)
        sin = freqs.sin()  # .type_as(tensor_)
        output = apply_rotary_emb(tensor_, cos, sin).type_as(tensor)
        return output

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = self._apply_rotary_pos_emb_flashatt(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = self._apply_rotary_pos_emb_flashatt(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5OmniVisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


QWEN2_5_OMNI_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniVisionAttention,
    "flash_attention_2": Qwen2_5OmniVisionFlashAttention2,
    "sdpa": Qwen2_5OmniVisionSdpaAttention,
}


class Qwen2_5OmniVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__(config, attn_implementation)
        self.attn = QWEN2_5_OMNI_VISION_ATTENTION_CLASSES[attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5OmniVisionEncoder(Qwen2_5_VisionTransformerPretrainedModel):
    config_class = Qwen2_5OmniVisionEncoderConfig
    _no_split_modules = ["Qwen2_5OmniVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.blocks = nn.ModuleList(
            [Qwen2_5OmniVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.gate_proj.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.gate_proj.weight.device

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

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Modification here
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, rotary_pos_emb
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    rotary_pos_emb=rotary_pos_emb,
                )
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class Qwen2_5OmniRotaryEmbedding(Qwen2VLRotaryEmbedding):
    def __init__(self, config: Qwen2_5OmniThinkerConfig, device=None):
        super().__init__(config, device)


# It's same as `Qwen2_5_VLAttention`, but talker model's hidden_size isn't divisible by num_heads.
# Removes the value error as a workaround.
class Qwen2_5OmniAttention(Qwen2_5_VLAttention, nn.Module):
    def __init__(self, config: Qwen2_5OmniConfig, layer_idx: Optional[int] = None):
        nn.Module.__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     raise ValueError(
        #         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        #         f" and `num_heads`: {self.num_heads})."
        #     )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)


class Qwen2MLP(Qwen2_5_VLMLP):
    pass


@add_start_docstrings(
    "The bare Qwen2.5OmniThinker Model outputting raw hidden-states without any specific head on top.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniTextConfig"),
)
class Qwen2_5OmniThinkerModel(Qwen2_5_VLModel):
    config_class = Qwen2_5OmniTextConfig
    _no_split_modules = ["Qwen2_5OmniDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTextConfig):
        super().__init__(config)


QWEN2_5OMNITHINKER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`NewTaskModelProcessor`] uses
            [`SiglipImageProcessor`] for processing images).
        pixel_values_videos(`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`NewTaskModelProcessor`] uses
            [`SiglipImageProcessor`] for processing videos).
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`list(torch.FloatTensor)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """The Qwen2.5OmniThinker model which consists of a audio backbone and a language model.""",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniThinkerConfig"),
)
class Qwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config_class = Qwen2_5OmniThinkerConfig
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(
            config.audio_config, attn_implementation=config._attn_implementation
        )

        self.visual = Qwen2_5OmniVisionEncoder._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )

        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.post_init()

    @add_start_docstrings_to_model_forward(QWEN2_5OMNITHINKER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Qwen2_5OmniThinkerCausalLMOutputWithPast, config_class="Qwen2_5OmniThinkerConfig"
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        >>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            embeds_to_talker = inputs_embeds.clone()

            # 2. Merge text , audios , image and video
            if input_ids.shape[1] != 1:
                if input_features is not None:
                    audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                        audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                    )
                    feature_lens = (
                        audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                    )
                    audio_outputs = self.audio_tower(
                        input_features,
                        feature_lens=feature_lens,
                        aftercnn_lens=audio_feat_lengths,
                    )
                    audio_features = audio_outputs.last_hidden_state
                    if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                        raise ValueError("length of audio_features should match audio_output_lengths")
                    audio_mask = (input_ids == self.config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
                    embeds_to_talker = embeds_to_talker.masked_scatter(audio_mask, torch.zeros_like(audio_features))

                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.get_dtype())
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                    image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                    embeds_to_talker = embeds_to_talker.masked_scatter(image_mask, torch.zeros_like(image_embeds))

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                    video_mask = (input_ids == self.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                    embeds_to_talker = embeds_to_talker.masked_scatter(video_mask, torch.zeros_like(video_embeds))

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + ((embeds_to_talker, outputs[0])) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(embeds_to_talker, outputs.hidden_states),
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas, input_ids, attention_mask = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0

            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "input_features": input_features,
                "feature_attention_mask": feature_attention_mask,
                "audio_feature_lengths": audio_feature_lengths,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update attention_mask
        if getattr(outputs, "attention_mask", None) is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def get_rope_index(
        self,
        raw_input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        raw_attention_mask=None,
        use_audio_in_video=False,
        audio_seqlens=None,
        second_per_grids=None,
    ):
        return super().get_rope_index(
            raw_input_ids,
            image_grid_thw,
            video_grid_thw,
            raw_attention_mask,
            use_audio_in_video,
            audio_seqlens,
            second_per_grids,
            padding_side="left",
        )


############################
#       Start Talker       #
############################


@dataclass
class Qwen2_5OmniTalkerCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2.5OmniTalker causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        attention_mask (`torch.FloatTensor`, *optional*):
            Attentions mask, used to update attention mask and position_ids.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    rope_deltas: Optional[torch.LongTensor] = None
    thinker_reply_part: torch.FloatTensor = None


@add_start_docstrings(
    "The bare Qwen2.5OmniTalker Model outputting raw hidden-states without any specific head on top.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniTalkerConfig"),
)
class Qwen2_5OmniTalkerModel(Qwen2_5_VLModel):
    config_class = Qwen2_5OmniTalkerConfig
    _no_split_modules = ["Qwen2_5OmniTalkerDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_size, self.padding_idx)


class Qwen2_5OmniTalkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config_class = Qwen2_5OmniTalkerConfig

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)

        self.thinker_to_talker_proj = nn.Linear(config.embedding_size, config.hidden_size)

        self.model = Qwen2_5OmniTalkerModel(config)
        self.codebook_size = config.vocab_size
        self.codec_head = nn.Linear(config.hidden_size, self.codebook_size, bias=False)

        self.codec_bos_token = config.tts_codec_start_token_id
        self.codec_eos_token = config.tts_codec_end_token_id
        self.codec_pad_token = config.tts_codec_pad_token_id
        self.codec_mask_token = config.tts_codec_mask_token_id

        self.text_bos_token = config.tts_text_start_token_id
        self.text_eos_token = config.tts_text_end_token_id
        self.text_pad_token = config.tts_text_pad_token_id

        self.spatial_merge_size = self.config.spatial_merge_size

        self.post_init()

    @add_start_docstrings_to_model_forward(QWEN2_5OMNITHINKER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Qwen2_5OmniTalkerCausalLMOutputWithPast, config_class="Qwen2_5OmniTalkerConfig"
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen2_5OmniTalkerCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from transformers import AutoProcessor, Qwen2_5OmniTalkerForConditionalGeneration

        >>> model = Qwen2_5OmniTalkerForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

        >>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        >>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        >>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        >>> inputs = processor(text=prompt, audios=audio, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Generate the caption in English: Glass is breaking."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. 推理第 2 个以及之后的 token
            codec_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = codec_embeds + thinker_reply_part[:, :1, :]
            if thinker_reply_part.shape[1] > 1:
                thinker_reply_part = thinker_reply_part[:, 1:, :]

        talker_lm_input = self.thinker_to_talker_proj(inputs_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=talker_lm_input,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.codec_head(hidden_states)
        logits = logits.float()

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniTalkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            rope_deltas=rope_deltas,
            thinker_reply_part=thinker_reply_part,
        )

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs and not self.config.is_encoder_decoder:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        elif "decoder_inputs_embeds" in model_kwargs and self.config.is_encoder_decoder:
            cache_position = (
                torch.ones_like(model_kwargs["decoder_inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            )
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    # prepare inputs for talker lm generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_text_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        thinker_reply_part=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_audio_features=None,
        audio_feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas, tmp_ids, attention_mask = self.get_rope_index(
                    input_text_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )

                inputs_embeds[:, -1, :] += self.get_input_embeddings()(
                    torch.LongTensor([self.codec_bos_token]).to(inputs_embeds.device)
                )
                inputs_embeds[:, -2, :] += self.get_input_embeddings()(
                    torch.LongTensor([self.codec_pad_token]).to(inputs_embeds.device)
                )
                # rope_deltas = rope_deltas * 0

            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {
                "inputs_embeds": inputs_embeds,
                "input_ids": None,
                "thinker_reply_part": thinker_reply_part,
            }
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None, "thinker_reply_part": thinker_reply_part}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.codec_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update attention_mask
        if getattr(outputs, "attention_mask", None) is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        if getattr(outputs, "thinker_reply_part", None) is not None:
            model_kwargs["thinker_reply_part"] = outputs.thinker_reply_part

        return model_kwargs


############################
#      Start Token2Wav     #
############################


# Using custom RoPE, will use LlamaRotaryEmbedding next version
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", t.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = freqs.reshape(*freqs.shape[:-2], -1)
        freqs = freqs.repeat(batch_size, *([1] * freqs.dim()))

        return freqs.cos(), freqs.sin()


# Modified from Llama with a different rotate function, will fixed in next release
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    def rotate_half_codec(x):
        # x = rearrange(x, "... (d r) -> ... d r", r=2)
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (rotate_half_codec(k) * sin)
    return q_embed, k_embed


class TDNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class Res2NetBlock(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    in_channels : int
        The number of channels expected in the input.
    out_channels : int
        The number of output channels.
    scale : int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the Res2Net block.
    dilation : int
        The dilation of the Res2Net block.
    """

    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.
    """

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        """Creates a binary mask for each sequence.

        Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

        Arguments
        ---------
        length : torch.LongTensor
            Containing the length of each sequence in the batch. Must be 1D.
        max_len : int
            Max length for the mask, also the size of the second dimension.
        dtype : torch.dtype, default: None
            The dtype of the generated mask.
        device: torch.device, default: None
            The device to put the mask variable.

        Returns
        -------
        mask : tensor
            The binary mask.
        """
        assert len(length.shape) == 1

        if max_len is None:
            max_len = length.max().long().item()  # using arange to generate mask
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, x):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = self._length_to_mask(lengths * L, max_len=L, dtype=x.dtype, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        total = mask.sum(dim=2, keepdim=True)

        mean, std = self._compute_statistics(x, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, L)
        std = std.unsqueeze(2).repeat(1, 1, L)
        attn = torch.cat([x, mean, std], dim=1)

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = self._compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

    def forward(self, x):
        residual = x

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)

        return x + residual


class ECAPA_TDNN(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    """

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        assert len(config.enc_channels) == len(config.enc_kernel_sizes)
        assert len(config.enc_channels) == len(config.enc_dilations)
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        # Final linear transformation
        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, x):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        for layer in self.blocks:
            x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x)

        # Final linear transformation
        x = self.fc(x)

        x = x.squeeze(-1)
        return x


class InputEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        self.proj = nn.Linear(
            config.mel_dim + config.enc_dim + config.enc_emb_dim + config.emb_dim,
            config.hidden_size,
        )
        self.spk_encoder = ECAPA_TDNN(config)

    def forward(self, x, spk, cond, code_embed, drop_audio_cond=False, code_embed_uncond=None, cfg=True):
        if cfg:
            x = torch.cat([x, x], dim=0)
            spk = torch.cat([spk, torch.zeros_like(spk)], dim=0)
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
            code_embed = torch.cat([code_embed, code_embed_uncond], dim=0)
        elif drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
            spk = torch.zeros_like(spk)
        cond = self.spk_encoder(cond).unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.proj(torch.cat((x, cond, code_embed, spk), dim=-1))

        return x


# Transformer backbone using DiT blocks
class CodecEmbedding(nn.Module):
    def __init__(self, codec_num_embeds, codec_dim, repeats):
        super().__init__()
        self.repeats = repeats
        self.codec_embed = nn.Embedding(codec_num_embeds + 1, codec_dim)

    def forward(self, code, drop_code=False):
        if drop_code:
            code = torch.zeros_like(code)
        code_embed = self.codec_embed(code)

        code_embed = torch.repeat_interleave(code_embed, repeats=self.repeats, dim=1)
        return code_embed


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation
class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation
class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.ff = nn.ModuleList(
            [
                nn.Linear(dim, inner_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim),
            ]
        )

    def forward(self, x):
        for layer in self.ff:
            x = layer(x)
        return x


class DiTAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()

        self.config = config
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.inner_dim = config.head_dim * config.num_attention_heads
        self.dropout = config.dropout
        self._attn_implementation = config._attn_implementation
        self.is_causal = False

        self.to_q = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_k = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_v = nn.Linear(config.hidden_size, self.inner_dim)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, config.hidden_size), nn.Dropout(config.dropout)])

    def forward(
        self,
        x,  # noised input x
        rope=None,  # rotary position embedding for x
        mask=None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # apply rotary position embedding
        # Due to training process, only first head is applied with RoPE, will be fixed at next release
        cos, sin = rope
        query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
        x, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=mask,
            is_causal=False,
        )

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        # x = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = self.to_out[0](x)
        # dropout
        x = self.to_out[1](x)

        return x


# time step conditioning embedding
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.type_as(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.ModuleList([nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)])

    def forward(self, timestep):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        for layer in self.time_mlp:
            time_hidden = layer(time_hidden)  # b d
        return time_hidden


class DiTBlock(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig, look_ahead_block=0, look_backward_block=0):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(config.hidden_size)

        self.attn = DiTAttention(config)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=config.hidden_size, mult=config.ff_mult, dropout=config.dropout)

    def forward(self, x, t, rope=None, block_diff=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(
            x=norm,
            rope=rope,
            mask=(block_diff >= -float(self.look_backward_block)) & (block_diff <= float(self.look_ahead_block)),
        )

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

        return out


class TorchActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


class AMPBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=self._get_padding(kernel_size, dilation[0]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=self._get_padding(kernel_size, dilation[1]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=self._get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
            ]
        )

        self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

        self.activations = nn.ModuleList(
            [TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def _get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x


@add_start_docstrings(
    "The full Qwen2.5Omni Token2WavBigVGAN model. Which take mel spectrogram as input and predict waveform.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniBigVGANConfig"),
)
class Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniBigVGANConfig

    def __init__(self, config: Qwen2_5OmniBigVGANConfig):
        super().__init__(config)

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        # pre conv
        self.conv_pre = nn.Conv1d(config.mel_dim, config.upsample_initial_channel, 7, 1, padding=3)

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        ConvTranspose1d(
                            config.upsample_initial_channel // (2**i),
                            config.upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    ]
                )
            )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock(ch, k, d))

        # post conv
        self.activation_post = TorchActivation1d(activation=SnakeBeta(ch))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def _normalize(self, S, max_abs_value, min_db):
        return torch.clamp(
            (2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value, -max_abs_value, max_abs_value
        )

    def _amp_to_db(self, x, min_level_db):
        min_level = np.exp(min_level_db / 20 * np.log(10))
        min_level = torch.ones_like(x) * min_level
        return 20 * torch.log10(torch.maximum(min_level, x))

    def apm_to_db(self, apm_mel):
        mel_spec = torch.exp(apm_mel)

        mel_spec = self._amp_to_db(mel_spec, -115) - 20
        mel_spec = self._normalize(mel_spec, 1, -115)

        return mel_spec

    def forward(self, apm_mel):
        mel_spec = self.apm_to_db(apm_mel)
        # pre conv
        hidden = self.conv_pre(mel_spec)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                hidden = self.ups[i][i_up](hidden)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](hidden)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](hidden)
            hidden = xs / self.num_kernels

        # post conv
        hidden = self.activation_post(hidden)
        hidden = self.conv_post(hidden)
        audio = torch.clamp(hidden, min=-1.0, max=1.0)  # bound the output to [-1, 1]

        return audio.squeeze().cpu()


class ODESolverRK4:
    def __init__(self, func, y0):
        self.func = func
        self.y0 = y0

        self._one_third = 1 / 3
        self._two_thirds = 2 / 3

    def _rk4_alt_step_func(self, func, t0, dt, t1, y0, f0=None):
        k1 = f0
        if k1 is None:
            k1 = func(t0, y0)
        k2 = func(t0 + dt * self._one_third, y0 + dt * k1 * self._one_third)
        k3 = func(t0 + dt * self._two_thirds, y0 + dt * (k2 - k1 * self._one_third))
        k4 = func(t1, y0 + dt * (k1 - k2 + k3))
        return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return self._rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0), f0

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)

    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1
            y0 = y1

        return solution


@add_start_docstrings(
    "The full Qwen2.5Omni Token2WavDiT model. Which take speech tokens as input and predict mel spectrogram.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniDiTConfig"),
)
class Qwen2_5OmniToken2WavDiTModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniDiTConfig
    _no_split_modules = ["DiTBlock"]

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__(config)
        self.mel_dim = config.mel_dim
        self.repeats = config.repeats
        self.time_embed = TimestepEmbedding(config.hidden_size)

        self.text_embed = CodecEmbedding(config.num_embeds, config.emb_dim, config.repeats)
        self.input_embed = InputEmbedding(config)

        self.rotary_embed = RotaryEmbedding(config.head_dim)
        # self.rotary_embed = Qwen2_5OmniDiTRotaryEmbedding(config)

        self.hidden_size = config.hidden_size
        self.layers = config.num_hidden_layers
        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads

        self.transformer_blocks = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.transformer_blocks.append(
                DiTBlock(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
            )

        self.norm_out = AdaLayerNormZero_Final(config.hidden_size)  # final modulation
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

    def _create_block_diff(self, x):
        batch, seq_len = x.shape[0], x.shape[1]
        block_indices = torch.arange(seq_len, device=x.device) // self.block_size  # [seq_length]

        block_i = block_indices.unsqueeze(1)  # [seq_length, 1]
        block_j = block_indices.unsqueeze(0)  # [1, seq_length]

        block_diff = block_j - block_i  # (n, n)

        return block_diff.expand(batch, self.num_attention_heads, seq_len, seq_len)

    def forward(
        self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
        drop_audio_cond=False,  # cfg for cond audio
        drop_code=False,  # cfg for code
        cfg=True,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (code + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        code_embed = self.text_embed(code, drop_code=False if cfg else drop_code)
        code_embed_uncond = self.text_embed(code, drop_code=True) if cfg else None
        hidden = self.input_embed(
            x,
            spk,
            cond,
            code_embed,
            drop_audio_cond=drop_audio_cond,
            code_embed_uncond=code_embed_uncond,
            cfg=cfg,
        )

        # rope = self.rotary_embed(x, torch.arange(seq_len, device=x.device).repeat(batch, 1))
        rope = self.rotary_embed(hidden)

        block_diff = self._create_block_diff(hidden)

        for block in self.transformer_blocks:
            hidden = block(hidden, t, rope=rope, block_diff=block_diff)

        hidden = self.norm_out(hidden, t)
        output = self.proj_out(hidden)

        return output

    @torch.no_grad()
    def sample(
        self,
        cond,
        ref_mel,
        code,
        steps=10,
        cfg_strength=0.5,
        sway_sampling_coef=-1.0,
    ):
        y_all = torch.randn([1, 30000, self.mel_dim], dtype=ref_mel.dtype)
        max_duration = code.shape[1] * self.repeats
        y0 = y_all[:, :max_duration].to(code.device)
        batch = ref_mel.shape[0]
        cond = cond.unsqueeze(1).repeat(1, max_duration, 1)
        assert batch == 1, "only support batch size = 1 currently"

        def fn(t, x):
            if cfg_strength < 1e-5:
                pred = self(x=x, spk=cond, cond=ref_mel, code=code, time=t, drop_audio_cond=False, drop_code=False)
                return pred

            out_put = self(x=x, code=code, spk=cond, cond=ref_mel, time=t, cfg=True)
            pred, null_pred = torch.chunk(out_put, 2, dim=0)

            return pred + (pred - null_pred) * cfg_strength

        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=code.device, dtype=cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        solver = ODESolverRK4(func=fn, y0=y0)
        trajectory = solver.integrate(t)

        generated = trajectory[-1]
        generated_mel_spec = generated.permute(0, 2, 1)
        return generated_mel_spec


@add_start_docstrings(
    "The full Qwen2.5Omni Token2Wav model. Consists a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.",
    QWEN2_5OMNI_START_DOCSTRING.format(config_class="Qwen2_5OmniToken2WavConfig"),
)
class Qwen2_5OmniToken2WavModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniToken2WavConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2_5OmniToken2WavDiTModel", "Qwen2_5OmniToken2WavBigVGANModel"]

    def __init__(self, config: Qwen2_5OmniToken2WavConfig):
        super().__init__(config)
        attn_impl = config._attn_implementation
        if config._attn_implementation == "flash_attention_2":
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel must inference with fp32, but flash_attention_2 only supports fp16 and bf16, "
                "attention implementation of Qwen2_5OmniToken2WavModel will fallback to sdpa."
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel does not support eager attention implementation, " "fall back to sdpa"
            )
            attn_impl = "sdpa"
        self.code2wav_dit_model = Qwen2_5OmniToken2WavDiTModel._from_config(
            config.dit_config, attn_implementation=attn_impl
        )
        self.code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )

    def forward(
        self,
        code,
        cond,
        ref_mel,
        steps=10,
        cfg_strength=0.5,
        sway_sampling_coef=-1.0,
        **kwargs,
    ):
        generated_mel = self.code2wav_dit_model.sample(
            cond,
            ref_mel,
            code,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )
        waveform = self.code2wav_bigvgan_model(generated_mel)
        return waveform


############################
#    Start Qwen2.5Omni     #
############################


@add_start_docstrings(
    """
    The full Qwen2.5Omni model, a multimodal model composed of 3 sub-models:
    - [`Qwen2_5OmniThinkerForConditionalGeneration`]:
    a causal auto-regressive transformer takes text, audio, image, video as input and predict text tokens.
    - [`Qwen2_5OmniTalkerForConditionalGeneration`]:
    a causal auto-regressive transformer takes thinker hidden states and response as input and predict speech tokens.
    - [`Qwen2_5OmniToken2WavModel`]:
    a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.
    """,
    QWEN2_5OMNI_START_DOCSTRING.format(config_class=Qwen2_5OmniConfig),
)
class Qwen2_5OmniModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniConfig
    _no_split_modules = [
        "Qwen2_5OmniTalkerForConditionalGeneration",
        "Qwen2_5OmniToken2WavModel",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)

        self.has_talker = config.enable_audio_output
        self.speaker_map = {}
        if config.enable_audio_output:
            self.enable_talker()

    def enable_talker(self):
        self.talker = Qwen2_5OmniTalkerForConditionalGeneration(self.config.talker_config)
        self.token2wav = Qwen2_5OmniToken2WavModel(self.config.token2wav_config)
        self.token2wav.float()
        self.has_talker = True

    def load_speakers(self, path):
        for key, value in torch.load(path).items():
            self.speaker_map[key] = value
        logger.info("Speaker {} loaded".format(list(self.speaker_map.keys())))

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "token2wav"):
            del self.token2wav
        self.has_talker = False

    @classmethod
    def can_generate(cls) -> bool:
        return True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        spk_path = cached_file(
            pretrained_model_name_or_path,
            "spk_dict.pt",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if spk_path is None:
            raise ValueError(f"""{pretrained_model_name_or_path}/{spk_path} not exists""")
        model.load_speakers(spk_path)

        return model

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.tensor] = None,
        spk: str = "Chelsie",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        r"""
        Generate text response and audio from input.

        Args:
            input_ids (`Optional[torch.Tensor]`, *optional*):
                Input ids, should obtain from processor.
            spk (`str` , defaults to "Chelsie"):
                Which speaker should be used in audio response.
            use_audio_in_video (`bool`, defaults to False):
                Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
            return_audio (`Optional[bool]`, *optional*):
                Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
            kwargs (*optional*):
                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
                thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
        Returns:
            When `return_audio=False`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
            When `return_audio=True`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
                - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
        """
        if spk not in self.speaker_map:
            raise ValueError(f"{spk} is not availible, availible speakers: {self.speaker_map.keys()}")
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initalized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")
        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value
        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value
        speaker_params = self.speaker_map[spk]

        # 1. Generate from thinker module
        thinker_result = self.thinker.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_hidden_states=True,
            **thinker_kwargs,
        )
        if not (return_audio and self.has_talker):
            return thinker_result.sequences

        # 2. Generate speech tokens from talker module
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :]
        thinker_token_embeds = [x[0] for x in thinker_result.hidden_states]
        thinker_hidden_states = [x[1][-1] for x in thinker_result.hidden_states]

        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids,
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.thinker.device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=self.talker.codec_mask_token),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=self.thinker.device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=self.thinker.device),
            ],
            dim=1,
        )

        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                self.thinker.get_input_embeddings()(
                    torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.thinker.device)
                ),
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                self.thinker.get_input_embeddings()(
                    torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=self.thinker.device)
                ),
                self.thinker.get_input_embeddings()(
                    torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=self.thinker.device)
                ),
            ],
            dim=1,
        )

        talker_attention_mask = torch.cat([kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1)

        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
            **talker_kwargs,
        )
        talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]

        # 3. Generate wavs from code
        if self.token2wav.dtype != torch.float:
            self.token2wav.float()
        wav = self.token2wav(
            talker_generate_codes,
            cond=speaker_params["cond"].to(talker_inputs_embeds).float(),
            ref_mel=speaker_params["ref_mel"].to(talker_inputs_embeds).float(),
            **token2wav_kwargs,
        )

        return thinker_result.sequences, wav.float()


__all__ = [
    "Qwen2_5OmniModel",
    "Qwen2_5OmniThinkerModel",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen2_5OmniTalkerModel",
    "Qwen2_5OmniTalkerForConditionalGeneration",
    "Qwen2_5OmniToken2WavDiTModel",
    "Qwen2_5OmniToken2WavBigVGANModel",
    "Qwen2_5OmniToken2WavModel",
    "Qwen2_5OmniPreTrainedModel",
    "Qwen2_5OmniPreTrainedModelForConditionalGeneration",
]
