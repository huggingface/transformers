# coding=utf-8
# Copyright 2025 Kyutai and The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..auto import AutoModel
from ..llama.modeling_llama import LlamaForCausalLM, KwargsForCausalLM
from ..moshi.modeling_moshi import MoshiModel

from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import (
    FlashAttentionKwargs,
    flash_attn_supports_top_left_mask,
    is_flash_attn_available,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    LossKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel
from .configuration_moshi_asr import MoshiAsrConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask


if is_flash_attn_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


class MoshiAsrEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: should it be splitted to audio and text embeddings?
        self.embed_tokens = nn.Embedding(
            config.vocab_size + (config.num_codebooks * config.codebook_vocab_size), config.hidden_size
        )
        audio_tokens_offsets = torch.arange(config.num_codebooks) * config.codebook_vocab_size
        audio_tokens_offsets += config.vocab_size
        audio_tokens_offsets = nn.functional.pad(
            audio_tokens_offsets, (1, 0)
        )  # pad one 0 to the left for the text token
        self.register_buffer("audio_tokens_offsets", audio_tokens_offsets, persistent=False)

    def forward(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids + self.audio_tokens_offsets)
        inputs_embeds = inputs_embeds.sum(dim=2)
        return inputs_embeds

class MoshiAsrModel(MoshiModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = MoshiAsrEmbeddings(config)


class MoshiAsrForConditionalGeneration(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.codec_model = AutoModel.from_config(config.codec_config)

    def _merge_input_ids_with_input_values(self, input_ids, input_values, cache_position, audio_codes=None):

        # constraints
        # 1. input_ids same seq_len as seq_len extracted from input_values and cache_position

        self.frame_size = 1920
        self.delay_in_tokens = 25
        self.num_codebooks = 32
        self.audio_bos_token_id = 2048
        self.audio_pad_token_id = 48000

        input_ids = input_ids.unsqueeze(0)
        if audio_codes is not None:
            input_ids = torch.cat([input_ids, audio_codes], dim=2)

        elif input_values is not None:
            is_init = cache_position[0] == 0
            # in kyutai original code, first frame is always replaced by an init frame

            if not is_init:
                start_idx = cache_position[0] * self.frame_size 
                end_idx = (cache_position[-1] + 1) * self.frame_size
                curr_audio_frames = input_values[..., start_idx:end_idx]

                codes = torch.load("/Users/eustachelebihan/dev/add-moshi-asr/codes.pt")
                audio_tokens = codes[:, cache_position - 1, :]

                #######
                # audio_tokens = self.codec_model.encode(curr_audio_frames).audio_codes
                # audio_tokens = audio_tokens.transpose(1, 2)
                #######

                input_ids = torch.cat([input_ids, audio_tokens], dim=2)

            elif is_init and len(cache_position) > 1:
                start_idx = cache_position[1] * self.frame_size
                end_idx = (cache_position[-1] + 1) * self.frame_size
                curr_audio_frames = input_values[..., start_idx:end_idx]

                codes = torch.load("/Users/eustachelebihan/dev/add-moshi-asr/codes.pt")
                audio_tokens = codes[:, cache_position, :]
   
                ########
                # audio_tokens = self.codec_model.encode(curr_audio_frames).audio_codes
                # audio_tokens = audio_tokens.transpose(1, 2)
                ########

                init_audio_tokens = torch.ones((input_ids.shape[0], 1, 32), dtype=torch.long, device=input_ids.device) * self.audio_bos_token_id
                audio_tokens = torch.cat([init_audio_tokens, audio_tokens], dim=1)
                input_ids = torch.cat([input_ids, audio_tokens], dim=2)

            else:
                # initialization state
                init_audio_tokens = torch.ones((input_ids.shape[0], 1, 32), dtype=torch.long, device=input_ids.device) * self.audio_bos_token_id
                input_ids = torch.cat([input_ids, init_audio_tokens], dim=2)

        else:
            raise ValueError("something") # TODO

        return input_ids
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None, # -> TODO: could be a generator
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        TODO
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, device=device
            )

        if inputs_embeds is None:
            input_ids = self._merge_input_ids_with_input_values(input_ids, input_values, cache_position)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["MoshiAsrModel", "MoshiAsrForConditionalGeneration"]