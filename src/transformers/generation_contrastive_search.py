# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Tuple

import torch
from torch import nn

from .utils import logging


logger = logging.get_logger(__name__)


"""
This file contains the utils functions for the contrastive search, which will be called in `generation_utils`
"""


def ranking_fast(
    context_hidden: torch.FloatTensor,
    next_hidden: torch.FloatTensor,
    next_top_k_probs: torch.FloatTensor,
    alpha: float,
    beam_width: int,
) -> Tuple[torch.FloatTensor]:
    """
    context_hidden: bsz*beam x seqlen x embed_dim next_hidden: bsz*beam x 1 x embed_dim next_top_k_probs: bsz x beam
    """
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)  # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)  # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores
    scores = torch.stack(torch.split(scores, beam_width))  # [B, K]
    selected_scores, selected_idx = scores.max(dim=-1)  # [B]
    return selected_scores, selected_idx


def ContrastiveDecodingOneStepFast(
    model,
    beam_width: int = 1,
    penalty_alpha: float = 0.0,
    past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
    last_hidden_states: torch.FloatTensor = None,
    logit_for_next_step: torch.FloatTensor = None,
    is_encoder_decoder: bool = False,
    **model_inputs,
) -> Tuple:
    """
    contrastive search first selects top-k candidates by the logit scores; then these candidate tokens are fed into the
    language models to compute the degeneration penalty, which will be used to re-rank these candidate tokens.
    """
    bsz, seqlen, embed_dim = last_hidden_states.size()
    next_probs = nn.functional.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)

    # build next attention mask
    attention_mask = model_inputs["attention_mask"]  # [B, S]
    # decoder-only model need the full attention mask, not only the mask for the last token
    if is_encoder_decoder is False:
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=-1)
    attention_mask = attention_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(-1, attention_mask.size(-1))

    # encoder-decoder model also contains the `encoder_outputs`
    if is_encoder_decoder and "encoder_outputs" in model_inputs:
        encoder_outputs = model_inputs["encoder_outputs"]
    else:
        encoder_outputs = None
    next_model_inputs = model.prepare_inputs_for_generation(
        top_k_ids.view(-1, 1),
        past=past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
        encoder_outputs=encoder_outputs,
    )
    # compute the candidate tokens by the language model and collects their hidden_states
    output = model(output_hidden_states=True, **next_model_inputs)
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]
    # name is different for encoder-decoder and decoder-only models
    if is_encoder_decoder:
        next_hidden = output.decoder_hidden_states[-1]
        full_hidden_states = output.decoder_hidden_states
    else:
        next_hidden = output.hidden_states[-1]
        full_hidden_states = output.hidden_states
    context_hidden = (
        last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz * beam_width, seqlen, embed_dim)
    )

    # compute the degeneratin penalty and re-rank the candidates based on the degeneration penalty and the model confidence
    # the scores and index of the selected tokens are returned
    selected_scores, selected_idx = ranking_fast(
        context_hidden,
        next_hidden,
        top_k_probs,
        penalty_alpha,
        beam_width,
    )
    # prepare for the next step: (1) next token_id; (2) past_key_values; (3) last_hidden_states for computing the degeneration penalty; (4) logits for selecting next top-k candidates; (5) selected tokens scores (model confidence minus degeneration penalty); (6) decoder hidden_states
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))
    next_hidden = next_hidden[range(bsz), selected_idx, :]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)

    decoder_hidden_states = []
    for layer in full_hidden_states:
        layer = torch.stack(torch.split(layer.squeeze(dim=1), beam_width))
        layer = layer[range(bsz), selected_idx, :]
        decoder_hidden_states.append(layer)

    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]
    return next_id.squeeze(dim=-1), past_key_values, last_hidden_states, logits, selected_scores, decoder_hidden_states


def enlarge_past_key_values(
    past_key_values: Tuple[Tuple[torch.FloatTensor]], beam_width: int
) -> Tuple[Tuple[torch.FloatTensor]]:
    """
    Copy and extend the past_key_values for the next step re-rank each item in `past_key_values` is the 4-dimension
    matrix, whose shapre is [batch_size, num_head, seq_len, embed_dim] Suppose the size of the next token candidate
    size is K, we need to obtain the new `past_key_values`, whose shape is [batch_size*K, num_head, seq_len, embed_dim]
    """
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        # item is either the key or the value matrix
        for item in layer:
            bsz, num_head, seq_len, esz = item.size()
            item = (
                item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz * beam_width, num_head, seq_len, esz)
            )  # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def select_past_key_values(
    past_key_values: Tuple[Tuple[torch.FloatTensor]], beam_width: int, selected_idx: torch.FloatTensor
) -> Tuple[Tuple[torch.FloatTensor]]:
    """
    Extract the `past_key_value` for the selected tokens, each item in `past_key_value` is the 4-dimension matrix,
    whose shape is [batch_size*K, num_head, seq_len, embed_dim], where K is the number of the candidate tokens. We aim
    to obtain the `past_key_value` of the selected next token, whose shape is [batch_size, num_head, seq_len,
    embed_dim]
    """
    new_key_values = []
    for layer in past_key_values:
        items = []
        # item is either the key or the value matrix
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam // beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))  # [B, K, num_head, seq_len, esz]
            item = item[range(bsz), selected_idx, :, :, :]  # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values
