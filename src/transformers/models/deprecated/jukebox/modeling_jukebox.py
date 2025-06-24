# coding=utf-8
# Copyright 2022 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""PyTorch Jukebox model."""

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm

from ....activations import ACT2FN
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, logging
from ....utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig


logger = logging.get_logger(__name__)


def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits (`torch.Tensor`):
            logits distribution shape (vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            When `top_k >0` keep only top key tokens with highest probability (top-k filtering).
        top_p (`int`, *optional*, defaults to 0):
            When `top_p>0.0` keep the top tokens with cumulative probability >= `top_p` (nucleus filtering).
    """
    logits = logits.clone()
    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def get_relevant_lyric_tokens(full_tokens, max_n_lyric_tokens, total_length, offset, duration):
    """
    Extract only the relevant tokens based on the character position. A total of `max_n_lyric_tokens` tokens will be
    returned. If the provided token sequence is smaller, it will be padded, otherwise, only characters ranging from the
    midpoint - `max_n_lyric_tokens//2` to the midpoint + `max_n_lyric_tokens//2` will be returned. This *focuses* on
    the most relevant tokens (in time) for the sequence.

    Args:
        full_tokens (`list[int]`):
            List containing the token ids of the entire lyrics.
        total_length (`int`):
            Total expected length of the music (not all of it is generated, see duration), in samples.
        offset (`int`):
            Starting sample in the music. If the offset is greater than 0, the lyrics will be shifted take that into
            account
        duration (`int`):
            Expected duration of the generated music, in samples. The duration has to be smaller than the total length,
            which represent the overall length of the signal,
    """
    full_tokens = full_tokens[0]
    if len(full_tokens) < max_n_lyric_tokens:
        tokens = torch.cat(
            [torch.zeros(max_n_lyric_tokens - len(full_tokens), dtype=torch.long).to(full_tokens.device), full_tokens]
        )
        indices = [-1] * (max_n_lyric_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
    else:
        midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
        midpoint = min(max(midpoint, max_n_lyric_tokens // 2), len(full_tokens) - max_n_lyric_tokens // 2)
        tokens = full_tokens[midpoint - max_n_lyric_tokens // 2 : midpoint + max_n_lyric_tokens // 2]
        indices = list(range(midpoint - max_n_lyric_tokens // 2, midpoint + max_n_lyric_tokens // 2))
    return tokens.unsqueeze(dim=0), indices


# Break total_length into hops/windows of size n_ctx separated by hop_length
def get_starts(total_length, n_ctx, hop_length):
    starts = []
    for start in range(0, total_length - n_ctx + hop_length, hop_length):
        if start + n_ctx >= total_length:
            # Last hop could be smaller, we make it n_ctx to maximise context
            start = total_length - n_ctx
        starts.append(start)
    return starts


def get_alignment(music_tokens, labels, prior, config):
    level = prior.levels - 1  # Top level used
    n_ctx = prior.n_ctx
    tokens = music_tokens[level]
    batch_size, total_length = tokens.shape[0], tokens.shape[1]
    if total_length < n_ctx:
        padding_length = n_ctx - total_length
        tokens = torch.cat(
            [tokens, torch.zeros(batch_size, n_ctx - total_length, dtype=tokens.dtype, device=tokens.device)], dim=1
        )
        total_length = tokens.shape[1]
    else:
        padding_length = 0

    hop_length = int(config.hop_fraction[-level - 1] * prior.n_ctx)
    alignment_head, alignment_layer = config.prior_alignment_head[0], config.prior_alignment_layer[0]
    attn_layers = {alignment_layer}
    alignment_hops = {}
    indices_hops = {}
    for start in tqdm(get_starts(total_length, n_ctx, hop_length), desc="Computing lyric to music alignment "):
        end = start + n_ctx
        # set metadata offset, sample_length and lyrics tokens
        metadata, indices_hop = prior.get_metadata(labels, start, config.sample_length, get_indices=True, offset=0)
        tokens_bs = torch.chunk(tokens, batch_size, dim=0)
        metadata_bs = torch.chunk(metadata, batch_size, dim=0)
        w_hops = []
        for tokens_i, metadata_i in zip(tokens_bs, metadata_bs):
            w_hop = prior.forward_tokens(tokens_i[:, start:end], [], metadata_i, get_attn_weights=attn_layers)
            w_hops.append(w_hop[0][:, alignment_head])
            del w_hop
        weights = torch.cat(w_hops, dim=0)
        del w_hops
        alignment_hop = weights.to(device="cpu", dtype=torch.float).numpy()
        del weights

        # alignment_hop has shape (bs, n_ctx, nb_relevant_lyric_tokens)
        # indices_hop is a list of len=bs, each entry of len hps.nb_relevant_lyric_tokens
        indices_hops[start] = indices_hop
        alignment_hops[start] = alignment_hop

    # Combine attn for each hop into attn for full range
    # Use indices to place them into correct place for corresponding source tokens
    alignments = []
    for item in range(batch_size):
        # Note each item has different length lyrics
        full_tokens = labels[0, 3:]
        alignment = np.zeros((total_length, len(full_tokens) + 1))
        for start in reversed(get_starts(total_length, n_ctx, hop_length)):
            end = start + n_ctx
            alignment_hop = alignment_hops[start][item]
            indices = indices_hops[start][item]
            alignment[start:end, indices] = alignment_hop
        alignment = alignment[: total_length - padding_length, :-1]  # remove token padding, and last lyric index
        alignments.append(alignment)
    return alignments


def save_temp_audio(fname, lvl, metas, aud):
    aud = torch.clamp(aud, -1, 1).cpu().numpy()
    for i in list(range(aud.shape[0])):
        if metas is not None:
            artists, genres, lyrics = list(metas)[i].values()
            path = f"{fname}/lvl_{lvl}-{artists}-{genres}-{lyrics[:5]}-{i}"
            np.save(path, aud[i])
        else:
            np.save(f"{fname}/lvl_{lvl}-sample-{i}", aud[i])


def get_mask(mask, query_length, key_value_length, blocks, spread, device, sample, sample_t):
    # returns a mask of shape 1 x 1 x query_length x key_value_length or None if masking is not needed.
    if mask is None or query_length == 1:
        return None
    offset = sample_t - query_length if sample else max(key_value_length - query_length, 0)
    if mask == "autoregressive":
        # Masked dense
        mask = torch.ones(query_length, key_value_length, device=device).tril(offset)
    elif mask == "summary":
        # Masked summary
        mask = torch.ones(query_length, query_length, device=device).tril()
        mask = torch.ones(query_length, query_length, device=device).tril()
        mask = mask.view(query_length, blocks, query_length // blocks)[:, :-1, -key_value_length // blocks :]
        mask = (
            torch.nn.functional.pad(
                mask,
                (0, 0, 1, 0),
                value=1,
            )
            .contiguous()
            .view(query_length, key_value_length)
        )
    elif mask == "prime":
        mask = torch.ones(query_length, key_value_length, device=device).tril(offset)
    return mask.view(1, 1, query_length, key_value_length)


class JukeboxConv1D(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        weight = torch.empty(input_width, output_width)
        bias = torch.zeros(output_width)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, hidden_states):
        size_out = (*hidden_states.size()[:-1], self.output_width)
        hidden_states = torch.addmm(
            self.bias.type_as(hidden_states),
            hidden_states.view(-1, hidden_states.size(-1)),
            self.weight.type_as(hidden_states),
        )
        hidden_states = hidden_states.view(*size_out)
        return hidden_states


class JukeboxResConv1DBlock(nn.Module):
    def __init__(self, config, conv_width, depth=1, res_scale=1.0):
        super().__init__()
        hidden_dim = config.res_convolution_multiplier * conv_width
        dilation = config.res_dilation_growth_rate**depth
        padding = dilation

        self.res_scale = res_scale
        self.activation = nn.ReLU()
        self.conv1d_1 = nn.Conv1d(conv_width, hidden_dim, 3, 1, padding, dilation)
        self.conv1d_2 = nn.Conv1d(hidden_dim, conv_width, 1, 1, 0)

    def forward(self, hidden_states):
        residuals = hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv1d_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv1d_2(hidden_states)
        return residuals + self.res_scale * hidden_states


class JukeboxResnet1D(nn.Module):
    def __init__(self, config, conv_width, n_depth, reverse_dilation=False):
        super().__init__()
        self.dilation_cycle = config.res_dilation_cycle
        res_scale = 1.0 if not config.conv_res_scale else 1.0 / math.sqrt(n_depth)

        blocks = []
        for depth in range(n_depth):
            block_depth = depth if self.dilation_cycle is None else depth % self.dilation_cycle
            blocks.append(JukeboxResConv1DBlock(config, conv_width, block_depth, res_scale))

        if reverse_dilation:
            blocks = blocks[::-1]
        self.resnet_block = nn.ModuleList(blocks)

    def forward(self, hidden_states):
        for block in self.resnet_block:
            hidden_states = block(hidden_states)
        return hidden_states


class JukeboxEncoderConvBlock(nn.Module):
    def __init__(self, config, embed_dim, hidden_dim, depth, down_t, stride_t):
        super().__init__()
        blocks = []
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                blocks.append(nn.Conv1d(embed_dim if i == 0 else hidden_dim, hidden_dim, filter_t, stride_t, pad_t))
                blocks.append(JukeboxResnet1D(config, hidden_dim, depth))
        self.proj_out = nn.Conv1d(hidden_dim, config.embed_dim, 3, 1, 1)
        self.downsample_block = nn.ModuleList(blocks)

    def forward(self, hidden_states):
        for block in self.downsample_block:
            hidden_states = block(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


class JukeboxEncoder(nn.Module):
    def __init__(self, config, width, depth, levels, downs_t, strides_t):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList()

        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for i, down_t, stride_t in iterator:
            self.level_blocks.append(
                JukeboxEncoderConvBlock(
                    config, config.conv_input_shape if i == 0 else config.embed_dim, width, depth, down_t, stride_t
                )
            )

    def forward(self, hidden_states):
        all_hidden_states = []

        # 64, 32, ...
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            hidden_states = level_block(hidden_states)
            all_hidden_states.append(hidden_states)

        return all_hidden_states


class JukeboxDecoderConvBock(nn.Module):
    def __init__(self, config, embed_dim, hidden_dim, depth, down_t, stride_t, reverse_dilation=True):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t = stride_t * 2
            pad_t = stride_t // 2
            self.proj_in = nn.Conv1d(embed_dim, hidden_dim, 3, 1, 1)
            for i in range(down_t):
                blocks.append(JukeboxResnet1D(config, hidden_dim, depth, reverse_dilation))
                blocks.append(
                    nn.ConvTranspose1d(
                        hidden_dim, hidden_dim if i < down_t - 1 else embed_dim, filter_t, stride_t, pad_t
                    )
                )
        self.upsample_block = nn.ModuleList(blocks)

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        for block in self.upsample_block:
            hidden_states = block(hidden_states)
        return hidden_states


class JukeboxDecoder(nn.Module):
    def __init__(self, config, hidden_dim, depth, levels, downs_t, strides_t):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList()
        for level, down_t, stride_t in zip(list(range(self.levels)), downs_t, strides_t):
            self.level_blocks.append(
                JukeboxDecoderConvBock(config, config.embed_dim, hidden_dim, depth, down_t, stride_t)
            )

        self.out = nn.Conv1d(config.embed_dim, config.conv_input_shape, 3, 1, 1)

    def forward(self, hidden_states, all_levels=True):
        hidden_state = hidden_states[-1]

        # 32, 64 ...
        for level in reversed(range(self.levels)):
            level_block = self.level_blocks[level]
            hidden_state = level_block(hidden_state)

            if level != 0 and all_levels:
                hidden_state = hidden_state + hidden_states[level - 1]

        hidden_state = self.out(hidden_state)
        return hidden_state


class JukeboxBottleneckBlock(nn.Module):
    def __init__(self, config: JukeboxVQVAEConfig):
        super().__init__()
        self.nb_discrete_codes = config.nb_discrete_codes
        self.codebook_width = config.embed_dim
        self.mu = config.lmu
        self.threshold = 1.0
        self.init = False
        self.codebook_sum = None
        self.codebook_elem = None
        self.register_buffer("codebook", torch.zeros(self.nb_discrete_codes, self.codebook_width))

    def _tile(self, hidden_states):
        dim, embed_width = hidden_states.shape
        if dim < self.nb_discrete_codes:
            n_repeats = (self.nb_discrete_codes + dim - 1) // dim
            std = 0.01 / np.sqrt(embed_width)
            hidden_states = hidden_states.repeat(n_repeats, 1)
            hidden_states = hidden_states + torch.randn_like(hidden_states) * std
        return hidden_states

    def init_codebook(self, hidden_states):
        nb_discrete_codes = self.nb_discrete_codes
        self.init = True
        codes = self._tile(hidden_states)
        self.codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]
        self.codebook_sum = self.codebook
        self.codebook_elem = torch.ones(nb_discrete_codes, device=self.codebook.device)

    def update_codebook(self, hidden_states, latent_states):
        mu, codebook_width, nb_discrete_codes = self.mu, self.codebook_width, self.nb_discrete_codes
        with torch.no_grad():
            # Calculate new centres
            # nb_discrete_codes, batch_size * seq_length
            latent_states_onehot = torch.zeros(nb_discrete_codes, hidden_states.shape[0], device=hidden_states.device)
            latent_states_onehot.scatter_(0, latent_states.view(1, hidden_states.shape[0]), 1)

            _codebook_sum = torch.matmul(latent_states_onehot, hidden_states)
            _codebook_elem = latent_states_onehot.sum(dim=-1)  # nb_discrete_codes
            codes = self._tile(hidden_states)
            _random_codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]

            # Update centres
            old_codebook = self.codebook
            self.codebook_sum = mu * self.codebook_sum + (1.0 - mu) * _codebook_sum
            self.codebook_elem = mu * self.codebook_elem + (1.0 - mu) * _codebook_elem  # nb_discrete_codes
            usage = (self.codebook_elem.view(nb_discrete_codes, 1) >= self.threshold).float()

            norm_code = self.codebook_sum.view(nb_discrete_codes, codebook_width) / self.codebook_elem.view(
                nb_discrete_codes, 1
            )
            self.codebook = usage * (norm_code) + (1 - usage) * _random_codebook
            _codebook_prob = _codebook_elem / torch.sum(_codebook_elem)  # prob of each bin
            entropy = -torch.sum(_codebook_prob * torch.log(_codebook_prob + 1e-8))  # entropy ie how diverse
            used_curr = (_codebook_elem >= self.threshold).sum()
            usage = torch.sum(usage)
            dk = torch.linalg.norm(self.codebook - old_codebook) / np.sqrt(np.prod(old_codebook.shape))
        return {"entropy": entropy, "used_curr": used_curr, "usage": usage, "dk": dk}

    def preprocess(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if hidden_states.shape[-1] == self.codebook_width:
            prenorm = torch.linalg.norm(hidden_states - torch.mean(hidden_states)) / np.sqrt(
                np.prod(hidden_states.shape)
            )
        elif hidden_states.shape[-1] == 2 * self.codebook_width:
            x1, x2 = hidden_states[..., : self.codebook_width], hidden_states[..., self.codebook_width :]
            prenorm = (torch.linalg.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (
                torch.linalg.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape))
            )

            # Normalise
            hidden_states = x1 + x2

        return hidden_states, prenorm

    def postprocess(self, latent_states, dequantised_states, x_shape):
        batch_size, time = x_shape
        dequantised_states = dequantised_states.view(batch_size, time, -1).permute(0, 2, 1).contiguous()
        latent_states = latent_states.view(batch_size, time)
        return latent_states, dequantised_states

    def quantise(self, latent_states):
        # Calculate latent code latent_states
        codebook_weights = self.codebook.t()
        distance = (
            torch.sum(latent_states**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(latent_states, codebook_weights)
            + torch.sum(codebook_weights**2, dim=0, keepdim=True)
        )  # (batch_size * latent_states , codebook_weights)
        min_distance, music_tokens = torch.min(distance, dim=-1)
        fit = torch.mean(min_distance)
        return music_tokens, fit

    def dequantise(self, music_tokens):
        dequantised_states = F.embedding(music_tokens, self.codebook)
        return dequantised_states

    def encode(self, latent_states):
        samples, _, seq_len = latent_states.shape

        # Preprocess.
        latent_states, _ = self.preprocess(latent_states)

        # Quantise
        music_tokens, _ = self.quantise(latent_states)

        # Postprocess.
        music_tokens = music_tokens.view(samples, seq_len)
        return music_tokens

    def decode(self, music_tokens):
        samples, seq_len = music_tokens.shape

        # Dequantise
        dequantised_states = self.dequantise(music_tokens)

        # Postprocess
        dequantised_states = (
            dequantised_states.view(samples, seq_len, self.codebook_width).permute(0, 2, 1).contiguous()
        )
        return dequantised_states

    def forward(self, hidden_states, update_codebook=True):
        samples, _, seq_len = hidden_states.shape

        # Preprocess
        hidden_states, prenorm = self.preprocess(hidden_states)

        # Init codebook if not inited
        if update_codebook and not self.init:
            self.init_codebook(hidden_states)

        # Quantise and dequantise through bottleneck
        music_tokens, fit = self.quantise(hidden_states)
        dequantised_states = self.dequantise(music_tokens)

        # Update embeddings
        if update_codebook:
            update_metrics = self.update_codebook(hidden_states, music_tokens)
        else:
            update_metrics = {}

        # Loss
        commit_loss = torch.linalg.norm(dequantised_states.detach() - hidden_states) ** 2 / np.prod(
            hidden_states.shape
        )

        # Passthrough
        dequantised_states = hidden_states + (dequantised_states - hidden_states).detach()

        # Postprocess
        music_tokens, dequantised_states = self.postprocess(music_tokens, dequantised_states, (samples, seq_len))
        return music_tokens, dequantised_states, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)


class JukeboxBottleneck(nn.Module):
    def __init__(self, config, levels):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(JukeboxBottleneckBlock(config))

    def encode(self, raw_audio):
        music_tokens = [
            level_block.encode(hidden_states) for (level_block, hidden_states) in zip(self.level_blocks, raw_audio)
        ]
        return music_tokens

    def decode(self, music_tokens, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        quantised_audio = [
            level_block.decode(z) for (level_block, z) in zip(self.level_blocks[start_level:end_level], music_tokens)
        ]
        return quantised_audio

    def forward(self, input_audio):
        music_tokens, quantised_states, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            level_block = self.level_blocks[-level - 1]
            hidden_states = input_audio[level]
            sampled_tokens, quantised_state, commit_loss, metric = level_block(
                hidden_states, update_codebook=self.training
            )
            music_tokens.append(sampled_tokens)
            if not self.training:
                # Be extra paranoid and make sure the encoder weights can't
                # change from straight-through estimator
                quantised_state = quantised_state.detach()
            quantised_states.append(quantised_state)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return music_tokens, quantised_states, commit_losses, metrics


JUKEBOX_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config (`JukeboxConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    """The Hierarchical VQ-VAE model used in Jukebox. This model follows the Hierarchical VQVAE paper from [Will Williams, Sam
Ringer, Tom Ash, John Hughes, David MacLeod, Jamie Dougherty](https://huggingface.co/papers/2002.08111).

    """,
    JUKEBOX_START_DOCSTRING,
)
class JukeboxVQVAE(PreTrainedModel):
    config_class = JukeboxVQVAEConfig
    base_model_prefix = "vqvae"

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):  # embed_tokens
            module.weight.data.normal_(mean=0.0, std=0.02 * self.config.init_scale)
        elif isinstance(module, JukeboxConv1D):
            if self.config.zero_out:
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=0.02 * self.config.init_scale)
        elif isinstance(module, JukeboxResConv1DBlock) and self.config.zero_out:
            module.conv1d_2.weight.data.zero_()
            module.conv1d_2.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, config: JukeboxVQVAEConfig):
        super().__init__(config)
        downs_t = config.res_downs_t
        strides_t = config.res_strides_t
        if not config.sample_length:
            downsamples = [stride**down for stride, down in zip(strides_t, downs_t)]
            top_raw_to_tokens = np.prod(downsamples)
            config.sample_length = (
                config.sample_length_in_seconds * config.sampling_rate // top_raw_to_tokens
            ) * top_raw_to_tokens
            config.sample_length = config.sample_length.astype(int)

        self.nb_discrete_codes = config.nb_discrete_codes
        self.commit = config.commit
        self.sample_length = config.sample_length

        self.downsamples = [stride**down for stride, down in zip(strides_t, downs_t)]
        self.hop_lengths = np.cumprod(self.downsamples)
        self.levels = levels = config.levels
        self.music_tokens_shapes = [
            (int(self.sample_length // self.hop_lengths[-level - 1])) for level in range(levels)
        ]

        self.multipliers = config.multipliers if config.multipliers is not None else [1] * levels

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            width = config.res_conv_width * self.multipliers[level]
            depth = config.res_conv_depth * self.multipliers[level]
            self.encoders.append(
                JukeboxEncoder(config, width, depth, level + 1, downs_t[: level + 1], strides_t[: level + 1])
            )
            self.decoders.append(
                JukeboxDecoder(config, width, depth, level + 1, downs_t[: level + 1], strides_t[: level + 1])
            )

        self.bottleneck = JukeboxBottleneck(config, levels)

    def _decode(self, music_tokens, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        latent_states = self.bottleneck.decode(music_tokens, start_level=start_level, end_level=end_level)
        # Use only lowest level
        decoder, dequantised_state = self.decoders[start_level], latent_states[0:1]
        dequantised_state = decoder(dequantised_state, all_levels=False)
        dequantised_state = dequantised_state.permute(0, 2, 1)
        return dequantised_state

    def decode(self, music_tokens, start_level=0, end_level=None, bs_chunks=1) -> torch.Tensor:
        """
        Transforms the input `music_tokens` to their `raw_audio` representation.

        Args:
            music_tokens (`torch.LongTensor`):
                Tensor of music tokens which will be decoded to raw audio by using the codebook. Each music token
                should be an index to a corresponding `code` vector in the codebook.
            start_level (`int`, *optional*):
                Level at which the decoding process will start. Default to 0.
            end_level (`int`, *optional*):
                Level at which the decoding process will start. Default to None.
            bs_chunks (int, *optional*):
                Number of chunks to process at the same time.
        """
        token_chunks = [torch.chunk(token, bs_chunks, dim=0) for token in music_tokens]
        dequantised_states = []
        for i in range(bs_chunks):
            music_tokens_i = [chunks[i] for chunks in token_chunks]
            dequantised_state = self._decode(music_tokens_i, start_level=start_level, end_level=end_level)
            dequantised_states.append(dequantised_state)
        return torch.cat(dequantised_states, dim=0)

    def _encode(self, raw_audio, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        input_audio = raw_audio.permute(0, 2, 1).float()
        latent_states = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            latent_state = encoder(input_audio)
            latent_states.append(latent_state[-1])
        music_tokens = self.bottleneck.encode(latent_states)
        return music_tokens[start_level:end_level]

    def encode(self, input_audio, start_level=0, end_level=None, bs_chunks=1):
        """
        Transforms the `input_audio` to a discrete representation made out of `music_tokens`.

        Args:
            input_audio (`torch.Tensor`):
                Raw audio which will be encoded to its discrete representation using the codebook. The closest `code`
                form the codebook will be computed for each sequence of samples.
            start_level (`int`, *optional*, defaults to 0):
                Level at which the encoding process will start. Default to 0.
            end_level (`int`, *optional*):
                Level at which the encoding process will start. Default to None.
            bs_chunks (int, *optional*, defaults to 1):
                Number of chunks of raw audio to process at the same time.
        """
        audio_chunks = torch.chunk(input_audio, bs_chunks, dim=0)
        music_tokens_list = []
        for chunk_i in audio_chunks:
            music_tokens_i = self._encode(chunk_i, start_level=start_level, end_level=end_level)
            music_tokens_list.append(music_tokens_i)
        music_tokens = [torch.cat(music_tokens_level, dim=0) for music_tokens_level in zip(*music_tokens_list)]
        return music_tokens

    def sample(self, n_samples):
        music_tokens = [
            torch.randint(0, self.nb_discrete_codes, size=(n_samples, *music_tokens_shape), device="cpu")
            for music_tokens_shape in self.music_tokens_shapes
        ]
        return self.decode(music_tokens)

    def forward(self, raw_audio: torch.FloatTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VQ-VAE, encodes the `raw_audio` to latent states, which are then decoded for each level.
        The commit loss, which ensure that the encoder's computed embeddings are close to the codebook vectors, is
        computed.

        Args:
            raw_audio (`torch.FloatTensor`):
                Audio input which will be encoded and decoded.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`


        Example:
        ```python
        >>> from transformers import JukeboxVQVAE, set_seed
        >>> import torch

        >>> model = JukeboxVQVAE.from_pretrained("openai/jukebox-1b-lyrics").eval()
        >>> set_seed(0)
        >>> zs = [torch.randint(100, (4, 1))]
        >>> model.decode(zs).shape
        torch.Size([4, 8, 1])
        ```
        """

        # Encode/Decode
        input_audio = raw_audio.permute(0, 2, 1).float()
        latent_states = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            latent_state = encoder(input_audio)
            latent_states.append(latent_state[-1])

        _, music_tokens, commit_losses, _ = self.bottleneck(latent_states)
        dequantised_states = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            dequantised_state = decoder(music_tokens[level : level + 1], all_levels=False)
            dequantised_states.append(dequantised_state.permute(0, 2, 1))

        commit_loss = sum(commit_losses)
        loss = self.commit * commit_loss

        return dequantised_states, loss


class JukeboxMLP(nn.Module):
    def __init__(self, config):
        # a single channel is always used in original code
        super().__init__()
        embed_dim = config.hidden_size
        hidden_dim = int(config.mlp_multiplier * embed_dim)

        self.c_fc = JukeboxConv1D(embed_dim, hidden_dim)
        self.c_proj = JukeboxConv1D(hidden_dim, embed_dim)
        self.act = ACT2FN[config.act_fn]
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class JukeboxLayerNorm(FusedLayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.width = np.prod(normalized_shape)
        self.max_numel = 65535 * self.width

    def forward(self, input):
        if input.numel() > self.max_numel:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps).type_as(input)
        else:
            return super().forward(input).type_as(input)


class JukeboxAttention(nn.Module):
    def __init__(self, config, n_ctx, attn_func="dense_attn"):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.n_heads = config.n_heads
        self.dropout = config.attn_dropout
        hidden_dim = int(config.attention_multiplier * self.embed_dim)

        self.head_dim = hidden_dim // config.n_heads
        self.n_ctx = n_ctx
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim**-0.25
        self.mask = config.mask

        if attn_func == "cross_attention":
            self.c_attn = JukeboxConv1D(self.embed_dim, hidden_dim)
            self.c_enc_kv = JukeboxConv1D(self.embed_dim, hidden_dim * 2)
        else:
            self.c_attn = JukeboxConv1D(self.embed_dim, hidden_dim * 3)

        self.c_proj = JukeboxConv1D(hidden_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        # Sequence of length seq_len is factored as [blocks, seq_len // blocks]
        self.attn_func = attn_func
        if attn_func == "cross_attention":
            self.qkv = self.decode_qkv
        elif attn_func == "prime_attn":
            self.qkv = self.prime_qkv
        else:
            self.qkv = self.factored_qkv

        ATTENTION_MAP = {
            "dense_attn": (self.dense_attn, "autoregressive"),
            "block_attn": (self.block_attn, "autoregressive"),
            "transpose_block_attn": (self.transpose_block_attn, "autoregressive"),
            "prev_block_attn": (self.prev_block_attn, None),
            "summary_attn": (self.summary_attn, "summary"),
            "summary_spread_attn": (self.summary_spread_attn, "summary"),
            "cross_attention": (self.dense_attn, None),
            "prime_attn": (self.prime_attn, "prime"),
        }
        self.attn, self.attn_mask = ATTENTION_MAP[attn_func]

        self.blocks = config.blocks
        self.spread = config.spread
        if self.blocks is not None:
            self.block_ctx = self.n_ctx // self.blocks

        self.sample_t = 0
        self.cache = {}
        self.encoder_len = config.nb_relevant_lyric_tokens  # length of the encoder input ids
        self.record_attn = False

    def _attn(self, query_states, key_states, value_states, sample):
        scale = self.scale
        if self.training:
            attention_weight = torch.matmul(query_states * scale, key_states * scale)
        else:
            attention_weight = torch.matmul(query_states, key_states)
            attention_weight.mul_(scale * scale)
        attn_weight_type = attention_weight.dtype
        attention_weight = attention_weight.float()
        if self.mask:
            # Generate appropriate mask to mask out all positions before current
            # Might take up lot of memory for dense, so can cache it
            mask = get_mask(
                self.attn_mask,
                query_states.size(-2),
                key_states.size(-1),
                self.blocks,
                self.spread,
                attention_weight.device,
                sample,
                self.sample_t,
            )
            if mask is not None:
                attention_weight = attention_weight * mask + -1e9 * (1 - mask)
        attention_prob = F.softmax(attention_weight, dim=-1).type(attn_weight_type)
        if self.record_attn:
            self.attention_prob = attention_prob
            if self.attn_func == "prime_attn":
                # only keep music queries and lyrics keys/values
                self.attention_prob = self.attention_prob[:, :, self.encoder_len :, : self.encoder_len]
        attention_prob = self.attn_dropout(attention_prob)
        context_states = torch.matmul(attention_prob, value_states)
        return context_states

    def merge_heads(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = (*hidden_states.size()[:-2], hidden_states.size(-2) * hidden_states.size(-1))
        return hidden_states.view(*new_hidden_states_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, hidden_states, is_key=False):
        new_hidden_states_shape = (
            *hidden_states.size()[:-1],
            self.n_heads,
            hidden_states.size(-1) // self.n_heads,
        )
        hidden_states = hidden_states.view(*new_hidden_states_shape)  # in Tensorflow implem: fct split_states
        if is_key:
            return hidden_states.permute(0, 2, 3, 1)
        else:
            return hidden_states.permute(0, 2, 1, 3)

    def dense_attn(self, query, key, value, sample):
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        context_states = self._attn(query, key, value, sample)
        context_states = self.merge_heads(context_states)
        return context_states

    def block_attn(self, query, key, value, sample):
        block_ctx = self.block_ctx
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        if sample:
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            query_length = query.shape[1]
            query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)
            if query_length < seq_len:
                seq_len = query_length
                key = key[:, -seq_len:].contiguous()
                value = value[:, -seq_len:].contiguous()
            key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    def transpose_block_attn(self, query, key, value, sample):
        block_ctx = self.block_ctx
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        if sample:
            block_len = (seq_len - 1) % block_ctx
            key = key[:, block_len::block_ctx, :]
            value = value[:, block_len::block_ctx, :]
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            query_length = query.shape[1]
            query = query.view(batch_size, query_length // block_ctx, block_ctx, embed_dim)
            query = query.transpose(1, 2).contiguous()
            query = query.view(batch_size * block_ctx, query_length // block_ctx, embed_dim)

            key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
            key = key.transpose(1, 2).contiguous()
            key = key.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)

            value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
            value = value.transpose(1, 2).contiguous()
            value = value.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)

            block_attn = self.dense_attn(query, key, value, sample)
            block_attn = block_attn.view(batch_size, block_ctx, query_length // block_ctx, embed_dim)
            block_attn = block_attn.transpose(1, 2).contiguous()
            block_attn = block_attn.view(batch_size, query_length, embed_dim)

            return block_attn

    def prev_block_attn(self, query, key, value, sample):
        block_ctx = self.block_ctx
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        if sample:
            block = (seq_len - 1) // block_ctx
            prev_l = (block - 1) * block_ctx
            if block > 0:
                key = key[:, prev_l : prev_l + block_ctx, :]
                value = value[:, prev_l : prev_l + block_ctx, :]
            else:
                key = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
                value = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            query_length = query.shape[1]
            query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)

            key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0))
            key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)

            value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0))
            value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)

            if query_length < seq_len:
                nb_query_blocks = query_length // block_ctx
                nb_key_blocks = seq_len // block_ctx
                seq_len = query_length
                key = key.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
                key = key.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)

                value = value.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
                value = value.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)

            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    def summary_attn(self, query, key, value, sample):
        blocks = self.blocks
        block_ctx = self.block_ctx
        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        if sample:
            key = key[:, block_ctx - 1 : blocks * block_ctx - 1 : block_ctx, :]
            key = torch.nn.functional.pad(key, (0, 0, 1, 0))

            value = value[:, block_ctx - 1 : blocks * block_ctx - 1 : block_ctx, :]
            value = torch.nn.functional.pad(value, (0, 0, 1, 0))
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            key = key.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -1, :]
            key = torch.nn.functional.pad(key, (0, 0, 1, 0))  # batch_size, blocks, embed_dim

            value = value.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -1, :]
            value = torch.nn.functional.pad(value, (0, 0, 1, 0))  # batch_size, blocks, embed_dim
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    def summary_spread_attn(self, query, key, value, sample):
        blocks = self.blocks
        spread = self.spread

        batch_size, seq_len, embed_dim = value.shape  # For sample, query_len= 1, key_len = value_len = sample_t
        if sample:
            raise NotImplementedError
        else:
            key = key.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0)).contiguous()
            key = key.view(batch_size, blocks * spread, embed_dim)

            value = value.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0)).contiguous()
            value = value.view(batch_size, blocks * spread, embed_dim)

            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    def prime_attn(self, query, key, value, sample):
        encoder_len = self._encoder_len
        key = key[:, :encoder_len]
        value = value[:, :encoder_len]
        return self.dense_attn(query, key, value, sample)

    def factored_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        curr_ctx = hidden_states.shape[1]
        if last_encoder_hidden_states is not None:
            raise TypeError("last_encoder_hidden_states should be None")

        query, key, value = hidden_states.chunk(3, dim=2)
        if sample:
            self.sample_t += curr_ctx
            key, value = self._append_cache(key, value)
            l_cache = self._suff_cache_len()
            if self._cache_len() > l_cache:
                self._slice_cache(-l_cache)
            if curr_ctx > 1:
                if self.attn_func != "dense_attn":
                    query = self._pad_to_block_ctx(query, query=True)
                    key = self._pad_to_block_ctx(key)
                    value = self._pad_to_block_ctx(value)
                sample = False
            else:
                key = self.cache["key"]
                value = self.cache["value"]
        return query, key, value, sample

    def prime_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        curr_ctx = hidden_states.shape[1]
        if last_encoder_hidden_states is not None:
            raise TypeError("last_encoder_hidden_states should be None")
        query, key, value = hidden_states.chunk(3, dim=2)
        if sample:
            if self._cache_len() < self._encoder_len:
                self._append_cache(key, value)
            if self._cache_len() > self._encoder_len:
                self._slice_cache(0, self._encoder_len)
            key, value = self.cache["key"], self.cache["value"]
            self.sample_t += curr_ctx
        return query, key, value, sample

    def decode_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        curr_ctx = hidden_states.shape[1]
        query = hidden_states
        if sample:
            if self.sample_t == 0:
                self.cache["key"], self.cache["value"] = self.c_enc_kv(
                    last_encoder_hidden_states.type_as(hidden_states)
                ).chunk(2, dim=2)
            key, value = self.cache["key"], self.cache["value"]
            self.sample_t += curr_ctx
        else:
            key, value = self.c_enc_kv(last_encoder_hidden_states.type_as(hidden_states)).chunk(2, dim=2)
        return query, key, value, sample

    def forward(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        curr_ctx = hidden_states.shape[1]
        hidden_states = self.c_attn(hidden_states)
        query, key, value, sample = self.qkv(
            hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=sample
        )
        attention_scores = self.attn(query, key, value, sample)
        if attention_scores.shape[1] != curr_ctx:
            offset = self._offset(curr_ctx)
            attention_scores = attention_scores[:, offset : offset + curr_ctx, :].contiguous()
        attention_scores = self.c_proj(attention_scores)
        return self.resid_dropout(attention_scores)

    @property
    def _encoder_len(self):
        encoder_len = self.encoder_len
        encoder_blocks = (encoder_len // self.blocks) + 1
        return encoder_blocks * self.blocks

    def _offset(self, curr_ctx):
        if self.attn_func == "dense_attn":
            return 0
        return (self.sample_t - curr_ctx) % self.block_ctx

    def _pad_to_block_ctx(self, hidden_states, query=False):
        seq_len = hidden_states.shape[1]
        offset = self._offset(seq_len) if query else 0
        n_blocks = (seq_len + offset + self.block_ctx - 1) // self.block_ctx
        pad = n_blocks * self.block_ctx - seq_len - offset
        if pad == 0 and offset == 0:
            return hidden_states
        else:
            return F.pad(hidden_states, (0, 0, offset, pad))

    def _cache_len(self):
        return 0 if "key" not in self.cache else self.cache["key"].shape[1]

    def _suff_cache_len(self):
        """
        Precondition:
            key and value are appended with the current context and self.sample_t reflects the 1-indexed sample
            location in the context.
        """
        previous_block_length = (self.sample_t - 1) % self.block_ctx + 1 + self.block_ctx
        REQUIRED_CACHE_LEN = {
            "dense_attn": self.sample_t,
            "block_attn": (self.sample_t - 1) % self.block_ctx + 1,
            "transpose_block_attn": self.sample_t,
            "prev_block_attn": self.sample_t if self.sample_t <= self.block_ctx else previous_block_length,
            "cross_attn": self.encoder_len,
            "prime_attn": min(self.sample_t, self._encoder_len),
        }

        return REQUIRED_CACHE_LEN[self.attn_func]

    def _slice_cache(self, start, end=None):
        self.cache["key"] = self.cache["key"][:, start:end]
        self.cache["value"] = self.cache["value"][:, start:end]

    def _append_cache(self, key, value):
        if "key" not in self.cache:
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            old_key, old_value = key, value
            key = torch.cat([self.cache["key"], old_key], dim=1)
            value = torch.cat([self.cache["value"], old_value], dim=1)
            del self.cache["key"]
            del self.cache["value"]
            del old_key
            del old_value
            self.cache["key"] = key
            self.cache["value"] = value
        return self.cache["key"], self.cache["value"]

    def del_cache(self):
        self.sample_t = 0
        if "key" in self.cache:
            del self.cache["key"]
        if "value" in self.cache:
            del self.cache["value"]
        self.cache = {}


class JukeboxBlock(nn.Module):
    def __init__(self, config, n_ctx, attn_func="dense_attn"):
        super().__init__()
        self.width = config.hidden_size
        self.attn = JukeboxAttention(config, n_ctx, attn_func=attn_func)

        self.layer_norm_0 = JukeboxLayerNorm(config.hidden_size)
        self.mlp = JukeboxMLP(config)
        self.layer_norm_1 = JukeboxLayerNorm(config.hidden_size)
        self.res_scale = 1.0 / config.num_layers if config.attn_res_scale else 1.0
        self.attn_func = attn_func

    def forward(self, hidden_states, last_encoder_hidden_states, sample=False):
        residuals = hidden_states
        hidden_states = self.layer_norm_0(hidden_states)
        hidden_states = self.attn(hidden_states, last_encoder_hidden_states, sample)

        output_states = self.layer_norm_1(residuals + hidden_states)
        output_states = self.mlp(output_states)
        if self.res_scale == 1.0:
            output = residuals + hidden_states + output_states
        else:
            output = residuals + self.res_scale * (hidden_states + output_states)
        return output


class JukeboxLayerStack(nn.Module):
    def __init__(self, config, n_ctx):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = config.hidden_size
        self.num_layers = config.num_layers
        self.blocks = config.blocks
        self.attention_pattern = config.attention_pattern
        if self.blocks is not None:
            self.block_ctx = n_ctx // self.blocks
        self.encoder_len = config.nb_relevant_lyric_tokens
        self.n_heads = config.n_heads

        # Orders of attn_func
        attention_pattern = ATTENTION_PATTERNS[self.attention_pattern]
        self._attn_mods = nn.ModuleList()
        for depth in range(self.num_layers):
            self._attn_mods.append(JukeboxBlock(config, n_ctx, attn_func=attention_pattern(depth)))

        self.saved_attn_weights = []

    def set_record_attn(self, record_attn):
        """
        Makes forward prop dump self-attention softmaxes to self.saved_attn_weights.

        Args:
            record_attn (`Union[bool,set]`):
                Either a set of layer indices indicating which layers to store, or a boolean value indicating Whether
                to dump all.
        """

        def _should_record_attn(layer_idx):
            if isinstance(record_attn, bool):
                return record_attn
            return layer_idx in record_attn

        for i, layer in enumerate(self._attn_mods):
            layer.attn.record_attn = _should_record_attn(i)

        if not record_attn:
            self.saved_attn_weights = []

    def forward(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        # Blocks
        for i, attn_layer in enumerate(self._attn_mods):
            if attn_layer.attn_func == "cross_attention":  # attend to the lyrics
                hidden_states = attn_layer(
                    hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=sample
                )
            else:
                hidden_states = attn_layer(hidden_states, last_encoder_hidden_states=None, sample=sample)
            if attn_layer.attn.record_attn:
                self.saved_attn_weights.append(attn_layer.attn.c_attn.weight)
        return hidden_states

    def del_cache(self):
        for attn_layer in self._attn_mods:
            attn_layer.attn.del_cache()


class JukeboxPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, width):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty((embed_dim, width)))

    def forward(self):
        pos_emb = self.pos_emb
        return pos_emb


class JukeboxConditionalAutoregressive(nn.Module):
    def __init__(
        self,
        config,
        n_ctx=None,
        embed_dim=None,
        audio_conditioning=False,
        metadata_conditioning=False,
        is_encoder=False,
    ):
        """
        Autoregressive model on either lyric tokens or music tokens, or both. The attention pattern should be properly
        set for each configuration.

        Args:
            config (`JukeboxPriorConfig`):
                Model configuration class with all the parameters of the model. Initializing with a config file does
                not load the weights associated with the model, only the configuration. Check out the
                [`~PreTrainedModel.from_pretrained`] method to load the model weights.
            n_ctx (`int`, *optional*):
                Number of tokens or lyrics tokens provided in a single pass.
            embed_dim (`int`, *optional*):
                Either equals to the dimension of the codebook, or the sum of n_vocab (lyrics) and codebook dimension,
                if the model combines lyrics and music tokens, or simply n_vocab if the model is a separate encoder
            audio_conditioning (`bool`, *optional*, defaults to `False`):
                Whether or not the prior supports conditioning on audio.
            metadata_conditioning (`bool`, *optional*, defaults to `False`):
                Whether or not the prior supports conditioning on artitst, genres, lyrics and timing.
            is_encoder (`bool`, *optional*, defaults to `False`):
                Whether the model is an encoder only model.
        """

        super().__init__()
        self.width = config.hidden_size
        self.num_layers = config.num_layers
        self.n_ctx = n_ctx if n_ctx is not None else config.n_ctx
        self.embed_dim = embed_dim if embed_dim is not None else config.music_vocab_size
        self.embed_tokens = nn.Embedding(self.embed_dim, config.hidden_size)
        self.embed_tokens_dropout = nn.Dropout(config.emb_dropout)
        self.metadata_conditioning = metadata_conditioning
        self.audio_conditioning = audio_conditioning
        if not metadata_conditioning:
            self.start_token = nn.Parameter(torch.empty((1, config.hidden_size)))
        self.pos_emb = JukeboxPositionalEmbedding(self.n_ctx, config.hidden_size)
        self.pos_emb_dropout = nn.Dropout(config.emb_dropout)

        self.transformer = JukeboxLayerStack(config, n_ctx=self.n_ctx)
        self.is_encoder = is_encoder
        self.encoder_len = config.nb_relevant_lyric_tokens

        if config.merged_decoder:
            # Merged piped model uses this setup
            self.add_cond_after_transformer = False
            self.share_embed_tokens_fc_proj_out = False
        else:
            self.add_cond_after_transformer = True
            self.share_embed_tokens_fc_proj_out = True

        if not is_encoder:
            self.fc_proj_out = nn.Linear(config.hidden_size, self.embed_dim, bias=False)
            if self.share_embed_tokens_fc_proj_out:
                self.fc_proj_out.weight = self.embed_tokens.weight
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        tokens,
        audio_conditioning=None,
        metadata_conditioning=None,
        last_encoder_hidden_states=None,
        get_preds=False,
        get_acts=False,
        get_sep_loss=False,
    ):
        """
        Args:
            tokens (`torch.tensor`):
                Can represent music tokens, lyrics tokens or both, depending on the configuration.
        """
        # Preprocess.
        batch_size = tokens.shape[0]
        with torch.no_grad():
            tokens = tokens.view(batch_size, -1).long()

        if not self.audio_conditioning:
            audio_conditioning = torch.zeros(
                (batch_size, 1, self.width),
                device=tokens.device,
                dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype,
            )

        target = tokens  # Target
        hidden_states = self.embed_tokens(tokens)
        # Shift by 1, and fill in start token
        hidden_states = torch.cat((hidden_states[:, -1:], hidden_states[:, :-1]), dim=1)
        if self.metadata_conditioning:
            hidden_states[:, 0] = metadata_conditioning.view(batch_size, self.width)
        else:
            hidden_states[:, 0] = self.start_token

        hidden_states = (
            self.embed_tokens_dropout(hidden_states) + self.pos_emb_dropout(self.pos_emb()) + audio_conditioning
        )  # Pos emb and dropout

        hidden_states = self.transformer(
            hidden_states, last_encoder_hidden_states=last_encoder_hidden_states
        )  # Transformer
        if self.add_cond_after_transformer:  # Piped doesn't add x_cond
            hidden_states = hidden_states + audio_conditioning

        activations = hidden_states
        if self.is_encoder:
            return hidden_states

        hidden_states = self.fc_proj_out(hidden_states)  # Predictions
        loss_fn = nn.CrossEntropyLoss()
        if get_sep_loss:
            lyric_hidden_states = hidden_states[:, : self.encoder_len].reshape(-1, self.embed_dim)
            token_hidden_states = hidden_states[:, self.encoder_len :].reshape(-1, self.embed_dim)

            lyric_loss = loss_fn(lyric_hidden_states, target[:, : self.encoder_len].reshape(-1)) / np.log(2.0)
            music_token_loss = loss_fn(token_hidden_states, target[:, self.encoder_len :].reshape(-1)) / np.log(2.0)

            loss = (lyric_loss, music_token_loss)  # Note order! Lyric is first
        else:
            loss = loss_fn(hidden_states.view(-1, self.embed_dim), target.view(-1)) / np.log(2.0)  # Loss

        if get_preds:
            return loss, hidden_states
        elif get_acts:
            return loss, activations
        else:
            return loss, None

    def get_emb(self, sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning):
        if sample_t == 0:
            hidden_states = torch.empty(n_samples, 1, self.width, dtype=self.embed_tokens.weight.dtype).to(
                self.embed_tokens.weight.device
            )
            if self.metadata_conditioning:
                hidden_states[:, 0] = metadata_conditioning.view(n_samples, self.width)
            else:
                hidden_states[:, 0] = self.start_token
        else:
            hidden_states = self.embed_tokens(tokens)
        if audio_conditioning.shape == (n_samples, self.n_ctx, self.width):
            cond = audio_conditioning[:, sample_t : sample_t + 1, :]
        else:
            cond = audio_conditioning
        # Pos emb, dropout is identity at eval time
        hidden_states = hidden_states + self.pos_emb()[sample_t : sample_t + 1] + cond
        return hidden_states, cond

    def sample(
        self,
        n_samples,
        audio_conditioning=None,
        metadata_conditioning=None,
        last_encoder_hidden_states=None,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        get_preds=False,
        sample_tokens=None,
    ):
        if sample_tokens is None:
            sample_tokens = self.n_ctx

        if not self.audio_conditioning:
            audio_conditioning = torch.zeros(
                (n_samples, 1, self.width), dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype
            ).to(self.fc_proj_out.device)

        with torch.no_grad():
            sampled_tokens = []
            tokens = None
            if get_preds:
                preds = []

            iter = tqdm(range(0, sample_tokens), leave=False)
            for sample_t in iter:
                iter.set_description(f"Ancestral sampling {sample_tokens} music tokens", refresh=True)
                hidden_states, cond = self.get_emb(
                    sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning
                )

                hidden_states = self.transformer(
                    hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=True
                )
                if self.add_cond_after_transformer:
                    hidden_states = hidden_states + cond
                hidden_states = self.fc_proj_out(hidden_states)  # Predictions
                if get_preds:
                    preds.append(hidden_states.clone())
                # Adjust logits
                hidden_states = hidden_states / temp
                hidden_states = filter_logits(hidden_states, top_k=top_k, top_p=top_p)
                # Sample and replace hidden_states
                tokens = torch.distributions.Categorical(logits=hidden_states).sample()
                sampled_tokens.append(tokens.clone())

            del tokens
            self.transformer.del_cache()

            tokens = torch.cat(sampled_tokens, dim=1)
            if get_preds:
                preds = torch.cat(preds, dim=1)
        if get_preds:
            return tokens, preds
        else:
            return tokens

    def split_chunks(self, length, chunk_size):
        n_passes = (length + chunk_size - 1) // chunk_size
        chunk_sizes = [*[chunk_size] * (n_passes - 1), (length - 1) % chunk_size + 1]
        return chunk_sizes

    def primed_sample(
        self,
        n_samples,
        lyric_and_music_tokens,
        audio_conditioning=None,
        metadata_conditioning=None,
        last_encoder_hidden_states=None,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        get_preds=False,
        chunk_size=None,
        sample_tokens=None,
    ):
        if sample_tokens is None:
            sample_tokens = self.n_ctx
        # Preprocess.
        batch_size = lyric_and_music_tokens.shape[0]
        with torch.no_grad():
            lyric_and_music_tokens = lyric_and_music_tokens.view(batch_size, -1).long()

        sampled_audio = torch.split(lyric_and_music_tokens, 1, dim=1)
        sampled_audio = list(sampled_audio)

        if not self.audio_conditioning:
            audio_conditioning = torch.zeros(
                (n_samples, 1, self.width), dtype=self.transformer._attn_mods[0].mlp.c_fc.weight.dtype
            ).to(lyric_and_music_tokens.device)

        with torch.no_grad():
            if get_preds:
                preds = []

            # Fill up key/value cache for past context by running forward pass.
            # We do so in chunks instead of doing the whole past in one forward pass to reduce max memory usage.
            if chunk_size is None:
                chunk_size = len(sampled_audio)
            chunk_sizes = self.split_chunks(len(sampled_audio), chunk_size)
            x_primes = []
            start = 0
            token = None

            for current_chunk_size in tqdm(chunk_sizes, desc="Preparing past key value", leave=False):
                sampled_audio_prime, conds_prime = [], []
                for sample_t in range(start, start + current_chunk_size):
                    x_prime, cond_prime = self.get_emb(
                        sample_t, n_samples, token, audio_conditioning, metadata_conditioning
                    )
                    token = sampled_audio[sample_t]
                    sampled_audio_prime.append(x_prime)
                    conds_prime.append(cond_prime)
                start = start + current_chunk_size
                x_prime, cond_prime = torch.cat(sampled_audio_prime, dim=1), torch.cat(conds_prime, dim=1)
                del sampled_audio_prime
                del conds_prime
                if not get_preds:
                    del cond_prime
                x_prime = self.transformer(x_prime, last_encoder_hidden_states=last_encoder_hidden_states, sample=True)

                if get_preds:
                    if self.add_cond_after_transformer:
                        x_prime = x_prime + cond_prime
                    del cond_prime
                    x_primes.append(x_prime)
                else:
                    del x_prime

            if get_preds:
                x_prime = torch.cat(x_primes, dim=1)
                x_prime = self.fc_proj_out(x_prime)  # Predictions
                preds.append(x_prime)

            # the input of the encoder and decoder can be merged into (lyrics, music tokens)
            input_tokens = sampled_audio[-1]

            itererator = tqdm(
                range(len(sampled_audio), sample_tokens),
                desc=f"Sampling {len(range(len(sampled_audio), sample_tokens))} music tokens",
                leave=False,
            )
            for sample_t in itererator:
                hidden_states, cond = self.get_emb(
                    sample_t, n_samples, input_tokens, audio_conditioning, metadata_conditioning
                )

                hidden_states = self.transformer(
                    hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=True
                )
                if self.add_cond_after_transformer:
                    hidden_states = hidden_states + cond
                hidden_states = self.fc_proj_out(hidden_states)  # Predictions
                if get_preds:
                    preds.append(hidden_states)
                # Adjust logits
                hidden_states = hidden_states / temp
                hidden_states = filter_logits(hidden_states, top_k=top_k, top_p=top_p)
                # only music tokens are sampled
                music_tokens = torch.distributions.Categorical(logits=hidden_states).sample()
                sampled_audio.append(music_tokens.clone())
                input_tokens = music_tokens

            del input_tokens, music_tokens
            self.transformer.del_cache()

            music_tokens = torch.cat(sampled_audio, dim=1)
            if get_preds:
                preds = torch.cat(preds, dim=1)
        if get_preds:
            return music_tokens, preds
        else:
            return music_tokens


class JukeboxMusicTokenConditioner(nn.Module):
    """
    The `JukeboxMusicTokenConditioner` takes music tokens as an input (corresponding to the codes of the VQVAE's
    codebook) and upsamples it using a single layer of decoder convolution block (the same is used in the VQVAE).
    """

    def __init__(self, config, level):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.music_vocab_size, config.hidden_size)
        config.embed_dim = config.music_vocab_size  # setting correct argument for the `JukeboxDecoder`

        self.upsampler = JukeboxDecoderConvBock(
            config,
            config.hidden_size,
            config.res_conv_width,
            config.res_conv_depth,
            config.res_downs_t[level],
            config.res_strides_t[level],
            reverse_dilation=False,
        )
        self.layer_norm = JukeboxLayerNorm(config.hidden_size)

    def forward(self, music_tokens, raw_audio_conditioning=None):
        """
        Args:
            music_tokens (`torch.LongTensor`):
                Music tokens form the upper level in range(nb_discrete_codes)
            raw_audio_conditioning (`torch.LongTensor`, *optional*):
                Audio used when primed sampling, raw audio information that conditions the generation
        """
        if raw_audio_conditioning is None:
            raw_audio_conditioning = 0.0
        # Embed music_tokens
        music_tokens = music_tokens.long()
        hidden_states = self.embed_tokens(music_tokens)
        hidden_states = hidden_states + raw_audio_conditioning

        # Run conditioner
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.upsampler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class JukeboxRangeEmbedding(nn.Module):
    """
    The `JukeboxRangeEmbedding` interpolate the given [pos_start, pos_end] to obtain an equivalent of time positional
    embedding of length `n_ctx`.

    Binning process : For each pos in position tensor, find its bin [start,end) mapped to [0,1,...,bins-1] [start,end)
    -> [0,1) -> [0, bins) -> floor -> [0,...,bins-1] NOTE: Open ended interval on right, so start <= pos < end, not <=
    end
    """

    def __init__(self, n_time, embed_dim, range, out_width, clamp=False):
        super().__init__()
        self.n_time = n_time
        self.embed_dim = embed_dim
        self.emb = nn.Embedding(embed_dim, out_width)
        self.pos_min, self.pos_max = range
        self.clamp = clamp

    def forward(self, pos_start, pos_end=None):
        # Check if [pos_start,pos_end] in [pos_min, pos_max)
        if not len(pos_start.shape) == 2:
            raise TypeError(f"Expected shape with 2 dims, got {pos_start.shape}")
        if not (self.pos_min <= pos_start).all() and (pos_start < self.pos_max).all():
            raise TypeError(f"Range is [{self.pos_min},{self.pos_max}), got {pos_start}")

        pos_start = pos_start.float()
        if pos_end is not None:
            if self.clamp:
                pos_end = pos_end.clamp(self.pos_min, self.pos_max)

            pos_end = pos_end.float()
        # Interpolate so that [pos_start, ..., pos_end] <-> position tensor of length n_ctx
        n_time = self.n_time
        if n_time != 1:
            interpolation = (
                torch.arange(0, n_time, dtype=torch.float, device=pos_start.device).view(1, n_time) / n_time
            )
            position = pos_start + (pos_end - pos_start) * interpolation
        else:
            position = pos_start

        # Bin each value to bins_
        # [0,1) -> [0,1..,embed_dim) -> [0,1...,embed_dim-1
        normalised_position = (position - self.pos_min) / (self.pos_max - self.pos_min)
        bins_ = (self.embed_dim * normalised_position).floor().long().detach()
        return self.emb(bins_)


class JukeboxLabelConditioner(nn.Module):
    def __init__(self, config, include_time_signal):
        super().__init__()

        embed_dim = config.hidden_size
        timing_dims = config.timing_dims
        sampling_rate = config.sampling_rate
        nb_genres, nb_artists = config.metadata_dims
        music_tokens_shape = config.n_ctx

        self.max_nb_genres = config.max_nb_genres
        self.bow_genre_emb = nn.Embedding(nb_genres, embed_dim)
        self.artist_emb = nn.Embedding(nb_artists, embed_dim)
        self.include_time_signal = include_time_signal
        if self.include_time_signal:
            total_length_range = (config.min_duration * sampling_rate, config.max_duration * sampling_rate)
            absolute_pos_range = (0.0, config.max_duration * sampling_rate)
            relative_pos_range = (0.0, 1.0)
            self.total_length_emb = JukeboxRangeEmbedding(1, timing_dims, total_length_range, embed_dim)
            self.absolute_pos_emb = JukeboxRangeEmbedding(
                music_tokens_shape, timing_dims, absolute_pos_range, embed_dim
            )
            self.relative_pos_emb = JukeboxRangeEmbedding(
                music_tokens_shape, timing_dims, relative_pos_range, embed_dim, clamp=True
            )

    def forward(self, metadata):
        total_length = metadata[:, 0:1]
        offset = metadata[:, 1:2]
        length = metadata[:, 2:3]
        artist = metadata[:, 3:4]
        genre = metadata[:, 4:]

        # Start embedding of length 1
        artist_emb = self.artist_emb(artist)
        # Empty genre slots are denoted by -1. We mask these out.
        mask = (genre >= 0).float().unsqueeze(2)
        genre_emb = (self.bow_genre_emb(genre.clamp(0)) * mask).sum(dim=1, keepdim=True)
        start_emb = genre_emb + artist_emb

        # Pos embedding of length n_ctx
        if self.include_time_signal:
            start, end = offset, offset + length
            total_length = total_length.float()
            start = start.float()
            end = end.float()
            pos_emb = (
                self.total_length_emb(total_length)
                + self.absolute_pos_emb(start, end)
                + self.relative_pos_emb(start / total_length, end / total_length)
            )
        else:
            pos_emb = None
        return start_emb, pos_emb


class JukeboxPrior(PreTrainedModel):
    """
    The JukeboxPrior class, which is a wrapper around the various conditioning and the transformer. JukeboxPrior can be
    seen as language models trained on music. They model the next `music token` prediction task. If a (lyric) `encoderù
    is defined, it also models the `next character` prediction on the lyrics. Can be conditioned on timing, artist,
    genre, lyrics and codes from lower-levels Priors.

    Args:
        config (`JukeboxPriorConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        level (`int`, *optional*):
            Current level of the Prior. Should be in range `[0,nb_priors]`.
        nb_priors (`int`, *optional*, defaults to 3):
            Total number of priors.
        vqvae_encoder (`Callable`, *optional*):
            Encoding method of the VQVAE encoder used in the forward pass of the model. Passing functions instead of
            the vqvae module to avoid getting the parameters.
        vqvae_decoder (`Callable`, *optional*):
            Decoding method of the VQVAE decoder used in the forward pass of the model. Passing functions instead of
            the vqvae module to avoid getting the parameters.
    """

    config_class = JukeboxPriorConfig

    def _init_weights(self, module):
        init_scale = self.config.init_scale

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        elif isinstance(module, JukeboxConv1D):
            if self.config.zero_out:
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        elif isinstance(module, JukeboxPositionalEmbedding):
            module.pos_emb.data.normal_(mean=0.0, std=0.01 * init_scale)
        elif isinstance(module, JukeboxRangeEmbedding):
            module.emb.weight.data.normal_(mean=0.0, std=0.01 * init_scale)
        elif isinstance(module, JukeboxConditionalAutoregressive) and hasattr(module, "lm_head"):
            module.lm_head.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        elif isinstance(module, JukeboxConditionalAutoregressive) and hasattr(module, "start_token"):
            module.start_token.data.normal_(mean=0.0, std=0.01 * init_scale)
        elif isinstance(module, JukeboxResConv1DBlock) and self.config.zero_out:
            module.conv1d_2.weight.data.zero_()
            module.conv1d_2.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, config: JukeboxPriorConfig, level=None, nb_priors=3, vqvae_encoder=None, vqvae_decoder=None):
        super().__init__(config)
        # Passing functions instead of the vqvae module to avoid getting params, only used in the
        # forward loop
        self.vqvae_encoder = vqvae_encoder
        self.vqvae_decoder = vqvae_decoder

        self.levels = nb_priors
        self.level = level if level is not None else config.level

        self.base_model_prefix = f"priors.{self.level}"

        self.n_ctx = config.n_ctx

        self.lyric_conditioning = config.nb_relevant_lyric_tokens > 0
        self.nb_relevant_lyric_tokens = config.nb_relevant_lyric_tokens
        self.encoder_loss_fraction = config.encoder_loss_fraction

        # Audio conditioning : conditioning on music tokens (either from audio or from previous levels or both)
        self.audio_conditioning = self.level != 0
        self.cond_level = self.level - 1
        if self.audio_conditioning:
            self.conditioner_blocks = JukeboxMusicTokenConditioner(config, self.level)

        # metadata conditioning : contioning on timing, genres, and artist
        self.metadata_conditioning = config.metadata_conditioning
        if self.metadata_conditioning:
            self.metadata_embedding = JukeboxLabelConditioner(config, include_time_signal=not self.audio_conditioning)

        # define encoder-decoder or encoder and decoder
        self.is_encoder_decoder = config.is_encoder_decoder
        if config.is_encoder_decoder:
            # encoder-decoder transformer
            self.input_shapes = [config.nb_relevant_lyric_tokens, config.n_ctx]
            self.embed_dim_shift = [0, config.lyric_vocab_size]
            self.width = config.hidden_size

            self.nb_relevant_lyric_tokens = config.nb_relevant_lyric_tokens

            self.prior = JukeboxConditionalAutoregressive(
                config,
                n_ctx=config.nb_relevant_lyric_tokens + config.n_ctx,
                embed_dim=config.lyric_vocab_size + config.music_vocab_size,
                audio_conditioning=(self.audio_conditioning or self.metadata_conditioning),
                metadata_conditioning=True,
            )

        else:
            # Separate encoder-decoder transformer
            encoder_config = config.encoder_config

            if self.nb_relevant_lyric_tokens != 0 and self.lyric_conditioning:
                self.lyric_acts_width = encoder_config.hidden_size
                self.encoder_width = config.hidden_size
                self.encoder_dim = config.lyric_vocab_size
                self.encoder = JukeboxConditionalAutoregressive(
                    encoder_config,
                    n_ctx=self.nb_relevant_lyric_tokens,
                    embed_dim=self.encoder_dim,
                    audio_conditioning=False,
                    metadata_conditioning=False,
                    is_encoder=True,
                )
                self.encoder.proj_in = JukeboxConv1D(encoder_config.hidden_size, config.hidden_size)
                self.encoder.final_layer_norm = JukeboxLayerNorm(config.hidden_size)
                self.encoder.lm_head = nn.Linear(config.hidden_size, config.lyric_vocab_size, bias=False)
            else:
                self.nb_relevant_lyric_tokens = 0

            # decoder model on the tokens
            self.prior = JukeboxConditionalAutoregressive(
                config,
                audio_conditioning=(self.audio_conditioning or self.metadata_conditioning),
                metadata_conditioning=self.metadata_conditioning,
            )

        self.next_token_prediction_loss_dims = config.n_ctx
        self.total_loss_dims = self.nb_relevant_lyric_tokens + self.next_token_prediction_loss_dims

        self.downsamples = [stride**down for stride, down in zip(config.res_strides_t, config.res_downs_t)]
        self.cond_downsample = self.downsamples[self.level] if self.level != 0 else None
        self.raw_to_tokens = np.prod(self.downsamples[: nb_priors - self.level])
        self.sample_length = self.n_ctx * self.raw_to_tokens

        logger.info(
            f"Level:{self.level}, Cond downsample:{self.cond_downsample}, Raw to tokens:{self.raw_to_tokens}, Sample"
            f" length:{self.sample_length}"
        )

    def get_metadata(self, labels, start, total_length, offset, get_indices=False):
        metadata = labels.clone()
        metadata[:, 0] = total_length
        # Set sample_length to match this level
        metadata[:, 2] = int(self.sample_length)

        # Set offset
        metadata[:, 1:2] = int(offset * self.raw_to_tokens) + int(start * self.raw_to_tokens)
        # here since metadata has the full token_list, we just need to selected the ones that are relevant

        # Set lyric tokens
        metadata, indices = self.set_metadata_lyric_tokens(metadata)
        if get_indices:
            return metadata, indices
        else:
            return metadata

    def set_metadata_lyric_tokens(self, labels):
        """
        Processes the full labels to only retrieve the relevant lyric tokens and keep the metadata conditioning tokens.
        """
        if self.nb_relevant_lyric_tokens > 0:
            tokens_list = torch.zeros(
                (labels.shape[0], self.nb_relevant_lyric_tokens), dtype=torch.long, device=labels.device
            )
            indices_list = []  # what's the index of each current character in original array
            for idx in range(labels.shape[0]):
                full_tokens = labels.clone()[:, 4 + self.metadata_embedding.max_nb_genres :]
                total_length, offset, duration = labels[idx, 0], labels[idx, 1], labels[idx, 2]
                tokens, indices = get_relevant_lyric_tokens(
                    full_tokens, self.nb_relevant_lyric_tokens, total_length, offset, duration
                )
                tokens_list[idx, :] = tokens
                indices_list.append(indices)

            return (
                torch.cat((labels[:, : 4 + self.metadata_embedding.max_nb_genres], tokens_list), dim=-1),
                indices_list,
            )
        else:
            return labels, None

    def get_music_tokens_conds(self, music_tokens, start, end):
        """
        Extracts current level's conditioning music tokens.
        """
        if self.level != 0:
            music_tokens_cond = music_tokens[self.level - 1]
            music_tokens = music_tokens_cond[:, start // self.cond_downsample : end // self.cond_downsample]
            missing_cond_len = self.n_ctx // self.cond_downsample - music_tokens_cond[-1].shape[-1]
            if missing_cond_len > 0:
                init_cond = torch.zeros(1, missing_cond_len).to(music_tokens_cond.device)
                music_tokens_cond = torch.cat((music_tokens_cond, init_cond), dim=-1).long()
            music_tokens_conds = [music_tokens_cond]
        else:
            music_tokens_conds = None
        return music_tokens_conds

    def prior_preprocess(self, tokens, conds):
        """
        Shifts the input tokens to account for the dictionary merge. The embed_dim_shift give by how much the music
        tokens should be shifted by. It is equal to `lyric_vocab_size`.
        """
        batch_size = tokens[0].shape[0]
        for i in range(len(tokens)):
            tokens[i] = (tokens[i] + int(self.embed_dim_shift[i])).view(batch_size, -1)

        for i in range(len(conds)):
            if conds[i] is None:
                conds[i] = torch.zeros(
                    (batch_size, self.input_shapes[i], self.width), dtype=tokens[0].dtype, device=tokens[0].device
                )

        return torch.cat(tokens, dim=1), torch.cat(conds, dim=1)

    def prior_postprocess(self, tokens):
        """
        Shifts back the input tokens if the model uses an encoder decoder architecture. As the embedding layer is
        shared, `prior_embed_dim_shift` shifts the music token ids by `lyric_vocab_size`. Only returns the music
        tokens.
        """
        batch_size = tokens.shape[0]
        dims = (self.input_shapes[0], tokens.shape[1] - self.input_shapes[0])
        tokens = list(torch.split(tokens, dims, dim=1))

        # Some of the input tokens might be shifted to take into account the voccabulary fusion
        for i in range(len(tokens)):
            bins_shift = int(self.embed_dim_shift[i])
            tokens[i] = (tokens[i] - bins_shift).view(batch_size, -1)
            tokens[i] = torch.clamp(tokens[i], min=0)
            # If not masking loss, model may have generated lyric/midi tokens which are now shifted <0 by bin_shift
        return tokens[-1]

    def embed_tokens(self, music_tokens_conds):
        """
        Embeds the upper level music tokens and upsamples them to provide as audio conditioning.
        """
        music_tokens_conds = music_tokens_conds[: self.cond_level + 1]
        audio_conditioning = None
        for music_tokens_cond, conditioner_block in reversed(list(zip(music_tokens_conds, [self.conditioner_blocks]))):
            audio_conditioning = conditioner_block(music_tokens_cond, audio_conditioning)
        return audio_conditioning

    def encode(self, hidden_states, start_level=None, end_level=None, bs_chunks=1):
        """
        Encodes the hidden states (raw audio) using the VQVAE's encoder. Returns latent_states.
        """
        if start_level is None:
            start_level = self.level
        if end_level is None:
            end_level = self.levels
        # Get latents
        with torch.no_grad():
            latent_states = self.vqvae_encoder(
                hidden_states, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks
            )
        return latent_states

    def decode(self, music_tokens, start_level=None, end_level=None, bs_chunks=1):
        """
        Usamples the sequence of codebook vectors to a raw audio.
        """
        if start_level is None:
            start_level = self.level
        if end_level is None:
            end_level = self.levels
        with torch.no_grad():
            output = self.vqvae_decoder(
                music_tokens, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks
            )
        return output

    def get_cond(self, music_tokens_conds, metadata):
        """
        Converts the input tokens to input_embeddings. Splits the lyrics form the rest of the metadata. Lyric tokens
        can be None.
        """
        if metadata is not None:
            n_labels = metadata.shape[1] - self.nb_relevant_lyric_tokens
            metadata, lyric_tokens = metadata[:, :n_labels], metadata[:, n_labels:]
        else:
            metadata, lyric_tokens = None, None
        metadata_conditioning, metadata_pos = (
            self.metadata_embedding(metadata) if self.metadata_conditioning else (None, None)
        )
        audio_conditioning = self.embed_tokens(music_tokens_conds) if self.audio_conditioning else metadata_pos
        return audio_conditioning, metadata_conditioning, lyric_tokens

    def sample(
        self,
        n_samples,
        music_tokens=None,
        music_tokens_conds=None,
        metadata=None,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        chunk_size=None,
        sample_tokens=None,
    ):
        """
        Ancestral/Prime sampling a window of tokens using the provided conditioning and metadatas.

        Args:
            n_samples (`int`):
                Number of samples to generate.
            music_tokens (`list[torch.LongTensor]`, *optional*):
                Previously generated tokens at the current level. Used as context for the generation.
            music_tokens_conds (`list[torch.FloatTensor]`, *optional*):
                Upper-level music tokens generated by the previous prior model. Is `None` if the generation is not
                conditioned on the upper-level tokens.
            metadata (`list[torch.LongTensor]`, *optional*):
                List containing the metadata tensor with the artist, genre and the lyric tokens.
            temp (`float`, *optional*, defaults to 1.0):
                Sampling temperature.
            top_k (`int`, *optional*, defaults to 0):
                Top k probabilities used for filtering.
            top_p (`float`, *optional*, defaults to 0.0):
                Top p probabilities used for filtering.
            chunk_size (`int`, *optional*):
                Size of the chunks used to prepare the cache of the transformer.
            sample_tokens (`int`, *optional*):
                Number of tokens to sample.

        """
        no_past_context = music_tokens is None or music_tokens.shape[1] == 0
        name = {True: "Ancestral", False: "Primed"}[no_past_context]
        logger.info(f"{name} sampling {n_samples} samples with temp={temp}, top_k={top_k}, top_p={top_p}")

        with torch.no_grad():
            # Currently audio_conditioning only uses immediately above layer
            audio_conditioning, metadata_conditioning, lyric_tokens = self.get_cond(music_tokens_conds, metadata)
            if self.is_encoder_decoder:
                if no_past_context:  # the prime_sample function will be used with music_tokens set to None
                    lyric_and_music_tokens, audio_conditioning = self.prior_preprocess(
                        [lyric_tokens], [None, audio_conditioning]
                    )
                else:
                    lyric_and_music_tokens, audio_conditioning = self.prior_preprocess(
                        [lyric_tokens, music_tokens], [None, audio_conditioning]
                    )
                if sample_tokens is not None:
                    sample_tokens += self.nb_relevant_lyric_tokens
                music_tokens = self.prior.primed_sample(
                    n_samples,
                    lyric_and_music_tokens,
                    audio_conditioning,
                    metadata_conditioning,
                    temp=temp,
                    top_k=top_k,
                    top_p=top_p,
                    chunk_size=chunk_size,
                    sample_tokens=sample_tokens,
                )
                music_tokens = self.prior_postprocess(music_tokens)
            else:
                last_encoder_hidden_states = self.get_encoder_states(lyric_tokens, sample=True)
                if no_past_context:
                    music_tokens = self.prior.sample(
                        n_samples,
                        audio_conditioning,
                        metadata_conditioning,
                        last_encoder_hidden_states,
                        temp=temp,
                        top_k=top_k,
                        top_p=top_p,
                        sample_tokens=sample_tokens,
                    )
                else:
                    music_tokens = self.prior.primed_sample(
                        n_samples,
                        music_tokens,
                        audio_conditioning,
                        metadata_conditioning,
                        last_encoder_hidden_states,
                        temp=temp,
                        top_k=top_k,
                        top_p=top_p,
                        chunk_size=chunk_size,
                        sample_tokens=sample_tokens,
                    )
        return music_tokens

    def get_encoder_states(self, lyric_tokens, sample=False):
        """
        Retrieve the last hidden_states of the lyric encoder that will be attended to by the decoder. Forwards through
        the lyric encoder.
        """
        if self.nb_relevant_lyric_tokens != 0 and self.lyric_conditioning:
            if sample:
                self.encoder = self.encoder.to(lyric_tokens.device)
            lyric_acts = self.encoder(lyric_tokens, None, None, None)
            lyric_acts = self.encoder.proj_in(lyric_acts)
            last_encoder_hidden_states = self.encoder.final_layer_norm(lyric_acts)
        else:
            last_encoder_hidden_states = None
        return last_encoder_hidden_states

    def get_encoder_loss(self, last_encoder_hidden_states, target_lyrics):
        """
        Computes the loss for the lyric encoder: next lyric token prediction.
        """
        if self.lyric_conditioning:
            last_encoder_hidden_states = self.encoder.lm_head(last_encoder_hidden_states)
            encoder_loss = nn.functional.cross_entropy(
                last_encoder_hidden_states.view(-1, self.encoder_dim), target_lyrics.view(-1)
            ) / np.log(2.0)
        else:
            encoder_loss = torch.tensor(0.0, device=last_encoder_hidden_states.device)
        return encoder_loss

    def forward_tokens(
        self, music_tokens, music_tokens_conds=[], metadata=None, get_preds=False, get_attn_weights=False
    ):
        """
        Applies a forward pass using the conditioning tokens. Different from the classic forward as it does not use the
        vqvae's encoding layers.
        """
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        audio_conditioning, metadata_conditioning, lyric_tokens = self.get_cond(music_tokens_conds, metadata)

        if self.is_encoder_decoder:  # the preprocess returns the full tokens (Lyrics and Music tokens), shifted
            tokens, audio_conditioning = self.prior_preprocess(
                [lyric_tokens, music_tokens], [None, audio_conditioning]
            )
            (encoder_loss, next_token_prediction_loss), preds = self.prior(
                tokens, audio_conditioning, metadata_conditioning, get_sep_loss=True, get_preds=get_preds
            )
        else:
            last_encoder_hidden_states = self.get_encoder_states(lyric_tokens)
            encoder_loss = self.get_encoder_loss(last_encoder_hidden_states, lyric_tokens)
            next_token_prediction_loss, preds = self.prior(
                music_tokens,
                audio_conditioning,
                metadata_conditioning,
                last_encoder_hidden_states,
                get_preds=get_preds,
            )
        loss = self.encoder_loss_fraction * encoder_loss * self.nb_relevant_lyric_tokens / self.total_loss_dims
        loss += next_token_prediction_loss * self.next_token_prediction_loss_dims / self.total_loss_dims

        metrics = {
            "bpd": next_token_prediction_loss.detach().clone(),
            "encoder_loss": encoder_loss.detach().clone(),
            "next_token_prediction_loss": next_token_prediction_loss.detach().clone(),
        }
        if get_preds:
            metrics["preds"] = preds.detach().clone()
        if get_attn_weights:
            saved_attn_weights = self.prior.transformer.saved_attn_weights
            self.prior.transformer.set_record_attn(False)
            return saved_attn_weights
        else:
            return loss, metrics

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: Optional[list[torch.LongTensor]],
        decode: Optional[bool] = False,
        get_preds: Optional[bool] = False,
    ) -> list[torch.Tensor]:
        """
        Encode the hidden states using the `vqvae` encoder, and then predicts the next token in the `forward_tokens`
        function. The loss is the sum of the `encoder` loss and the `decoder` loss.

        Args:
            hidden_states (`torch.Tensor`):
                Hidden states which should be raw audio
            metadata (`list[torch.LongTensor]`, *optional*):
                List containing the metadata conditioning tensor with the lyric and the metadata tokens.
            decode (`bool`, *optional*, defaults to `False`):
                Whether or not to decode the encoded to tokens.
            get_preds (`bool`, *optional*, defaults to `False`):
                Whether or not to return the actual predictions of the model.
        """
        batch_size = hidden_states.shape[0]
        music_tokens, *music_tokens_conds = self.encode(hidden_states, bs_chunks=batch_size)
        loss, metrics = self.forward_tokens(
            music_tokens=music_tokens,
            music_tokens_conds=music_tokens_conds,
            metadata=metadata,
            get_preds=get_preds,
        )
        if decode:
            dequantised_states = self.decode([music_tokens, *music_tokens_conds])
        else:
            dequantised_states = None
        return dequantised_states, loss, metrics


class JukeboxPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = JukeboxConfig
    base_model_prefix = "jukebox"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, JukeboxPrior) or isinstance(module, JukeboxVQVAE):
            module.apply(module._init_weights)

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


JUKEBOX_SAMPLING_INPUT_DOCSTRING = r"""
            labels (`list[torch.LongTensor]` of length `n_sample`, and shape `(self.levels, self.config.max_nb_genre + lyric_sequence_length)` :
                List of metadata such as `artist_id`, `genre_id` and the full list of lyric tokens which are used to
                condition the generation.
            sampling_kwargs (`dict[Any]`):
                Various additional sampling arguments that are used by the `_sample` function. A detail list of the
                arguments can bee seen in the [`_sample`] function documentation.
"""


@add_start_docstrings(
    """The bare JUKEBOX Model used for music generation. 4 sampling techniques are supported : `primed_sample`, `upsample`,
    `continue_sample` and `ancestral_sample`. It does not have a `forward` method as the training is not end to end. If
    you want to fine-tune the model, it is recommended to use the `JukeboxPrior` class and train each prior
    individually.
    """,
    JUKEBOX_START_DOCSTRING,
)
class JukeboxModel(JukeboxPreTrainedModel):
    _no_split_modules = ["JukeboxBlock"]

    def __init__(self, config):
        super().__init__(config)
        vqvae_config = config.vqvae_config
        self.vqvae = JukeboxVQVAE(vqvae_config)
        self.set_shared_params(config)
        self.priors = nn.ModuleList(
            [JukeboxPrior(config.prior_configs[level], level) for level in range(config.nb_priors)]
        )

    def set_shared_params(self, model_config):
        """
        Initialises the parameters that are shared. This has to be done here because the list of `JukeboxPriorConfig`
        is nest, and is thus unreachable in the `from_dict` function
        """
        for config in model_config.prior_configs:
            config.sampling_rate = model_config.sampling_rate
            config.timing_dims = model_config.timing_dims
            config.min_duration = model_config.min_duration
            config.max_duration = model_config.max_duration
            config.max_nb_genres = model_config.max_nb_genres
            config.metadata_conditioning = model_config.metadata_conditioning

    def decode(self, music_tokens, start_level=0, end_level=None, bs_chunks=1):
        return self.vqvae.decode(music_tokens, start_level, end_level, bs_chunks)

    def encode(self, input_audio, start_level=0, end_level=None, bs_chunks=1):
        return self.vqvae.encode(input_audio, start_level, end_level, bs_chunks)

    def split_batch(self, obj, n_samples, split_size):
        n_passes = (n_samples + split_size - 1) // split_size
        if isinstance(obj, torch.Tensor):
            return torch.split(obj, split_size, dim=0)
        elif isinstance(obj, list):
            return list(zip(*[torch.split(item, split_size, dim=0) for item in obj]))
        elif obj is None:
            return [None] * n_passes
        else:
            raise TypeError("Unknown input type")

    # Sample a partial window of length<n_ctx with tokens_to_sample new tokens on level=level
    def sample_partial_window(
        self, music_tokens, labels, offset, sampling_kwargs, level, tokens_to_sample, max_batch_size
    ):
        prior = self.priors[level]
        sampled_tokens = music_tokens[level]
        n_ctx = prior.n_ctx
        nb_sampled_tokens = sampled_tokens.shape[1]
        if nb_sampled_tokens < n_ctx - tokens_to_sample:
            sampling_kwargs["sample_tokens"] = nb_sampled_tokens + tokens_to_sample
            start = 0
        else:
            sampling_kwargs["sample_tokens"] = n_ctx
            start = nb_sampled_tokens - n_ctx + tokens_to_sample

        return self.sample_single_window(music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size)

    # Sample a single window of length=n_ctx at position=start on level=level
    def sample_single_window(self, music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size):
        prior = self.priors[level]
        n_samples = music_tokens[0].shape[0]
        n_ctx = prior.n_ctx
        end = start + n_ctx
        # get music_tokens already sampled at current level
        previous_sampled_tokens = music_tokens[level][:, start:end]

        sample_tokens = sampling_kwargs.get("sample_tokens", None)
        if "sample_tokens" in sampling_kwargs:
            sample_tokens = end - start

        conditioning_tokens = previous_sampled_tokens.shape[1]
        new_tokens = sample_tokens - previous_sampled_tokens.shape[1]

        logger.info(
            f"Sampling {sample_tokens} tokens for [{start},{start + sample_tokens}]. Conditioning on"
            f" {conditioning_tokens} tokens"
        )

        if new_tokens <= 0:
            # Nothing new to sample
            return music_tokens

        # get music_tokens_conds from level above
        music_tokens_conds = prior.get_music_tokens_conds(music_tokens, start, end)
        # if there are no levels above should return None!

        # set metadata offset, sample_length and lyrics tokens
        metadata = prior.get_metadata(labels, start, self.total_length, offset)

        music_tokens_list = self.split_batch(previous_sampled_tokens, n_samples, max_batch_size)
        music_tokens_conds_list = self.split_batch(music_tokens_conds, n_samples, max_batch_size)
        metadata_list = self.split_batch(metadata, n_samples, max_batch_size)
        tokens = []
        iterator = tqdm(zip(music_tokens_list, music_tokens_conds_list, metadata_list), leave=False)
        for music_tokens_i, music_tokens_conds_i, metadata_i in iterator:
            name = ["Ancestral", "Primed"][music_tokens_i.shape[1] == 0]
            iterator.set_description(
                f"[prior level {level}] {name} Sampling {sample_tokens} tokens out of"
                f" {self.total_length // prior.raw_to_tokens}",
                refresh=True,
            )
            tokens_i = prior.sample(
                n_samples=music_tokens_i.shape[0],
                music_tokens=music_tokens_i,
                music_tokens_conds=music_tokens_conds_i,
                metadata=metadata_i,
                **sampling_kwargs,
            )
            tokens.append(tokens_i)
        sampled_tokens = torch.cat(tokens, dim=0)

        # Update music_tokens with new sample
        music_tokens_new = sampled_tokens[:, -new_tokens:]
        music_tokens[level] = torch.cat([music_tokens[level], music_tokens_new], dim=1)
        return music_tokens

    # Sample total_length tokens at level=level with hop_length=hop_length
    def sample_level(
        self, music_tokens, labels, offset, sampling_kwargs, level, total_length, hop_length, max_batch_size
    ):
        if total_length >= self.priors[level].n_ctx:
            iterator = get_starts(total_length, self.priors[level].n_ctx, hop_length)
            for start in iterator:
                music_tokens = self.sample_single_window(
                    music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size
                )

        else:
            music_tokens = self.sample_partial_window(
                music_tokens, labels, offset, sampling_kwargs, level, total_length, max_batch_size
            )
        return music_tokens

    @torch.no_grad()
    def _sample(
        self,
        music_tokens,
        labels,
        sample_levels,
        metas=None,
        chunk_size=32,
        sampling_temperature=0.98,
        lower_batch_size=16,
        max_batch_size=16,
        sample_length_in_seconds=24,
        compute_alignments=False,
        sample_tokens=None,
        offset=0,
        save_results=True,
        sample_length=None,
    ) -> list[torch.LongTensor]:
        """
        Core sampling function used to generate music tokens. Iterates over the provided list of levels, while saving
        the generated raw audio at each step.

        Args:
            music_tokens (`list[torch.LongTensor]`):
                A sequence of music tokens of length `self.levels` which will be used as context to continue the
                sampling process. Should have `self.levels` tensors, each corresponding to the generation at a certain
                level.
            labels (`list[torch.LongTensor]`):
                List of length `n_sample`, and shape `(self.levels, 4 + self.config.max_nb_genre +
                lyric_sequence_length)` metadata such as `artist_id`, `genre_id` and the full list of lyric tokens
                which are used to condition the generation.
            sample_levels (`list[int]`):
                List of the desired levels at which the sampling will be done. A level is equivalent to the index of
                the prior in the list of priors
            metas (`list[Any]`, *optional*):
                Metadatas used to generate the `labels`
            chunk_size (`int`, *optional*, defaults to 32):
                Size of a chunk of audio, used to fill up the memory in chunks to prevent OOM errors. Bigger chunks
                means faster memory filling but more consumption.
            sampling_temperature (`float`, *optional*, defaults to 0.98):
                Temperature used to adjust the randomness of the sampling.
            lower_batch_size (`int`, *optional*, defaults to 16):
                Maximum batch size for the lower level priors
            max_batch_size (`int`, *optional*, defaults to 16):
                Maximum batch size for the top level priors
            sample_length_in_seconds (`int`, *optional*, defaults to 24):
                Desired length of the generation in seconds
            compute_alignments (`bool`, *optional*, defaults to `False`):
                Whether or not to compute the alignment between the lyrics and the audio using the top_prior
            sample_tokens (`int`, *optional*):
                Precise number of tokens that should be sampled at each level. This is mostly useful for running dummy
                experiments
            offset (`int`, *optional*, defaults to 0):
                Audio offset used as conditioning, corresponds to the starting sample in the music. If the offset is
                greater than 0, the lyrics will be shifted take that intoaccount
            save_results (`bool`, *optional*, defaults to `True`):
                Whether or not to save the intermediate results. If `True`, will generate a folder named with the start
                time.
            sample_length (`int`, *optional*):
                Desired length of the generation in samples.

        Returns: torch.Tensor

        Example:

        ```python
        >>> from transformers import AutoTokenizer, JukeboxModel, set_seed
        >>> import torch

        >>> metas = dict(artist="Zac Brown Band", genres="Country", lyrics="I met a traveller from an antique land")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
        >>> model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()

        >>> labels = tokenizer(**metas)["input_ids"]
        >>> set_seed(0)
        >>> zs = [torch.zeros(1, 0, dtype=torch.long) for _ in range(3)]
        >>> zs = model._sample(zs, labels, [0], sample_length=40 * model.priors[0].raw_to_tokens, save_results=False)
        >>> zs[0]
        tensor([[1853, 1369, 1150, 1869, 1379, 1789,  519,  710, 1306, 1100, 1229,  519,
              353, 1306, 1379, 1053,  519,  653, 1631, 1467, 1229, 1229,   10, 1647,
             1254, 1229, 1306, 1528, 1789,  216, 1631, 1434,  653,  475, 1150, 1528,
             1804,  541, 1804, 1434]])
        ```
        """

        top_prior = self.priors[0]
        if sample_length is not None:
            total_length = sample_length
        else:
            total_length = (
                int(sample_length_in_seconds * self.config.sampling_rate) // top_prior.raw_to_tokens
            ) * top_prior.raw_to_tokens

        if sample_levels is None:
            sample_levels = range(len(self.priors))

        # total length of the signal, might be bit different from the actual generated length
        self.total_length = total_length
        for level in sample_levels:
            sampling_kwargs = {
                "temp": 0.99 if level == len(self.priors) - 1 else sampling_temperature,
                "chunk_size": chunk_size,
                "sample_tokens": sample_tokens,
            }
            # Set correct total_length, hop_length, labels and sampling_kwargs for level

            total_token_to_sample = total_length // self.priors[level].raw_to_tokens
            hop_length = int(self.config.hop_fraction[level] * self.priors[level].n_ctx)
            max_batch_size = lower_batch_size if level != sample_levels else max_batch_size
            music_tokens = self.sample_level(
                music_tokens,
                labels[level],
                offset,
                sampling_kwargs,
                level,
                total_token_to_sample,
                hop_length,
                max_batch_size,
            )

            if save_results:
                self.vqvae.to(music_tokens[level].device)
                # Decode sample
                with torch.no_grad():
                    start_level = len(self.priors) - level - 1  # vqvae levels are reversed
                    raw_audio = self.vqvae.decode(
                        music_tokens[: level + 1], start_level=start_level, bs_chunks=music_tokens[level].shape[0]
                    )
                logdir = f"jukebox/level_{level}"
                if not os.path.exists(logdir):
                    os.makedirs(logdir)
                save_temp_audio(logdir, level, metas=metas, aud=raw_audio.float())
                if compute_alignments and self.priors[0] is not None and self.priors[0].nb_relevant_lyric_tokens > 0:
                    with torch.no_grad():
                        alignments = get_alignment(music_tokens, labels[0], self.priors[0], self.config)
                    torch.save({"alignments": alignments}, f"{logdir}/lyric_alignments.pt")

        return music_tokens

    @add_start_docstrings(
        """
        Generates music tokens based on the provided `labels. Will start at the desired prior level and automatically
        upsample the sequence. If you want to create the audio, you should call `model.decode(tokens)`, which will use
        the VQ-VAE decoder to convert the music tokens to raw audio.

        Args:
            labels (`list[torch.LongTensor]`) :
                List of length `n_sample`, and shape `(self.levels, 4 + self.config.max_nb_genre +
                lyric_sequence_length)` metadata such as `artist_id`, `genre_id` and the full list of lyric tokens
                which are used to condition the generation.
            n_samples (`int`, *optional*, default to 1) :
                Number of samples to be generated in parallel.
        """,
    )
    def ancestral_sample(self, labels, n_samples=1, **sampling_kwargs) -> list[torch.LongTensor]:
        """
        Example:

        ```python
        >>> from transformers import AutoTokenizer, JukeboxModel, set_seed

        >>> model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/jukebox-1b-lyrics")

        >>> lyrics = "Hey, are you awake? Can you talk to me?"
        >>> artist = "Zac Brown Band"
        >>> genre = "Country"
        >>> metas = tokenizer(artist=artist, genres=genre, lyrics=lyrics)
        >>> set_seed(0)
        >>> music_tokens = model.ancestral_sample(metas.input_ids, sample_length=400)

        >>> with torch.no_grad():
        ...     model.decode(music_tokens)[:, :10].squeeze(-1)
        tensor([[-0.0219, -0.0679, -0.1050, -0.1203, -0.1271, -0.0936, -0.0396, -0.0405,
            -0.0818, -0.0697]])
        ```
        """

        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors))))
        music_tokens = [
            torch.zeros(n_samples, 0, dtype=torch.long, device=labels[0].device) for _ in range(len(self.priors))
        ]
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        return music_tokens

    @add_start_docstrings(
        """Generates a continuation of the previously generated tokens.

        Args:
            music_tokens (`list[torch.LongTensor]` of length `self.levels` ) :
                A sequence of music tokens which will be used as context to continue the sampling process. Should have
                `self.levels` tensors, each corresponding to the generation at a certain level.
        """,
        JUKEBOX_SAMPLING_INPUT_DOCSTRING,
    )
    def continue_sample(self, music_tokens, labels, **sampling_kwargs) -> list[torch.LongTensor]:
        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors))))
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        return music_tokens

    @add_start_docstrings(
        """Upsamples a sequence of music tokens using the prior at level `level`.

        Args:
            music_tokens (`list[torch.LongTensor]` of length `self.levels` ) :
                A sequence of music tokens which will be used as context to continue the sampling process. Should have
                `self.levels` tensors, each corresponding to the generation at a certain level.
        """,
        JUKEBOX_SAMPLING_INPUT_DOCSTRING,
    )
    def upsample(self, music_tokens, labels, **sampling_kwargs) -> list[torch.LongTensor]:
        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors) - 1)))
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        return music_tokens

    @add_start_docstrings(
        """Generate a raw audio conditioned on the provided `raw_audio` which is used as conditioning at each of the
        generation levels. The audio is encoded to music tokens using the 3 levels of the VQ-VAE. These tokens are
        used: as conditioning for each level, which means that no ancestral sampling is required.

        Args:
            raw_audio (`list[torch.Tensor]` of length `n_samples` ) :
                A list of raw audio that will be used as conditioning information for each samples that will be
                generated.
        """,
        JUKEBOX_SAMPLING_INPUT_DOCSTRING,
    )
    def primed_sample(self, raw_audio, labels, **sampling_kwargs) -> list[torch.LongTensor]:
        sample_levels = sampling_kwargs.pop("sample_levels", list(range(len(self.priors))))
        self.vqvae.to(raw_audio.device).float()
        with torch.no_grad():
            music_tokens = self.vqvae.encode(
                raw_audio, start_level=0, end_level=len(self.priors), bs_chunks=raw_audio.shape[0]
            )
        music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
        return music_tokens


__all__ = ["JukeboxModel", "JukeboxPreTrainedModel", "JukeboxVQVAE", "JukeboxPrior"]
