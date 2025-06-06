# Copyright (c) Meta Platforms, Inc. and affiliates.

from enum import Enum, auto
from typing import Any, List, Optional, Tuple, Union

import torch
from huggingface_hub import PyTorchModelHubMixin
from pydantic import model_validator
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask, flex_attention
from typing_extensions import Self
import json
import logging

import torch
import torch.nn
import torch.nn as nn
from pydantic import ConfigDict
from torch.nn import functional as F

import abc

import os
import time
from collections import defaultdict

from pydantic import BaseModel

SEP = " "
BOS_ID: int = 1
EOS_ID: int = 2
PAD_ID: int = -1
BOE_ID: int = 0
BPE_ID: int = 3
OFFSET: int = 4

BYTE_UNITS: int = 256

RMSNorm = nn.RMSNorm

logger = logging.getLogger()

from .blt_args import (
    BaseTransformerArgs,
    ByteLatentTransformerArgs,
    GlobalTransformerArgs,
    LocalDecoderArgs,
    LocalModelArgs,
    LMTransformerArgs,

)

flex_attention_comp = flex_attention


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def create_causal_mask(
    seqlen,
    attn_impl: str,
    attn_bias_type: str | None,
    *,
    eos_id: int | None = None,
    tokens: torch.Tensor | None = None,
    sliding_window: int | None = None,
):
    if attn_impl == "sdpa":
        BLT_SUPPRESS_ATTN_ERROR = int(os.environ.get("BLT_SUPPRESS_ATTN_ERROR", 0))

        if attn_bias_type == "causal":
            return "causal"

        if BLT_SUPPRESS_ATTN_ERROR == 1:
            return "causal"
        else:
            raise ValueError(
                "SDPA attention being used, which doesn't have specialized attention implementations for block_causal and local_block_causal attention. To suppress this error and run the model anyway, set the environment variable BLT_SUPPRESS_ATTN_ERROR=1"
            )
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


class InitStdFactor(str, Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096

def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    rope_use_fp32_in_outer_product: bool = False,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    if rope_use_fp32_in_outer_product:
        t = t.to(torch.float32)

    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seqlen: int = 1024,
        rope_use_fp32_in_outer_product: bool = False,
    ):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                dim=head_dim,
                end=max_seqlen,
                theta=theta,
                rope_use_fp32_in_outer_product=self.rope_use_fp32_in_outer_product,
            ),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim,
            end=self.max_seqlen,
            theta=self.theta,
            rope_use_fp32_in_outer_product=self.rope_use_fp32_in_outer_product,
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


def _reshape_for_attn_bias(
    attn_bias: None,
    *tensors: torch.Tensor,
) -> list[torch.Tensor]:
    to_transform = list(tensors)
    if isinstance(attn_bias):
        # could be `view` instead of reshape during training, but for inference
        # have to reshape due to strides mismatch
        to_transform = [t.reshape(1, -1, *t.shape[2:]) for t in to_transform]
    return to_transform


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )
        
        output_reshaped = output.reshape(output_shape)

        output = self.wo(output_reshaped)

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5)) / factor

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5)) / factor
        out_init_std = init_std or (self.hidden_dim ** (-0.5)) / factor

        nn.init.trunc_normal_(
            self.w1.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )
        nn.init.trunc_normal_(
            self.w3.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        norm_x = self.attention_norm(x)
        attn_out = self.attention(
            norm_x,
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        h = x + attn_out
        h_norm = self.ffn_norm(h)
        out = h + self.feed_forward(h_norm)
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class SequenceModelWithOutput(abc.ABC):
    @abc.abstractmethod
    def get_output_seq_len(self) -> int:
        pass


class BaseTransformer(nn.Module, SequenceModelWithOutput):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.attn_impl = args.attn_impl
        self.attn_bias_type = args.attn_bias_type
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
            rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
        )
        self.eos_id = args.eos_id

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

    def get_output_seq_len(self):
        return self.max_seqlen

    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, str]] = None,
        attn_impl: str = "sdpa",
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        return h

    def init_weights(self):
        self.rope_embeddings.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)


class LMTransformer(
    BaseTransformer,
    PyTorchModelHubMixin,
    repo_url="https://github.com/facebookresearch/blt",
    # paper_url="https://arxiv.org/abs/2412.09871",
    pipeline_tag="text-generation",
    license="other",
    license_name="fair-noncommercial-research-license",
    license_link="https://huggingface.co/facebook/blt/blob/main/LICENSE",
    coders={
        LMTransformerArgs: (
            lambda x: {"args": x.model_dump()},
            lambda data: LMTransformerArgs(**data),
        )
    },
):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

        if args.weight_tying:
            self.output.weight = self.embeddings.tok_embeddings.weight

    def push_to_hub(self, *args, **kwargs):
        raise ValueError(
            "For meta authors: Do not push BLT weights with this, save weights with save_pretrained() then push them manually to HF hub to ensure the repository metadata is correct."
        )

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        attn_impl: str | None = None,
    ):
        if attn_impl is None:
            attn_impl = self.attn_impl
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)
     #   attn_impl = "sdpa"
        mask = (
            mask
            if mask is not None
            else create_causal_mask(
                seqlen,
                attn_impl,
                self.attn_bias_type,
                sliding_window=self.sliding_window,
                tokens=token_values,
                eos_id=self.eos_id,
            )
        )
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits

    def reset_parameters(self, init_std=None):
        self.norm.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        init_std = self.dim ** (-0.5)
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        super().init_weights()

        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


class PatchingModeEnum(str, Enum):
    entropy = "entropy"
    bpe = "bpe"
    bpe_patcher = "bpe_patcher"
    space = "space"
    static = "static"
    byte = "byte"


class PatcherArgs(BaseModel):
    patching_mode: PatchingModeEnum = PatchingModeEnum.entropy
    patching_device: str = "cpu"
    entropy_model_checkpoint_dir: str | None = None
    realtime_patching: bool = False
    threshold: float = 1.335442066192627
    threshold_add: float | None = None
    max_patch_length: int | None = None
    patch_size: float = 4.5
    patching_batch_size: int = 1
    device: str = "cpu"
    monotonicity: bool = False
    log_time: bool = False

    def build(self) -> "Patcher":
        return Patcher(self)

def rightpad(seq, pad_id, max_len):
    return seq + [pad_id] * (max_len - len(seq))


def check_non_zero_after_zero(tensor):
    zero_mask = tensor == 0
    shifted_mask = torch.cat(
        [
            torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
            zero_mask[:, :-1],
        ],
        dim=1,
    )
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()

def entropy(scores):
    """
    scores: [bs, seq_len, vocab]
    returns [bs, seq_len]

    Computes the entropy for each token in the batch.
    Note: uses natural log.
    """
    log_probs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy


def calculate_entropies(
    tokens: torch.tensor,
    entropy_model,
    patching_batch_size,
    device: str | None = None,
    enable_grad: bool = False,
):
    """
    tokens: 2D tensor of shape [batch_size, seq_len]
    Return 2D tensor of shape [batch_size, seq_len] with entropies for each token.

    Splits the tokens into chunks of size max_length and calculates entropies for each chunk.
    Entropy model can be executed on cpu or gpu, specify either 'cuda' or 'cpu' in the device argument.
    """

    grad_context = nullcontext() if enable_grad else torch.no_grad()

    with grad_context:
        entropies = []
        preds = []
        max_length = getattr(entropy_model, "max_length", 8192)
        batch_numel = max_length * patching_batch_size
        splits = torch.split(tokens.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(
                pad_size, dtype=split.dtype, device=split.device, requires_grad=False
            )
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length)
            if device is not None:
                split = split.to(device)
            # assert torch.all(split >= 0) and torch.all(split < 260)
            pred = entropy_model(split)
            pred = pred.reshape(-1, pred.shape[-1])[
                : split.numel() - pad_size, :
            ]  # [batch_size * seq_len, vocab]
            preds.append(pred)
            pred_entropies = entropy(pred)
            entropies.append(pred_entropies)

        concat_entropies = torch.cat(entropies, dim=0)
        concat_entropies = concat_entropies.reshape(tokens.shape)
        concat_preds = torch.cat(preds, dim=0)
        concat_preds = concat_preds.reshape(tokens.shape[0], -1)
    return concat_entropies, concat_preds


def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
    else:
        patch_ids = (
            torch.arange(trunc_seq_len, device=patch_start_mask.device)
            .unsqueeze(0)
            .repeat(bs, 1)
        )
        extra_patch_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat(
            (patch_start_mask, ~patch_start_mask), dim=1
        )
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(
            bs, trunc_seq_len
        )[:, :max_patches]
    return patch_start_ids


def patch_lengths_from_start_ids(patch_start_ids, seq_len):
    """
    Calculate patch lengths from start ids.
    start ids: ex: [0, 1, 7, 7, 7, 7, 7], it has the start ids of the patches (here 0, 1), and then
        the rest are filled to the seq len.
    seq_len: ex: 7 length of the sequence

    returns the patch lengths:
    [1, 6] for the above example.
    """
    last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
    patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
    patch_lengths = patch_end_ids - patch_start_ids + 1
    assert torch.all(patch_lengths >= 0), f"{patch_lengths}"
    assert not check_non_zero_after_zero(patch_lengths), f"{patch_lengths}"
    return patch_lengths

def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    threshold=None,
    threshold_add=None,
    monotonicity=False,
    include_next_token=True,
):
    """
    Use entropies to find the start ids of each patch.
    Use patch_size or threshold to figure out the total number of patches to allocate.

    When threshold is not None the number of patches is not constant between
    different sequences, but patches can be identified incrementally rather than
    decided globally using the entire sequence.
    """
    bs, seq_len = entropies.shape[:2]

    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=entropies.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[
        1
    ]  # remove the first preds because they will be start of patches.
    entropies = entropies[:, 1:]
    if threshold is None:
        num_patches = seq_len // patch_size
        patch_start_ids = entropies.topk(num_patches - 2, dim=1).indices
        patch_start_ids = patch_start_ids.sort(dim=1).values
    else:
        patch_start_mask = entropies > threshold
        if not include_next_token:
            patch_start_mask = patch_start_mask[:, :-1]
        # patch_start_mask[1:] |= tokens[:-1] < OFFSET
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)

    patch_start_ids = torch.cat(
        (first_ids, patch_start_ids + preds_truncation_len), dim=1
    )
    return patch_start_ids

def split_large_numbers(lst, m):
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    assert sum(new_lst) == sum(lst), f"{sum(new_lst)} != {sum(lst)}"
    return new_lst

class Patcher:
    def __init__(self, patcher_args: PatcherArgs):
        self.patcher_args = patcher_args
        self.patching_mode = patcher_args.patching_mode
        self.realtime_patching = patcher_args.realtime_patching
        self.realtime_patching = True
        if self.realtime_patching:
            assert (
                patcher_args.entropy_model_checkpoint_dir is not None
            ), "Cannot require realtime patching without an entropy model checkpoint"
            maybe_consolidated = os.path.join(
                patcher_args.entropy_model_checkpoint_dir,
                "consolidated/consolidated.pth",
            )
            if os.path.exists(maybe_consolidated):
                state_path = maybe_consolidated
            else:
                state_path = os.path.join(
                    patcher_args.entropy_model_checkpoint_dir, "consolidated.pth"
                )
            entropy_model, _ = load_entropy_model(
                patcher_args.entropy_model_checkpoint_dir,
                state_path,
            )
            entropy_model = entropy_model.to(patcher_args.patching_device)
            self.entropy_model = entropy_model
        else:
            self.entropy_model = None
        self.threshold = patcher_args.threshold
        self.threshold_add = patcher_args.threshold_add
        self.max_patch_length = patcher_args.max_patch_length
        self.patch_size = patcher_args.patch_size
        self.patching_batch_size = patcher_args.patching_batch_size
        self.device = patcher_args.device
        self.monotonicity = patcher_args.monotonicity
        self.log_time = patcher_args.log_time
        if self.log_time:
            self.log = defaultdict(float)

    def patch(
        self,
        tokens: torch.Tensor,
        include_next_token: bool = False,
        preds: torch.Tensor | None = None,
        entropies: torch.Tensor | None = None,
        threshold: float = None,
    ) -> torch.Tensor:
        """
        tokens: 2D tensor of shape [batch_size, seq_len] that needs to be patched
        Returns patch lengths and optionally scores associated with the tokens (i.e. entropies, logprobs etc.)
        -> output tensor: [batch_size, max_num_patches]
            each tensor is processed independently and gets right padded with zeros.

        Patching with the following modes:
        1. patching_mode = None: static patch size
        2. patching_mode = "entropy":
            calculate entropy of each token, allocate patches so that the total
            number of patches is the same as static patching but choose to begin
            patches on tokens where the model is most uncertain (highest entropy).

            When threshold is provided, it uses the threshold to decide when to
            start a new patch.
        3. patching_mode = "space":
            use space like tokens to define the patches.
        4. patching_mode = "bpe":
            use bpe delim tokens to define the patches.

        To correctly patch the last token, it may be necessary to include the next token in the patch
        lengths calculations. This is controlled by the include_next_token argument.
        """
        bs, seq_len = tokens.shape
        seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
        scores = None
        # STATIC
        if self.patching_mode == PatchingModeEnum.byte:
            patch_lengths = torch.ones(
                (bs, seq_len_next_tok), dtype=tokens.dtype, device=tokens.device
            )
        elif self.patching_mode == PatchingModeEnum.entropy:
            if self.log_time:
                s = time.time()
            if entropies is not None:
                scores = entropies.to(dtype=torch.float32)
            elif preds is not None:
                scores = entropy(preds)
            else:
                start_entropies = time.time()
                scores, _ = calculate_entropies(
                    tokens,
                    self.entropy_model,
                    self.patching_batch_size,
                    self.device,
                )
            if self.log_time:
                self.log["calculate_entropies"] += time.time() - s
                s = time.time()
            patch_start_ids = find_entropy_patch_start_ids(
                scores,
                self.patch_size,
                include_next_token=include_next_token,
                threshold=threshold if threshold is not None else self.threshold,
                threshold_add=self.threshold_add,
                monotonicity=self.monotonicity,
            )
            if self.log_time:
                self.log["find_entropy_patch_start_ids"] += time.time() - s
                s = time.time()
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
            if self.log_time:
                self.log["patch_lengths_from_start_ids"] += time.time() - s
                s = time.time()
        else:
            raise NotImplementedError(f"self.patching_mode {self.patching_mode}")

        # Apply any processing to patch lengths
        if self.max_patch_length is not None:
            # TODO: avoid going back to a list here.
            patch_lengths = [
                split_large_numbers(pl, self.max_patch_length)
                for pl in patch_lengths.tolist()
            ]
            max_len = max([len(pl) for pl in patch_lengths])
            patch_lengths = [rightpad(pl, 0, max_len=max_len) for pl in patch_lengths]
            patch_lengths = torch.tensor(
                patch_lengths, dtype=tokens.dtype, device=tokens.device
            )
        assert not check_non_zero_after_zero(patch_lengths)
        # Find the last non-zero column index using argmax on a reversed version of the tensor
        last_non_zero_col_reversed = (
            (patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min()
        )
        # Slice the tensor up to the last non-zero column
        patch_lengths = patch_lengths[
            :, : patch_lengths.shape[1] - last_non_zero_col_reversed
        ]
        assert (
            torch.sum(patch_lengths)
            == tokens.numel() + include_next_token * tokens.shape[0]
        ), f"{torch.sum(patch_lengths)} != {tokens.numel() + include_next_token * tokens.shape[0]}"
        if self.log_time:
            self.log["postprocessing_patch_lengths"] += time.time() - s
            self.log["tokens"] += patch_lengths.sum().item()
        return patch_lengths, scores



def load_entropy_model(entropy_model_checkpoint_dir, state_dict_path, device="cpu"):
    with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
        reloaded = json.loads(fr.read())

    torch.set_default_dtype(torch.bfloat16)
    model_params = reloaded["entropy_model"]
    logger.warning(
        "Update checkpoint to load attn and sliding window args from checkpoint"
    )
    entropy_model_args = LMTransformerArgs(
        dim=model_params["dim"],
        n_layers=model_params["n_layers"],
        n_heads=model_params["n_heads"],
        max_seqlen=model_params["max_seqlen"],
        ffn_dim_multiplier=model_params["ffn_dim_multiplier"],
        vocab_size=model_params["vocab_size"],
        attn_bias_type="local_block_causal",
        attn_impl="sdpa", #originally xformers
        sliding_window=512,
    )
    entropy_model = LMTransformer(entropy_model_args)

    entropy_model.load_state_dict(
        torch.load(state_dict_path, map_location=device)["model"], strict=False
    )
    entropy_model.to(device)
    entropy_model = entropy_model.eval()
    # no grads for the model:
    for param in entropy_model.parameters():
        param.requires_grad = False
    return entropy_model, entropy_model_args


def get_encoder_dim_token_emb(args):
    if args.dim_token is not None:
        dim_token_emb = args.dim_token
    elif args.use_local_encoder_transformer:
        dim_token_emb = args.dim_local_encoder
    else:
        dim_token_emb = args.dim_global // args.patch_size
    return dim_token_emb


def get_encoder_dim_patch_emb(args):
    dim_patch_emb = None
    if args.cross_attn_encoder:
        if args.cross_attn_init_by_pooling:
            dim_patch_emb = args.dim_local_encoder
        else:
            dim_patch_emb = args.dim_global
    return dim_patch_emb


def get_global_dim_patch_emb(args):
    dim_token_emb = get_encoder_dim_token_emb(args)
    if args.cross_attn_encoder:
        dim_patch_emb = dim_token_emb * args.cross_attn_k
    elif (
        args.downsampling_by_pooling is None
        or not args.downsampling_by_pooling
        or len(args.downsampling_by_pooling) == 0
    ):
        dim_patch_emb = dim_token_emb * args.patch_size
    else:
        dim_patch_emb = dim_token_emb * sum(
            [
                pooling in args.downsampling_by_pooling
                for pooling in ["avg", "min", "max"]
            ]
        )
    return dim_patch_emb


def get_decoder_dim_token_emb(args):
    if args.share_encoder_decoder_emb:
        dim_token_emb = get_encoder_dim_token_emb(args)
    elif args.dim_token is not None:
        dim_token_emb = args.dim_token
    else:
        dim_token_emb = args.dim_local_decoder
    return dim_token_emb


def parse_ngram_to_size(ngram_to_size_str: str | None) -> dict[int, int]:
    if ngram_to_size_str is None:
        return None
    ngram_to_size = {}
    for entry in ngram_to_size_str.split(","):
        ngram, size = entry.split(":")
        ngram = int(ngram)
        size = int(size)
        ngram_to_size[ngram] = size
    return ngram_to_size


def fill_tokens(tokens, patch_size, fill_id):
    batch_size, seq_len = tokens.shape
    if seq_len % patch_size == 0:
        return tokens
    else:
        remaining = patch_size - seq_len % patch_size
        final_padding = tokens.new(batch_size, remaining).fill_(fill_id)
        return torch.cat((tokens, final_padding), dim=1)


def decoder_patch_ids_from_lengths(patch_lengths, nb_boe, seq_len):
    first_patch_length = patch_lengths[0, 0]
    assert torch.all(
        first_patch_length == patch_lengths[:, 0]
    ), "first patch should always be the same size (1 for dynamic, patch_size for static)."
    assert (
        first_patch_length - nb_boe == 1
    ), f"First patch (patch length: {first_patch_length}) should have one non-boe token (boe toks: {nb_boe})"
    # Remove first patch from patch_ids for local decoder inputs and shift the last patch.
    # decoder_patch_lengths = patch_lengths[:, 1:].clone()
    # decoder_patch_lengths = add_to_last_nonzero_patch(decoder_patch_lengths, 1)
    decoder_patch_lengths = patch_lengths[:, 1:]
    assert (
        decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]
        == patch_lengths.sum()
    ), f"{decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]} != {patch_lengths.sum()}"
    assert torch.all(decoder_patch_lengths >= 0), f"{decoder_patch_lengths}"
    decoder_patch_ids = patch_ids_from_lengths(
        patch_lengths=decoder_patch_lengths, seq_len=seq_len
    )
    return decoder_patch_ids


def rolling_polynomial_hash(t, hash_func_nb: int = 0):
    primes = [
        1000000007,
        5915587277,
        1500450271,
        3267000013,
        5754853343,
        4093082899,
        9576890767,
        3628273133,
        2860486313,
        5463458053,
        3367900313,
    ]
    prime = torch.tensor(primes[hash_func_nb], dtype=torch.int64, device=t.device)
    prime_powers = torch.stack([prime**i for i in range(t.shape[-1])])
    return torch.sum(t * prime_powers, dim=-1)

def byte_group_hash_function(
    x: torch.Tensor, group_size: int = 2, hash_func_nb: int = 0, max_hash: int = 30000
):
    """
    Returns a hash of the input x and maps it to a value in the range [0, max_hash].

    expects: x of shape (batch_size, seq_len) with values as ids in the token vocab.
    returns a tensor  of shape (batch_size, seq_len) with values in the range [0, max_hash].

    Note: max hash can make a big difference on the number of collisions.
    """
    with torch.no_grad():
        bs, seq_len = x.shape
        # x_numpy = x.numpy()
        # hash_values = torch.zeros(bs, seq_len, dtype=torch.int64, requires_grad=False)
        # for i in range(bs):
        #     for j in range(seq_len):
        #         start = max(j, j-group_size+1)
        #         end = j+1
        #         hash_values[i, j] = hash_array(x_numpy[i, start:end], max_hash)

        prefix = torch.zeros(bs, group_size - 1, dtype=torch.int64, device=x.device)
        x = torch.cat([prefix, x], dim=1)
        windows = x.unfold(1, group_size, 1)
        # hashes = get_rolling_polynomial_hash_fn(hash_func_nb, group_size)(windows)
        hashes = rolling_polynomial_hash(windows, hash_func_nb)
        hash_values_range = hashes % max_hash
    hash_values_range.requires_grad = False
    return hash_values_range


def create_patch_mask_from_ids(
    patch_ids, num_patches, window=None, patches_as_queries=False
):
    """
    Creates a tensor of shape [bs, seq_len, num_patches] where each element at position (i, j, k)
    is True if the patch id at position (i, j) is less than or equal to k.
    Args:
        patch_ids (torch.Tensor): Tensor of shape [bs, seq_len] containing patch ids.
        num_patches (int): Total number of patches.
        window (int): If not None, only considers patches within a window of size window.
        patches_as_queries (bool): If True, the patches are used as queries
    Returns:
        torch.Tensor: Tensor of shape [bs, q_len, kv_len] with the desired mask.
    """
    bs, seq_len = patch_ids.shape
    if not patches_as_queries:
        q_ids = patch_ids.unsqueeze(-1).expand(bs, seq_len, num_patches)
        kv_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, seq_len, num_patches)
        )
    else:
        kv_ids = patch_ids.unsqueeze(1).expand(bs, num_patches, seq_len)
        q_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, num_patches, seq_len)
        )
    if window is None:
        mask = q_ids == kv_ids
    else:
        mask = (kv_ids <= q_ids) & (q_ids < kv_ids + window)
    return mask


def cross_attn_mask(
    patch_ids,
    patch_lengths,
    N,
    patches_as_queries=False,
    cross_attn_k=1,
    window=None,
    block_mask=True,
):
    bs = patch_ids.shape[0]
    with torch.no_grad():
        # Create the patch mask
        cross_mask = create_patch_mask_from_ids(
            patch_ids,
            patch_lengths.shape[1],
            window=window,
            patches_as_queries=patches_as_queries,
        ).repeat_interleave(cross_attn_k, dim=1 if patches_as_queries else -1)
        q_len = patch_lengths.shape[1] * cross_attn_k if patches_as_queries else N
        kv_len = N if patches_as_queries else patch_lengths.shape[1] * cross_attn_k
        assert cross_mask.shape == (
            bs,
            q_len,
            kv_len,
        ), f"{cross_mask.shape} != {(bs, q_len, kv_len)}"
        block_mask = None
        if block_mask:

            def patch_mask(b, h, q_idx, kv_idx):
                return cross_mask[b, q_idx, kv_idx]

            block_mask = create_block_mask(
                patch_mask,
                B=bs,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                _compile=True,
            )
            return block_mask
        else:
            return torch.where(
                cross_mask, torch.tensor(0.0), torch.tensor(float("-inf"))
            ).unsqueeze(
                1
            )  # [bs, 1, q_len, kv_len]


def get_blt_input(
    tokens: torch.Tensor,
    enforce_patch_size_multiple: bool,
    nb_boe: torch.Tensor,
    patch_size: int,
    boe_id: int,
):
    """
        This function returns X_et, X_gt and X_dt, the encoder, global, and decoder
    tokens respectively.

    Consider the input and target sequences:
    X=[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12,13]
    Y=[4,5,6,7,eos,bos,8,9,10,eos,bos,11,12,13,14]
    with patch_size=4

    Note 1: that there will be no special tokens introduced at the patch level.
    Note 2: X_e needs to be trimmed to be passed to Global

    Current without boe:
    X_et = [[boe,boe,boe,boe] [3,4,5,6],      [7,eos,bos,8],    [9,10,eos,bos] [11,12,13, pad]]
    X_g =  [[boe,boe,boe,boe] [3,4,5,6],      [7,eos,bos,8],    [9,10,eos,bos] [11,12,13, pad]] # remove last glob patch
    X_dt = [[3,4,5,6]         [7,eos,bos,8],  [9,10,eos,bos],   [11,12,13]]
    Y =    [[4,5,6,7]         [eos,bos,8,9],  [10,eos,bos,11],  [12,13,14]]

    --> lag fix:
    X_et = [[boe,boe,boe,3]   [4,5,6,7],      [eos,bos,8,9],    [10,eos,bos,11] [12,13,pad,pad]]
    X_g =  [[boe,boe,boe,3]   [4,5,6,7],      [eos,bos,8,9],    [10,eos,bos,11]]
    X_dt = [[3,4,5,6]         [7,eos,bos,8],  [9,10,eos,bos],   [11,12,13]]
    Y =    [[4,5,6,7]    	  [eos,bos,8,9],  [10,eos,bos,11],  [12,13,14]]

    Dynamic (current):
    X = [3,4,5,6,7,eos,bos,8,9,10,eos,bos]
    Y = [4,5,6,7,eos,bos,8,9,10,eos,bos,11]

    entropy patching:
    input: 7, bos, 9, 10
    pred (high entropy): eos, 8, 10, eos

    X_et = [[boe,3,4,5,6,7,eos,bos,8,9,10,eos,bos]
    X_g =  [[boe],      [3,4,5,6], [7,eos],[bos,8],[9],     [10,eos]]
    X_dt = [[3,4,5,6],  [7,eos],   [bos,8],[9],    [10,eos],[bos]]
    Y =    [4,5,6,7,eos,bos,8,9,10,eos,bos,11]

    --> lag fix no boe (force single byte first patch):
    X_et = [[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12]
    X_g =  [[3],        [4,5,6,7], [eos,bos],[8,9], [10],       [eos,bos],      [11,12]] # remove last global patch
    X_dt = [[3,4,5,6],  [7,eos],   [bos,8], [9],    [10,eos],   [bos,11,12]]
    Y =    [4,5,6,7,    eos,bos,    8,9,    10,     eos,bos,    11,12,13]

    input: 4, 7, bos, 9, 10
    pred (high entropy): 5, eos, 8, 10, eos

    X_et = [[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12]
    X_g =  [[3],        [4]   ,   [5,6,7], [eos,bos],[8,9], [10],       [eos,bos],      [11,12]] # remove last global patch
    X_dt = [[3]         [4,5,6],  [7,eos],   [bos,8], [9],    [10,eos],   [bos,11,12]]
    Y =    [4,]         [5,6,7,    eos,bos,    8,9,    10,     eos,bos,    11,12,13]

    Handle the last byte properly.
    patch_lengths = [1, 1,         3,      2,         2      1           2               2         1]
    X_et = [[3,4,5,6,7,eos,bos,8,9,10,eos,bos,11,12]
    X_g =  [[3],        [4]   ,   [5,6,7], [eos,bos],[8,9], [10],       [eos,bos],      [11,12]] # do not remove last global patch
    X_dt = [[3]         [4,5,6],  [7,eos],   [bos,8], [9],    [10,eos],   [bos,11]       [12]]
    Y =    [4,]         [5,6,7,    eos,bos,    8,9,    10,     eos,bos,    11,12,        13]]


    bpe delim
    X_et = [[3,4,5,6,7,<d>,eos,bos,<d>,8,9,<d>,10,<d>,eos,bos,11,12]
    X_g =  [[3],          [4,5,6,7,<d>],     [eos,bos,<d>], ..
    X_dt = [[3,4,5,6,7],  [<d>,eos,bos],     [<d>,bos,8], ..
    Y =    [4,5,6,7,<d>,    eos,bos,<d>       8,9,<d>, ..


    Note 1: that there will be no special tokens introduced at the patch level.
    Note 2: X_e needs to be trimmed to be passed to Global
    """
    batch_size, seq_len = tokens.shape
    local_encoder_tokens = tokens
    local_decoder_tokens = tokens

    if nb_boe > 0:
        padded_patch = tokens.new(batch_size, nb_boe).fill_(boe_id)
        local_encoder_tokens = torch.cat((padded_patch, local_encoder_tokens), dim=1)
    # global_tokens = tokens.new(batch_size, ((seq_len-1) // patch_size)+1).fill_(boe_id)

    # create global tokens, contains boe tokens and eos
    # padded_local_encoder_tokens = fill_tokens(local_encoder_tokens, patch_size, boe_id)
    # patches = padded_local_encoder_tokens.view(batch_size, -1, patch_size)
    # global_tokens = (patches.eq(eos_id).any(dim=2).int() * eos_id)[:, 1:]
    # global_tokens += global_tokens.eq(0).int() * boe_id
    # TODO: fix this when we want to use block causal in the global.

    if enforce_patch_size_multiple and local_encoder_tokens.shape[-1] % patch_size != 0:
        local_encoder_tokens = fill_tokens(local_encoder_tokens, patch_size, boe_id)

    return local_encoder_tokens, None, local_decoder_tokens


def patch_ids_from_lengths(patch_lengths, seq_len):
    bs, num_patches = patch_lengths.shape
    # Create a tensor of cumulative sums of the patch lengths
    cum_d = torch.cat(
        [
            torch.zeros(bs, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
            patch_lengths.cumsum(dim=-1),
        ],
        dim=-1,
    )
    patch_ids = (cum_d.unsqueeze(-1) <= torch.arange(seq_len, device=cum_d.device)).sum(
        dim=-2
    ) - 1
    assert not (
        torch.max(patch_ids) > patch_lengths.shape[-1] or torch.min(patch_ids) < 0
    ), f"{torch.max(patch_ids)} > {patch_lengths.shape[-1]} or {torch.min(patch_ids)} < 0"
    return patch_ids


class LocalModelBase(nn.Module):
    def __init__(self, args: LocalModelArgs):
        super().__init__()

        self.dim = args.dim
        self.dropout = args.dropout
        self.vocab_size = args.vocab_size
        self.patch_size = args.patch_size
        self.dim_patch_emb = args.dim_patch_emb

        self.attn_impl = args.attn_impl
        self.sliding_window = args.sliding_window
        self.use_rope = args.use_rope
        self.init_std_factor = args.init_std_factor
        self.cross_attn_encoder = getattr(args, "cross_attn_encoder", None)
        self.cross_attn_decoder = getattr(args, "cross_attn_decoder", None)
        self.cross_attn_k = getattr(args, "cross_attn_k", None)
        self.eos_id = args.eos_id

        self.boe_id = BOE_ID

        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layers)]
        )

        if not self.use_rope:
            self.pos_embeddings = nn.Embedding(args.max_length, args.dim)
        else:
            self.rope = RotaryEmbedding(
                theta=args.rope_theta,
                head_dim=args.head_dim or args.dim // args.n_heads,
                max_seqlen=args.max_seqlen,
                rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
            )
            self.pos_embeddings = None

        self.token_embedding_projection = (
            nn.Linear(args.dim_token_emb, args.dim, bias=False)
            if hasattr(args, "dim_token_emb") and args.dim_token_emb != self.dim
            else None
        )

        self.patch_embedding_projection = self._create_patch_projection(args)

    def _should_create_patch_projection(self, args: LocalModelArgs):
        dimension_mismatch = (
            getattr(args, "dim_patch_emb") and args.dim_patch_emb != self.dim
        )

        # Check cross attention conditions
        cross_attn_conditions = (
            args.cross_attn_encoder and args.cross_attn_init_by_pooling
        ) or (args.cross_attn_decoder and args.cross_attn_init_by_pooling)

        return dimension_mismatch or cross_attn_conditions

    def _create_patch_projection(self, args):
        if not self._should_create_patch_projection(args):
            return None

        output_dim = args.dim_token_emb * (self.cross_attn_k or 1)

        return nn.Linear(
            in_features=args.dim_patch_emb,
            out_features=output_dim,
            bias=False,
        )

    def apply_embedding(self, tokens, embeds):
        if embeds is not None:
            return embeds
        else:
            return self.tok_embeddings(tokens)

    def init_weights(self, init_std=None):
        self.rope.reset_parameters()
        if hasattr(self, "norm"):
            self.norm.reset_parameters()

        init_std = init_std or (self.dim ** (-0.5))
        if hasattr(self, "tok_embeddings"):
            nn.init.trunc_normal_(
                self.tok_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
        if self.pos_embeddings is not None:
            nn.init.trunc_normal_(
                self.pos_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(None, factor)

        if hasattr(self, "output"):
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        if self.token_embedding_projection is not None:
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        if self.patch_embedding_projection is not None:
            patch_emb_std = self.dim_patch_emb ** (-0.5)
            nn.init.trunc_normal_(
                self.patch_embedding_projection.weight,
                mean=0.0,
                std=patch_emb_std,
                a=-3 * patch_emb_std,
                b=3 * patch_emb_std,
            )

        if self.cross_attn_layers is not None:
            for depth, layer in enumerate(self.cross_attn_layers):
                factor = {
                    InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                    InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                    InitStdFactor.DIM_RATIO: self.dim / 4096,
                    InitStdFactor.DISABLED: 1.0,
                }[self.init_std_factor]

                layer.init_weights(None, factor)


class LocalEncoder(LocalModelBase):
    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        self.apply_transformer = args.use_local_encoder_transformer
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.expects_hash_embeddings = args.encoder_hash_byte_group_size is not None
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_all_layers_encoder = args.cross_attn_all_layers_encoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        if self.cross_attn_encoder:
            self.cross_attn_layers = torch.nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_encoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )

    def apply_embedding(self, tokens, embeds):
        if embeds is not None:
            assert (
                self.expects_hash_embeddings
            ), "Not expecting embeddings to be passed."
            return embeds
        else:
            return self.tok_embeddings(tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["BlockMask", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """ """
        bs, seqlen = tokens.shape
        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                "local_block_causal",
                sliding_window=self.sliding_window,
                tokens=tokens,
                eos_id=self.eos_id,
            )

        h = self.apply_embedding(tokens, embeds)
        freqs_cis = self.rope(seqlen=seqlen) if self.use_rope else None

        h = F.dropout(h, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.attn_impl)
            # check if cross attention should be applied to either all layer or only the last layer
            if self.cross_attn_encoder and (
                i == len(self.layers) - 1 or self.cross_attn_all_layers_encoder
            ):
                patch_embeds = self.apply_cross_attention(
                    h, patch_embeds, i, bs, num_patches, patch_ids, cross_mask
                )

        h_residual = patch_embeds if self.cross_attn_encoder else None
        return (h, h_residual), cache

    def apply_cross_attention(
        self, h, patch_embeds, layer_idx, bs, num_patches, patch_ids, cross_mask
    ):
        # apply pooling and project
        if self.cross_attn_init_by_pooling and patch_embeds is None:
            # patch_embeds = downsample(
            #     h,
            #     num_patches,
            #     patch_ids=patch_ids,
            #     downsampling_by_pooling=self.downsampling_by_pooling,
            #     patch_size=self.patch_size,
            # )
            patch_embeds = self.patch_reduce(h, num_patches, "amax", patch_ids)
            if self.patch_embedding_projection is not None:
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        layer_idx = layer_idx if self.cross_attn_all_layers_encoder else 0
        patch_embeds_cross = self.cross_attn_layers[layer_idx](
            x=patch_embeds,
            kv=h,
            mask=cross_mask,
        )
        return patch_embeds + patch_embeds_cross

    def patch_reduce(self, h, max_num_patches, reduction, patch_ids):
        """
        Reduce variable length patches to single embedding per patch
        Note: this works with variable number of patches for different sequences in the batch
        It handles variable length patches by assuming that patch_lengths will be 0 for any
        extra patches on the *right*. Since there can be a variable number of patches
        this function also return the number of patches for each sequence in the batch.
        Any embeddings on the right that are not allocated to a patch
        (i.e. if the sum(patch_lengths[i]) < seq_len for any i)
        will be sent to a dummy patch, which is trimmed before returning.
        """
        bs, seq_len, emb_dim = h.shape

        patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])

        reduced_embs = torch.zeros(
            (bs, max_num_patches, emb_dim), dtype=h.dtype, device=h.device
        )
        reduced_embs = reduced_embs.scatter_reduce(
            src=h,
            dim=1,
            index=patch_ids,
            reduce=reduction,
            include_self=False,
        )
        reduced_embs = reduced_embs[:, :max_num_patches, :]

        return reduced_embs


class LocalDecoder(LocalModelBase):
    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        # Model configuration flags
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_all_layers_decoder = args.cross_attn_all_layers_decoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        if self.cross_attn_decoder:
            self.cross_attn_layers = torch.nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_decoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )

        self.output = nn.Linear(
            self.dim,
            args.vocab_size,
            bias=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor],
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["BlockMask", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        bs, seqlen = tokens.shape
        assert embeds is not None, "Embeddings must be provided"

        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                "local_block_causal",
                sliding_window=self.sliding_window,
                tokens=tokens,
                eos_id=self.eos_id,
            )

        h = embeds

        if self.patch_embedding_projection is not None:
            assert patch_embeds is not None, "Patch embeddings must be passed."
            patch_embeds = self.patch_embedding_projection(patch_embeds)
            if self.cross_attn_k is not None:
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        if patch_embeds is not None and not self.cross_attn_decoder:
            h = h + patch_embeds

        freqs_cis = self.rope(seqlen=seqlen) if self.use_rope else None

        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            if self.cross_attn_decoder and (
                i == 0 or self.cross_attn_all_layers_decoder
            ):
                # Use cross attention to extract info from patch_embeds into h
                h_cross = self.cross_attn_layers[i](
                    x=h,
                    kv=patch_embeds,
                    mask=cross_mask,
                )
                h = h + h_cross

            h = layer(h, mask=mask, freq_cis=freqs_cis, attn_impl=self.attn_impl)

        h_preds = self.norm(h)
        h_preds = F.dropout(h_preds, p=self.dropout, training=self.training)
        h_preds = self.output(h_preds)
        h_preds = h_preds.float()
        return h_preds, cache


class CrossAttention(nn.Module):
    """
    CrossAttention block to attend to the encoder states from the decoder.
    Rope is not supported.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.cross_attn_norm_q = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = RMSNorm(dim, eps=norm_eps)

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[Union[BlockMask, str]] = None,
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, _ = x.shape
        _, slen_kv, _ = kv.shape
        x_norm = self.cross_attn_norm_q(x)
        kv = self.cross_attn_norm_kv(kv)

        xq = self.wq(x_norm)
        xk = self.wk(kv)
        xv = self.wv(kv)

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

       # assert mask is None or isinstance(mask, BlockMask)
        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
        #output = flex_attention_comp(xq, xk, xv, block_mask=mask)
        is_causal = (mask == "causal") if isinstance(mask, str) else False
        mask = mask if isinstance(mask, torch.Tensor) else None
        mask = mask.to(dtype=xq.dtype)
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            is_causal=is_causal,
            attn_mask=mask,
        )
        output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        output = self.wo(output.reshape(output_shape))

        return x + output

    def init_weights(self, base_std: float, factor: float = 1.0):
        std = base_std or (self.dim ** (-0.5)) / factor

        nn.init.trunc_normal_(
            self.wq.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wk.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wv.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        self.cross_attn_norm_q.reset_parameters()
        self.cross_attn_norm_kv.reset_parameters()


class GlobalTransformer(BaseTransformer):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__(args)
        self.dropout = args.dropout
        self.eos_id = args.eos_id
        self.dim_token_emb = args.dim_token_emb

        self.token_embedding_projection = None
        if args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                args.dim_token_emb,
                args.dim,
                bias=False,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """
        Similar to BaseTransformer.forward, but with an additional embeds argument
        and projection to the token space.
        """
        bs, seqlen = tokens.shape

        h = embeds

        mask = (
            mask
            if mask is not None
            else create_causal_mask(
                seqlen,
                self.attn_impl,
                self.attn_bias_type,
                tokens=tokens,
                eos_id=self.eos_id,
            )
        )

        if self.token_embedding_projection is not None and h.shape[-1] != self.dim:
            h = self.token_embedding_projection(h)

        h = F.dropout(h, p=self.dropout, training=self.training)

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=self.attn_impl)
        return h, cache

    def init_weights(self):
        super().init_weights()
        std = self.dim_token_emb ** (-0.5)
        if self.token_embedding_projection is not None:
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )

class EmbeddingType(Enum):
    HASH_TOK = auto()
    NGRAM = auto()


def init_embeddings(
    args,
    embedding_type: EmbeddingType,
    local_encoder_dim: int,
    encoder_hash_byte_group_size: list = None,
):
    if (
        embedding_type == EmbeddingType.HASH_TOK
        and args.encoder_hash_byte_group_size is None
    ):
        return None
    if embedding_type == EmbeddingType.NGRAM and args.encoder_ngram_to_size_str is None:
        return None

    embeddings = []

    if embedding_type == EmbeddingType.HASH_TOK:
        emb_dim = local_encoder_dim
        encoder_hash_byte_group_vocab = args.encoder_hash_byte_group_vocab
        for _ in range(args.encoder_hash_byte_group_nb_functions):
            for _ in encoder_hash_byte_group_size:
                embeddings.append(
                    nn.Embedding(
                        encoder_hash_byte_group_vocab,
                        emb_dim,
                    )
                )

    elif embedding_type == EmbeddingType.NGRAM:
        encoder_ngram_to_size = parse_ngram_to_size(args.encoder_ngram_to_size_str)
        emb_dim = local_encoder_dim
        OFFSET = 4  # This should be passed as parameter if it's variable
        for ngram_vocab_size in encoder_ngram_to_size.values():
            embeddings.append(nn.Embedding(ngram_vocab_size + OFFSET, emb_dim))

    return nn.ModuleList(embeddings)


def compute_hash_embeddings(
    local_encoder_tokens: torch.Tensor,
    local_encoder,
    encoder_hash_tok_embedding: nn.ModuleList,
    encoder_hash_byte_group_nb_functions: int,
    encoder_hash_byte_group_size: list,
    encoder_hash_byte_group_vocab: int,
) -> torch.Tensor:
    """
    Compute embeddings using hash token embeddings.

    Args:
        local_encoder_tokens: Input tokens tensor
        local_encoder: Encoder object with tok_embeddings method
        encoder_hash_tok_embedding: ModuleList of hash token embeddings
        encoder_hash_byte_group_nb_functions: Number of hash functions
        encoder_hash_byte_group_size: List of byte group sizes
        encoder_hash_byte_group_vocab: Vocabulary size for hash embeddings

    Returns:
        torch.Tensor: Combined embeddings
    """
    if encoder_hash_tok_embedding is None:
        return None

    local_encoder_embeds = local_encoder.tok_embeddings(local_encoder_tokens)

    i = 0
    for func_nb in range(encoder_hash_byte_group_nb_functions):
        for byte_group_size in encoder_hash_byte_group_size:
            hash_ids = byte_group_hash_function(
                local_encoder_tokens,
                byte_group_size,
                hash_func_nb=func_nb,
                max_hash=encoder_hash_byte_group_vocab,
            )
            hash_tok_embedding = encoder_hash_tok_embedding[i]
            local_encoder_embeds = local_encoder_embeds + hash_tok_embedding(hash_ids)
            i += 1

    assert i == len(encoder_hash_tok_embedding)
    return local_encoder_embeds


class ByteLatentTransformer(
    nn.Module,
    SequenceModelWithOutput,
    PyTorchModelHubMixin,
    repo_url="https://github.com/facebookresearch/blt",
    # paper_url="https://arxiv.org/abs/2412.09871",
    pipeline_tag="text-generation",
    license="other",
    license_name="fair-noncommercial-research-license",
    license_link="https://huggingface.co/facebook/blt/blob/main/LICENSE",
    coders={
        ByteLatentTransformerArgs: (
            lambda x: {"args": x.model_dump()},
            lambda data: ByteLatentTransformerArgs(**data),
        )
    },
):
    """
    The ByteLatentTransformer (BLT) is a byte-level language model architecture that processes byte sequences
    by dynamically segmenting them into patches. It uses a combination of local encoders, global transformers,
    and local decoders to efficiently encode and decode byte sequences, leveraging patch-based processing for
    improved performance and inference efficiency.
    """

    def __init__(self, args: ByteLatentTransformerArgs):
        super().__init__()

        # General configuration
        self.weight_tying = args.weight_tying
        self.patch_size = args.patch_size
        self.patching_mode = args.patching_mode
        self.boe_id, self.bos_id, self.pad_id, self.eos_id = (
            BOE_ID,
            BOS_ID,
            PAD_ID,
            EOS_ID,
        )
        self.downsampling_by_pooling = args.downsampling_by_pooling
        self.patching_threshold = args.patching_threshold
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen

        # Cross attention configuration
        self.cross_attn_encoder = args.cross_attn_encoder
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_k = args.cross_attn_k
        self.cross_attn_window_encoder = args.cross_attn_window_encoder
        self.cross_attn_window_decoder = args.cross_attn_window_decoder
        self.cross_attn_use_flex_attention = args.cross_attn_use_flex_attention

        # Encoder hash configuration
        self.encoder_hash_byte_group_size = args.encoder_hash_byte_group_size
        self.encoder_hash_byte_group_vocab = args.encoder_hash_byte_group_vocab
        self.encoder_hash_byte_group_nb_functions = (
            args.encoder_hash_byte_group_nb_functions
        )

        # ByteLatent modules
        local_encoder_args = LocalModelArgs(
            # Updated args
            dim=args.dim_local_encoder,
            n_layers=args.n_layers_local_encoder,
            n_heads=args.n_heads_local_encoder,
            dim_token_emb=get_encoder_dim_token_emb(args),
            dim_patch_emb=get_encoder_dim_patch_emb(args),
            cross_attn_encoder=args.cross_attn_encoder,
            cross_attn_decoder=False,
            cross_attn_k=args.cross_attn_k if args.cross_attn_encoder else None,
            cross_attn_init_by_pooling=args.cross_attn_init_by_pooling,
            # Defaults
            head_dim=args.head_dim,
            max_seqlen=args.max_encoder_seq_length,
            dropout=args.dropout,
            vocab_size=args.vocab_size + args.pm_size,
            norm_eps=args.norm_eps,
            patch_size=args.patch_size,
            sliding_window=args.local_attention_window_len,
            use_rope=args.use_rope,
            rope_theta=args.rope_theta,
            rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
            init_base_std=args.init_base_std,
            init_std_factor=args.init_std_factor,
            n_kv_heads=args.n_kv_heads,
            attn_impl=args.attn_impl,
            attn_bias_type="local_block_causal",
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            patching_mode=args.patching_mode,
            use_local_encoder_transformer=args.use_local_encoder_transformer,
            downsampling_by_pooling=args.downsampling_by_pooling,
            encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
            cross_attn_all_layers_encoder=args.cross_attn_all_layers_encoder,
            cross_attn_all_layers_decoder=args.cross_attn_all_layers_decoder,
            cross_attn_nheads=args.cross_attn_nheads,
            eos_id=args.eos_id,
        )
        self.local_encoder = LocalEncoder(local_encoder_args)
        local_decoder_args = LocalModelArgs(
            dim=args.dim_local_decoder,
            n_layers=args.n_layers_local_decoder,
            n_heads=args.n_heads_local_decoder,
            dim_token_emb=get_decoder_dim_token_emb(args),
            dim_patch_emb=args.dim_global,
            cross_attn_encoder=False,
            cross_attn_decoder=args.cross_attn_decoder,
            cross_attn_init_by_pooling=False,  # states are already defined
            cross_attn_k=args.cross_attn_k if args.cross_attn_decoder else None,
            # Defaults
            head_dim=args.head_dim,
            max_seqlen=args.max_encoder_seq_length,
            dropout=args.dropout,
            vocab_size=args.vocab_size + args.pm_size,
            norm_eps=args.norm_eps,
            patch_size=args.patch_size,
            sliding_window=args.local_attention_window_len,
            use_rope=args.use_rope,
            rope_theta=args.rope_theta,
            rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
            init_base_std=args.init_base_std,
            init_std_factor=args.init_std_factor,
            n_kv_heads=args.n_kv_heads,
            attn_impl=args.attn_impl,
            attn_bias_type="local_block_causal",
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            patching_mode=args.patching_mode,
            use_local_encoder_transformer=args.use_local_encoder_transformer,
            downsampling_by_pooling=args.downsampling_by_pooling,
            encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
            cross_attn_all_layers_encoder=args.cross_attn_all_layers_encoder,
            cross_attn_all_layers_decoder=args.cross_attn_all_layers_decoder,
            cross_attn_nheads=args.cross_attn_nheads,
            eos_id=args.eos_id,
        )

        global_args = args.model_copy(
            deep=True,
            update=dict(
                dim=args.dim_global,
                n_layers=args.n_layers_global,
                n_heads=args.n_heads_global,
                n_kv_heads=args.n_kv_heads_global,
                local_attention_window_len=None,
                dim_token_emb=get_global_dim_patch_emb(args),
                dim_patch_emb=None,
                cross_attn_encoder=False,
                cross_attn_decoder=False,
            ),
        )

        self.global_transformer = GlobalTransformer(global_args)

        self.local_decoder = LocalDecoder(local_decoder_args)
        self.encoder_hash_tok_embedding = init_embeddings(
            args,
            EmbeddingType.HASH_TOK,
            local_encoder_dim=self.local_encoder.dim,
            encoder_hash_byte_group_size=self.encoder_hash_byte_group_size,
        )
        self.encoder_ngram_embedding = init_embeddings(
            args,
            EmbeddingType.NGRAM,
            local_encoder_dim=self.local_encoder.dim,
            encoder_hash_byte_group_size=None,
        )

        # Encoder ngram embedding tables
        self.encoder_ngram_embedding = None
        if args.encoder_enable_byte_ngrams:
            self.encoder_ngram_embedding = nn.ModuleList()
            assert args.ngram_vocab_sizes is not None
            self.encoder_ngram_to_size = parse_ngram_to_size(
                args.encoder_ngram_to_size_str
            )
            ngram_emb_dim = self.local_encoder.dim
            for ngram_vocab_size in self.encoderngram_to_size.values():
                self.encoder_ngram_embedding.append(
                    nn.Embedding(ngram_vocab_size + OFFSET, ngram_emb_dim)
                )

        # Output layer
        assert args.vocab_size > 0, "vocab_size must be greater than 0"

        # Patcher module
        if args.patch_in_forward:
            self.patcher = Patcher(
                PatcherArgs(
                    patch_size=args.patch_size,
                    patching_mode=args.patching_mode,
                    patching_threshold=args.patching_threshold,
                    patching_threshold_add=args.patching_threshold_add,
                    monotonicity=args.monotonicity,
                    max_patch_length=args.max_patch_length,
                    entropy_model_checkpoint_dir=args.entropy_model_checkpoint_dir
                )
            )

    def push_to_hub(self, *args, **kwargs):
        raise ValueError(
            "For meta authors: Do not push BLT weights with this, save weights with save_pretrained() then push them manually to HF hub to ensure the repository metadata is correct."
        )

    def get_output_seq_len(self):
        return self.max_seqlen

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lengths: Optional[torch.Tensor] = None,
        ngram_ids: Optional[torch.Tensor] = None,
    ):
        # Ensure ngram_ids is either a tensor or None
        assert (
            isinstance(ngram_ids, torch.Tensor) or ngram_ids is None
        ), f"ngram_ids must be a tensor or None, but was: {type(ngram_ids)}"

        bs, N = tokens.shape  # Batch size and sequence length

        # Get megabyte inputs
        nb_boe = int(0 if self.patching_mode != "" else self.patch_size - 1)
        local_encoder_tokens, _, local_decoder_tokens = get_blt_input(
            tokens=tokens,
            enforce_patch_size_multiple=False,
            nb_boe=nb_boe,
            patch_size=self.patch_size,
            boe_id=self.boe_id,
        )

        # Patching
        if patch_lengths is None:
            assert (
                getattr(self, "patcher", None) is not None
            ), "Patcher not defined and no patch_lengths passed."
            patch_lengths, tok_scores = self.patcher.patch(
                local_encoder_tokens,
                include_next_token=True,
                threshold=self.patcher.threshold,
            )
        else:
            if nb_boe > 0:
                patch_lengths[:, 0] += nb_boe

        assert torch.min(patch_lengths) >= 0

        # Generate patch IDs from patch_lengths
        patch_ids = patch_ids_from_lengths(
            patch_lengths, local_encoder_tokens.shape[-1]
        )
        assert torch.max(patch_ids) + 1 <= torch.max(
            (patch_lengths != 0).sum(dim=-1)
        ), f"{torch.max(patch_ids) + 1} > {torch.max((patch_lengths != 0).sum(dim=-1))}"

        cross_attn_mask_enc = None
        # Cross-attention encoder
        if self.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                N,
                patches_as_queries=True,
                cross_attn_k=self.cross_attn_k,
                window=self.cross_attn_window_encoder,
                block_mask=self.cross_attn_use_flex_attention,
            )

        # Hashing and embedding
        local_encoder_embeds = compute_hash_embeddings(
            local_encoder_tokens=local_encoder_tokens,
            local_encoder=self.local_encoder,
            encoder_hash_tok_embedding=self.encoder_hash_tok_embedding,
            encoder_hash_byte_group_nb_functions=self.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.encoder_hash_byte_group_vocab,
        )

        # N-gram table embeddings
        if self.encoder_ngram_embedding is not None:
            assert ngram_ids is not None, "ngram_ids must be provided"
            if local_encoder_embeds is None:
                local_encoder_embeds = self.local_encoder.tok_embeddings(
                    local_encoder_tokens
                )
            assert len(ngram_ids) == len(
                self.encoder_ngram_embedding
            ), f"ngram_ids.shape[0]={ngram_ids.shape[0]} versus len(encoder_ngram_embedding)={len(self.encoder_ngram_embedding)}, ngram_ids.shape={ngram_ids.shape}"
            for i in range(ngram_ids.shape[0]):
                ngram_embedding = self.encoder_ngram_embedding[i]
                ngram_embeds = ngram_embedding(ngram_ids[i])
                assert (
                    local_encoder_embeds.shape == ngram_embeds.shape
                ), f"Shape mismatch: {local_encoder_embeds.shape} vs {ngram_embeds.shape}, ngram_ids.shape={ngram_ids.shape}"
                local_encoder_embeds = local_encoder_embeds + ngram_embeds

        # Local encoder
        (h_encoder, h_cross), cache_encoder = self.local_encoder(
            tokens=local_encoder_tokens,
            embeds=local_encoder_embeds,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        # Downsampling
        h = h_cross.view(bs, patch_lengths.shape[1], -1)

        # Global transformer
        global_tokens = tokens.new(h.shape[0], h.shape[1]).fill_(self.boe_id)
        rows, cols = torch.where(local_encoder_tokens == self.eos_id)
        eos_patch_ids = patch_ids[rows, cols]
        global_tokens[rows, eos_patch_ids] = self.eos_id

        h, _ = self.global_transformer(
            embeds=h,
            tokens=global_tokens,
        )

        # Unpatching
        dec_embeds = h_encoder[:, nb_boe : nb_boe + N, :]

        # Generate decoder patch IDs
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, nb_boe, local_decoder_tokens.shape[-1]
        )
        assert (
            torch.max(decoder_patch_ids) + 1 <= h.shape[1]
        ), f"{torch.max(decoder_patch_ids) + 1} > {h.shape[1]}"
        assert (
            decoder_patch_ids.shape[1] == dec_embeds.shape[1]
        ), f"{decoder_patch_ids.shape[1]} != {dec_embeds.shape[1]}"

        # Cross-attention decoder
        if not self.cross_attn_decoder:
            h = torch.gather(
                h, 1, decoder_patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            )
            cross_attn_mask_dec = None
            assert local_decoder_tokens.shape == h.shape[:-1]
        else:
            cross_attn_mask_dec = cross_attn_mask(
                decoder_patch_ids,
                patch_lengths,
                N,
                patches_as_queries=False,
                cross_attn_k=self.cross_attn_k,
                window=self.cross_attn_window_decoder,
                block_mask=self.cross_attn_use_flex_attention,
            )

        # Local decoder
        output, _ = self.local_decoder(
            embeds=dec_embeds,
            patch_embeds=h,
            tokens=local_decoder_tokens,
            cross_mask=cross_attn_mask_dec,
        )
        return output

    def init_weights(self):
        self.local_encoder.init_weights()
        self.global_transformer.init_weights()
        self.local_decoder.init_weights()

        emb_std = self.local_encoder.dim ** (-0.5)
        for emb in self.encoder_hash_tok_embedding:
            nn.init.trunc_normal_(
                emb.weight,
                mean=0.0,
                std=emb_std,
                a=-3 * emb_std,
                b=3 * emb_std,
            )