# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

"""Various utility models"""

import copy
import math
from typing import Optional, Union

import numpy as np

import torch
import torch.nn.functional as F

import torch.utils._pytree as pytree
from torch import nn, Tensor
from torch.nn.functional import pad


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DotProductScoring(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, torch.nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp  # an optional MLP projection for prompt
        self.prompt_proj = torch.nn.Linear(d_model, d_proj)
        self.hs_proj = torch.nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
        is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
        # num_valid has shape (bs, 1)
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        # mean pool over all the valid tokens -- pooled_prompt has shape (bs, proj_dim)
        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        # hs has shape (num_layer, bs, num_query, d_model)
        # prompt has shape (seq, bs, d_model)
        # prompt_mask has shape (bs, seq), where 1 is valid and 0 is padding
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2

        # apply MLP on prompt if specified
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)

        # first, get the mean-pooled version of the prompt
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)

        # then, project pooled_prompt and hs to d_proj dimensions
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # (bs, d_proj)
        proj_hs = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # finally, get dot-product scores of shape (num_layer, bs, num_query, 1)
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale

        # clamp scores to a max value to avoid numerical issues in loss or matcher
        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",  # ["none"] only for now
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries

        # for two stage
        assert two_stage_type in ["none"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if (
                    "box_embed" not in n
                    and "query_embed" not in n
                    and "reference_points" not in n
                ):
                    nn.init.xavier_uniform_(p)


class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def clone(self):
        new_tensors = self.tensors.clone()
        new_mask = None if self.mask is None else self.mask.clone()
        return NestedTensor(new_tensors, new_mask)

    @property
    def device(self):
        return self.tensors.device

    @property
    def shape(self):
        return self.tensors.shape

    # custom memory pinning method on custom type
    def pin_memory(self, device=None):
        self.tensors = self.tensors.pin_memory(device)
        if self.mask is not None:
            self.mask = self.mask.pin_memory(device)
        return self

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list, rounding=None):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            if rounding is not None:
                # Round to an even size to avoid rounding issues in fpn
                p = rounding
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, h, w

            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            mask_required = False
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
                if img.shape[1] != h or img.shape[2] != w:
                    mask_required = True

            if not mask_required:
                mask = None
        else:
            raise ValueError("not supported")
        return cls(tensor, mask)

    def to_tensor_list(self):
        """
        Undo the batching and padding, returning a list of tensors of
        different sizes.
        """
        data = self.tensors
        masks = self.mask.to(torch.int)
        # mask is 0 in the tensor and 1 outside it; argmax finds the smallest 1 index
        # and can be used to get heights and widths. Since padding is always to the
        # bottom and right, we can take the first value along x to get height and
        # visa versa for y and width. To handle cases with no padding, pad masks
        # out by one element so an edge is always found.
        masks = pad(masks, (0, 1, 0, 1), value=1)
        heights = torch.argmax(masks, dim=-2)
        assert torch.all(
            (heights == heights[:, 0:1]) | (heights == 0)
        ), "Image mask is not a box."
        heights = heights[:, 0]
        widths = torch.argmax(masks, dim=-1)
        assert torch.all(
            (widths == widths[:, 0:1]) | (widths == 0)
        ), "Image mask is not a box."
        widths = widths[:, 0]
        return [data[i, :, :h, :w] for i, (h, w) in enumerate(zip(heights, widths))]

    def __repr__(self):
        return repr(self.tensors)


pytree.register_pytree_node(
    NestedTensor,
    lambda x: ([x.tensors, x.mask], None),
    lambda values, _: NestedTensor(values[0], values[1]),
)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # whether to add the output as a residual connection to the input
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_clones_seq(module, N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_activation_module(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(pos_tensor, num_feats=256):
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
