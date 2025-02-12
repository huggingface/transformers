import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops.einops import rearrange
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from transformers.models.superglue.modeling_superglue import KeypointMatchingOutput


INF = 1e9

if hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
    from torch.backends.cuda import sdp_kernel
else:
    FLASH_AVAILABLE = False


def format_output(mkpts0_f, mkpts1_f, mconf, m_bids, hw0_i):
    keypoints0 = mkpts0_f
    keypoints1 = mkpts1_f
    scores = mconf
    batch_idx = m_bids

    h, w = hw0_i
    keypoints0[:, 0] = keypoints0[:, 0] / w
    keypoints1[:, 0] = keypoints1[:, 0] / w
    keypoints0[:, 1] = keypoints0[:, 1] / h
    keypoints1[:, 1] = keypoints1[:, 1] / h

    unique_values, counts = torch.unique_consecutive(batch_idx, return_counts=True)
    split_keypoints0 = torch.split(keypoints0, counts.tolist())
    split_keypoints1 = torch.split(keypoints1, counts.tolist())
    split_scores = torch.split(scores, counts.tolist())
    split_mask = [torch.ones(size, device=keypoints0.device) for size in counts.tolist()]
    split_indices = [torch.arange(size, device=keypoints0.device) for size in counts.tolist()]

    keypoints0 = pad_sequence(split_keypoints0, batch_first=True)
    keypoints1 = pad_sequence(split_keypoints1, batch_first=True)
    keypoints = torch.stack([keypoints0, keypoints1], dim=1)
    matching_scores = pad_sequence(split_scores, batch_first=True)
    matching_scores = torch.stack([matching_scores, matching_scores], dim=1)
    mask = pad_sequence(split_mask, batch_first=True)
    mask = torch.stack([mask, mask], dim=1)
    matches = pad_sequence(split_indices, batch_first=True)
    matches = torch.stack([matches, matches], dim=1)

    output = KeypointMatchingOutput(
        loss=None,
        matches=matches,
        matching_scores=matching_scores,
        keypoints=keypoints,
        mask=mask,
        hidden_states=None,
        attentions=None,
    )

    return output


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels

        self.nonlinearity = nn.ReLU()

        self.rbr_identity = (
            nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        )
        self.rbr_dense = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
        )
        self.rbr_1x1 = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=1,
        )

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class RepVGG(nn.Module):
    def __init__(
        self,
        num_blocks,
        width_multiplier=None,
        use_checkpoint=False,
    ):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=1,
            out_channels=self.in_planes,
            stride=2,
        )
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    stride=stride,
                )
            )
            self.in_planes = planes
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class RepVGG_8_1_align(nn.Module):
    """
    RepVGG backbone, output resolution are 1/8 and 1.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        backbone = RepVGG(
            num_blocks=[2, 4, 14, 1],
            width_multiplier=[1, 1, 1, 2.5],
            use_checkpoint=False,
        )

        self.layer0, self.layer1, self.layer2, self.layer3 = (
            backbone.stage0,
            backbone.stage1,
            backbone.stage2,
            backbone.stage3,
        )

        for layer in [self.layer0, self.layer1, self.layer2, self.layer3]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer0(x)  # 1/2
        for module in self.layer1:
            out = module(out)  # 1/2
        x1 = out
        for module in self.layer2:
            out = module(out)  # 1/4
        x2 = out
        for module in self.layer3:
            out = module(out)  # 1/8
        x3 = out

        return {"feats_c": x3, "feats_f": None, "feats_x2": x2, "feats_x1": x1}


class RoPEPositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), npe=None, ropefp16=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        i_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(-1)  # [H, 1]
        j_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(-1)  # [W, 1]

        assert npe is not None
        train_res_H, train_res_W, test_res_H, test_res_W = (
            npe[0],
            npe[1],
            npe[2],
            npe[3],
        )  # train_res_H, train_res_W, test_res_H, test_res_W
        i_position, j_position = i_position * train_res_H / test_res_H, j_position * train_res_W / test_res_W

        div_term = torch.exp(torch.arange(0, d_model // 4, 1).float() * (-math.log(10000.0) / (d_model // 4)))
        div_term = div_term[None, None, :]  # [1, 1, C//4]

        sin = torch.zeros(*max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32)
        cos = torch.zeros(*max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32)
        sin[:, :, 0::2] = torch.sin(i_position * div_term).half() if ropefp16 else torch.sin(i_position * div_term)
        sin[:, :, 1::2] = torch.sin(j_position * div_term).half() if ropefp16 else torch.sin(j_position * div_term)
        cos[:, :, 0::2] = torch.cos(i_position * div_term).half() if ropefp16 else torch.cos(i_position * div_term)
        cos[:, :, 1::2] = torch.cos(j_position * div_term).half() if ropefp16 else torch.cos(j_position * div_term)

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        self.register_buffer("sin", sin.unsqueeze(0), persistent=False)  # [1, H, W, C//2]
        self.register_buffer("cos", cos.unsqueeze(0), persistent=False)  # [1, H, W, C//2]

    def forward(self, x, ratio=1):
        """
        Args:
            x: [N, H, W, C]
        """
        return (x * self.cos[:, : x.size(1), : x.size(2), :]) + (
            self.rotate_half(x) * self.sin[:, : x.size(1), : x.size(2), :]
        )

    def rotate_half(self, x):
        x = x.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def crop_feature(query, key, value, x_mask, source_mask):
    mask_h0, mask_w0, mask_h1, mask_w1 = (
        x_mask[0].sum(-2)[0],
        x_mask[0].sum(-1)[0],
        source_mask[0].sum(-2)[0],
        source_mask[0].sum(-1)[0],
    )
    query = query[:, :mask_h0, :mask_w0, :]
    key = key[:, :mask_h1, :mask_w1, :]
    value = value[:, :mask_h1, :mask_w1, :]
    return query, key, value, mask_h0, mask_w0


def pad_feature(m, mask_h0, mask_w0, x_mask):
    bs, L, H, D = m.size()
    m = m.view(bs, mask_h0, mask_w0, H, D)
    if mask_h0 != x_mask.size(-2):
        m = torch.cat(
            [
                m,
                torch.zeros(
                    m.size(0), x_mask.size(-2) - mask_h0, x_mask.size(-1), H, D, device=m.device, dtype=m.dtype
                ),
            ],
            dim=1,
        )
    elif mask_w0 != x_mask.size(-1):
        m = torch.cat(
            [
                m,
                torch.zeros(
                    m.size(0), x_mask.size(-2), x_mask.size(-1) - mask_w0, H, D, device=m.device, dtype=m.dtype
                ),
            ],
            dim=2,
        )
    return m


class Attention(nn.Module):
    def __init__(self, no_flash=False, nhead=8, dim=256, fp32=False):
        super().__init__()
        self.flash = FLASH_AVAILABLE and not no_flash
        self.nhead = nhead
        self.dim = dim
        self.fp32 = fp32

    def attention(self, query, key, value, q_mask=None, kv_mask=None):
        assert q_mask is None and kv_mask is None, "Not support generalized attention mask yet."
        if self.flash and not self.fp32:
            args = [x.contiguous() for x in [query, key, value]]
            with sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                out = F.scaled_dot_product_attention(*args)
        elif self.flash:
            args = [x.contiguous() for x in [query, key, value]]
            out = F.scaled_dot_product_attention(*args)
        else:
            QK = torch.einsum("nlhd,nshd->nlsh", query, key)

            # Compute the attention and the weighted average
            softmax_temp = 1.0 / query.size(3) ** 0.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)

            out = torch.einsum("nlsh,nshd->nlhd", A, value)
        return out

    def _forward(self, query, key, value, q_mask=None, kv_mask=None):
        if q_mask is not None:
            query, key, value, mask_h0, mask_w0 = crop_feature(query, key, value, q_mask, kv_mask)

        if self.flash:
            query, key, value = map(
                lambda x: rearrange(x, "n h w (nhead d) -> n nhead (h w) d", nhead=self.nhead, d=self.dim),
                [query, key, value],
            )
        else:
            query, key, value = map(
                lambda x: rearrange(x, "n h w (nhead d) -> n (h w) nhead d", nhead=self.nhead, d=self.dim),
                [query, key, value],
            )

        m = self.attention(query, key, value, q_mask=None, kv_mask=None)

        if self.flash:
            m = rearrange(m, "n nhead L d -> n L nhead d", nhead=self.nhead, d=self.dim)

        if q_mask is not None:
            m = pad_feature(m, mask_h0, mask_w0, q_mask)

        return m

    def forward(self, query, key, value, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            if FLASH_AVAILABLE: # pytorch scaled_dot_product_attention
                queries: [N, H, L, D]
                keys: [N, H, S, D]
                values: [N, H, S, D]
            else:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        bs = query.size(0)
        if bs == 1 or q_mask is None:
            m = self._forward(query, key, value, q_mask=q_mask, kv_mask=kv_mask)
        else:  # for faster trainning with padding mask while batch size > 1
            m_list = []
            for i in range(bs):
                m_list.append(
                    self._forward(
                        query[i : i + 1],
                        key[i : i + 1],
                        value[i : i + 1],
                        q_mask=q_mask[i : i + 1],
                        kv_mask=kv_mask[i : i + 1],
                    )
                )
            m = torch.cat(m_list, dim=0)
        return m


class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        agg_size0=4,
        agg_size1=4,
        no_flash=False,
        rope=False,
        npe=None,
        fp32=False,
    ):
        super(AG_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = (
            nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False, groups=d_model)
            if self.agg_size0 != 1
            else nn.Identity()
        )
        self.max_pool = (
            torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1)
            if self.agg_size1 != 1
            else nn.Identity()
        )
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature
        query, source = (
            self.norm1(self.aggregate(x).permute(0, 2, 3, 1)),
            self.norm1(self.max_pool(source).permute(0, 2, 3, 1)),
        )  # [N, H, W, C]
        if x_mask is not None:
            x_mask, source_mask = map(lambda x: self.max_pool(x.float()).bool(), [x_mask, source_mask])
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        m = self.merge(m.reshape(bs, -1, self.nhead * self.dim))  # [N, L, C]

        # Upsample feature
        m = rearrange(m, "b (h w) c -> b c h w", h=H0 // self.agg_size0, w=W0 // self.agg_size0)  # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(
                m, scale_factor=self.agg_size0, mode="bilinear", align_corners=False
            )  # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1))  # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2)  # [N, C, H0, W0]

        return x + m


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.fp32 = not (config.mp or config.half)
        config = config.coarse
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.layer_names = config.layer_names
        self.agg_size0, self.agg_size1 = config.agg_size0, config.agg_size1
        self.rope = config.rope

        self_layer = AG_RoPE_EncoderLayer(
            config.d_model,
            config.nhead,
            config.agg_size0,
            config.agg_size1,
            config.no_flash,
            config.rope,
            config.npe,
            self.fp32,
        )
        cross_layer = AG_RoPE_EncoderLayer(
            config.d_model,
            config.nhead,
            config.agg_size0,
            config.agg_size1,
            config.no_flash,
            False,
            config.npe,
            self.fp32,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(self_layer) if _ == "self" else copy.deepcopy(cross_layer) for _ in self.layer_names]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        H0, W0, H1, W1 = feat0.size(-2), feat0.size(-1), feat1.size(-2), feat1.size(-1)
        bs = feat0.shape[0]

        feature_cropped = False
        if bs == 1 and mask0 is not None and mask1 is not None:
            mask_H0, mask_W0, mask_H1, mask_W1 = mask0.size(-2), mask0.size(-1), mask1.size(-2), mask1.size(-1)
            mask_h0, mask_w0, mask_h1, mask_w1 = (
                mask0[0].sum(-2)[0],
                mask0[0].sum(-1)[0],
                mask1[0].sum(-2)[0],
                mask1[0].sum(-1)[0],
            )
            mask_h0, mask_w0, mask_h1, mask_w1 = (
                mask_h0 // self.agg_size0 * self.agg_size0,
                mask_w0 // self.agg_size0 * self.agg_size0,
                mask_h1 // self.agg_size1 * self.agg_size1,
                mask_w1 // self.agg_size1 * self.agg_size1,
            )
            feat0 = feat0[:, :, :mask_h0, :mask_w0]
            feat1 = feat1[:, :, :mask_h1, :mask_w1]
            feature_cropped = True

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if feature_cropped:
                mask0, mask1 = None, None
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        if feature_cropped:
            # padding feature
            bs, c, mask_h0, mask_w0 = feat0.size()
            if mask_h0 != mask_H0:
                feat0 = torch.cat(
                    [feat0, torch.zeros(bs, c, mask_H0 - mask_h0, mask_W0, device=feat0.device, dtype=feat0.dtype)],
                    dim=-2,
                )
            elif mask_w0 != mask_W0:
                feat0 = torch.cat(
                    [feat0, torch.zeros(bs, c, mask_H0, mask_W0 - mask_w0, device=feat0.device, dtype=feat0.dtype)],
                    dim=-1,
                )

            bs, c, mask_h1, mask_w1 = feat1.size()
            if mask_h1 != mask_H1:
                feat1 = torch.cat(
                    [feat1, torch.zeros(bs, c, mask_H1 - mask_h1, mask_W1, device=feat1.device, dtype=feat1.dtype)],
                    dim=-2,
                )
            elif mask_w1 != mask_W1:
                feat1 = torch.cat(
                    [feat1, torch.zeros(bs, c, mask_H1, mask_W1 - mask_w1, device=feat1.device, dtype=feat1.dtype)],
                    dim=-1,
                )

        return feat0, feat1


def mask_border(m, b: int, v):
    """Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd :] = v
        m[b_idx, :, w0 - bd :] = v
        m[b_idx, :, :, h1 - bd :] = v
        m[b_idx, :, :, :, w1 - bd :] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config.thr
        self.border_rm = config.border_rm
        self.temperature = config.dsmax_temperature
        self.skip_softmax = config.skip_softmax
        self.fp16matmul = config.fp16matmul
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config.train_coarse_percent
        self.train_pad_num_gt_min = config.train_pad_num_gt_min

    def forward(self, feat_c0, feat_c1, hw0_i, hw1_i, hw0_c, hw1_c, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1])

        if self.fp16matmul:
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            del feat_c0, feat_c1
            if mask_c0 is not None:
                sim_matrix = sim_matrix.masked_fill(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e4)
        else:
            with torch.autocast(enabled=False, device_type="cuda"):
                sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
                del feat_c0, feat_c1
                if mask_c0 is not None:
                    sim_matrix = sim_matrix.float().masked_fill(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)
        if self.skip_softmax:
            sim_matrix = sim_matrix
        else:
            sim_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        # predict coarse matches from conf_matrix
        # data.update(**self.get_coarse_match(sim_matrix, hw0_i, hw1_i, hw0_c, hw1_c))
        coarse_match_output = self.get_coarse_match(sim_matrix, hw0_i, hw1_i, hw0_c, hw1_c)

        outputs = (sim_matrix,) + coarse_match_output
        return outputs

    @torch.no_grad()
    def get_coarse_match(
        self,
        conf_matrix,
        hw0_i,
        hw1_i,
        hw0_c,
        hw1_c,
        mask0=None,
        mask1=None,
        spv_b_ids=None,
        spv_i_ids=None,
        spv_j_ids=None,
        scale0=None,
        scale1=None,
    ):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            "h0c": hw0_c[0],
            "w0c": hw0_c[1],
            "h1c": hw1_c[0],
            "w1c": hw1_c[1],
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask = rearrange(mask, "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c", **axes_lengths)

        if mask0 is None:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, mask0, mask1)
        mask = rearrange(mask, "b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)", **axes_lengths)

        # 2. mutual nearest
        mask = (
            mask
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        )

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if mask0 is not None:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(mask0, mask1)
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert (
                self.train_pad_num_gt_min < num_matches_train
            ), "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred, (num_matches_train - self.train_pad_num_gt_min,), device=_device
                )

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(spv_b_ids),
                (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
                device=_device,
            )
            mconf_gt = torch.zeros(len(spv_b_ids), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip(
                    [b_ids, spv_b_ids],
                    [i_ids, spv_i_ids],
                    [j_ids, spv_j_ids],
                    [mconf, mconf_gt],
                ),
            )

        # These matches select patches that feed into fine-level network
        # coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        # 4. Update with matches in original image resolution
        scale = hw0_i[0] / hw0_c[0]

        scale0 = scale
        scale1 = scale
        mkpts0_c = torch.stack([i_ids % hw0_c[1], i_ids // hw0_c[1]], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % hw1_c[1], j_ids // hw1_c[1]], dim=1) * scale1

        m_bids = b_ids[mconf != 0]
        # These matches is the current prediction (for visualization)
        # coarse_matches.update(
        #     {
        #         "m_bids": m_bids,  # mconf == 0 => gt matches
        #         "mkpts0_c": mkpts0_c[mconf != 0],
        #         "mkpts1_c": mkpts1_c[mconf != 0],
        #         "mconf": mconf[mconf != 0],
        #     }
        # )

        mkpts0_c = mkpts0_c[mconf != 0]
        mkpts1_c = mkpts1_c[mconf != 0]
        mconf = mconf[mconf != 0]

        return m_bids, mkpts0_c, mkpts1_c, mconf, b_ids, i_ids, j_ids, scale0, scale1


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        block_dims = config.backbone["block_dims"]
        self.W = self.config.fine_window_size
        self.fine_d_model = block_dims[0]

        self.layer3_outconv = nn.Conv2d(block_dims[2], block_dims[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.layer2_outconv = nn.Conv2d(block_dims[1], block_dims[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.layer2_outconv2 = nn.Sequential(
            nn.Conv2d(block_dims[2], block_dims[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            nn.Conv2d(block_dims[2], block_dims[1], kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layer1_outconv = nn.Conv2d(block_dims[0], block_dims[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.layer1_outconv2 = nn.Sequential(
            nn.Conv2d(block_dims[1], block_dims[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            nn.Conv2d(block_dims[1], block_dims[0], kernel_size=3, stride=1, padding=1, bias=False),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def inter_fpn(self, feat_c, x2, x1):
        feat_c = self.layer3_outconv(feat_c)
        feat_c = F.interpolate(feat_c, scale_factor=2.0, mode="bilinear", align_corners=False)

        x2 = self.layer2_outconv(x2)
        x2 = self.layer2_outconv2(x2 + feat_c)
        x2 = F.interpolate(x2, scale_factor=2.0, mode="bilinear", align_corners=False)

        x1 = self.layer1_outconv(x1)
        x1 = self.layer1_outconv2(x1 + x2)
        x1 = F.interpolate(x1, scale_factor=2.0, mode="bilinear", align_corners=False)
        return x1

    def forward(self, feat_c0, feat_c1, hw0_f, hw0_c, b_ids, hw0_i, hw1_i, feats_x2, feats_x1, i_ids, j_ids):
        W = self.W
        stride = hw0_f[0] // hw0_c[0]

        if b_ids.shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.fine_d_model, device=feat_c0.device)
            feat1 = torch.empty(0, self.W**2, self.fine_d_model, device=feat_c0.device)
            return feat0, feat1

        if hw0_i == hw1_i:
            feat_c = rearrange(torch.cat([feat_c0, feat_c1], 0), "b (h w) c -> b c h w", h=hw0_c[0])  # 1/8 feat
            x2 = feats_x2  # 1/4 feat
            x1 = feats_x1  # 1/2 feat

            # 1. fine feature extraction
            x1 = self.inter_fpn(feat_c, x2, x1)
            feat_f0, feat_f1 = torch.chunk(x1, 2, dim=0)

            # 2. unfold(crop) all local windows
            feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
            feat_f0 = rearrange(feat_f0, "n (c ww) l -> n l ww c", ww=W**2)
            feat_f1 = F.unfold(feat_f1, kernel_size=(W + 2, W + 2), stride=stride, padding=1)
            feat_f1 = rearrange(feat_f1, "n (c ww) l -> n l ww c", ww=(W + 2) ** 2)

            # 3. select only the predicted matches
            feat_f0 = feat_f0[b_ids, i_ids]  # [n, ww, cf]
            feat_f1 = feat_f1[b_ids, j_ids]

            return feat_f0, feat_f1


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature = config.match_fine["local_regress_temperature"]
        self.local_regress_slicedim = config.match_fine["local_regress_slicedim"]
        self.fp16 = config.half
        self.validate = False

    def forward(self, feat_0, feat_1, hw0_i, hw0_f, mkpts0_c, mkpts1_c, mconf, b_ids, scale0, scale1, bs):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW))
        scale = hw0_i[0] / hw0_f[0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            conf_matrix_f = (torch.empty(0, WW, WW, device=feat_0.device),)
            mkpts0_f = mkpts0_c
            mkpts1_f = mkpts1_c
            return mkpts0_f, mkpts1_f, conf_matrix_f, None

        # compute pixel-level confidence matrix
        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type="cuda"):
            feat_f0, feat_f1 = feat_0[..., : -self.local_regress_slicedim], feat_1[..., : -self.local_regress_slicedim]
            feat_ff0, feat_ff1 = (
                feat_0[..., -self.local_regress_slicedim :],
                feat_1[..., -self.local_regress_slicedim :],
            )
            feat_f0, feat_f1 = feat_f0 / C**0.5, feat_f1 / C**0.5
            conf_matrix_f = torch.einsum("mlc,mrc->mlr", feat_f0, feat_f1)
            conf_matrix_ff = torch.einsum("mlc,mrc->mlr", feat_ff0, feat_ff1 / (self.local_regress_slicedim) ** 0.5)

        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W + 2, self.W + 2)
        softmax_matrix_f = softmax_matrix_f[..., 1:-1, 1:-1].reshape(M, self.WW, self.WW)

        # for fine-level supervision
        # if self.training or self.validate:
        # data.update({"sim_matrix_ff": conf_matrix_ff})
        # data.update({"conf_matrix_f": softmax_matrix_f})

        # compute pixel-level absolute kpt coords
        idx_l, idx_r, mkpts0_c, mkpts1_c = self.get_fine_ds_match(
            softmax_matrix_f, mconf, mkpts0_c, mkpts1_c, b_ids, scale0, scale1
        )

        # generate seconde-stage 3x3 grid
        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)
        m_ids = m_ids[: len(mconf)]
        idx_r_iids, idx_r_jids = idx_r // W, idx_r % W

        m_ids, idx_l, idx_r_iids, idx_r_jids = (
            m_ids.reshape(-1),
            idx_l.reshape(-1),
            idx_r_iids.reshape(-1),
            idx_r_jids.reshape(-1),
        )
        delta = create_meshgrid(3, 3, True, conf_matrix_ff.device).to(torch.long)  # [1, 3, 3, 2]

        m_ids = m_ids[..., None, None].expand(-1, 3, 3)
        idx_l = idx_l[..., None, None].expand(-1, 3, 3)  # [m, k, 3, 3]

        idx_r_iids = idx_r_iids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 0]

        if idx_l.numel() == 0:
            mkpts0_f = mkpts0_c
            mkpts1_f = mkpts1_c
            return mkpts0_f, mkpts1_f, conf_matrix_ff, softmax_matrix_f

        # compute second-stage heatmap
        conf_matrix_ff = conf_matrix_ff.reshape(M, self.WW, self.W + 2, self.W + 2)
        conf_matrix_ff = conf_matrix_ff[m_ids, idx_l, idx_r_iids, idx_r_jids]
        conf_matrix_ff = conf_matrix_ff.reshape(-1, 9)
        conf_matrix_ff = F.softmax(conf_matrix_ff / self.local_regress_temperature, -1)
        heatmap = conf_matrix_ff.reshape(-1, 3, 3)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]

        if bs == 1:
            scale1 = scale
        else:
            scale1 = (
                scale * scale1[b_ids][: len(mconf), ...][:, None, :].expand(-1, -1, 2).reshape(-1, 2)
                if scale0 is not None
                else scale
            )

        # compute subpixel-level absolute kpt coords
        mkpts0_f, mkpts1_f = self.get_fine_match_local(coords_normalized, scale1, mkpts0_c, mkpts1_c)

        return mkpts0_f, mkpts1_f, conf_matrix_ff, softmax_matrix_f

    def get_fine_match_local(self, coords_normed, scale1, mkpts0_c, mkpts1_c):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normed * (3 // 2) * scale1)

        return mkpts0_f, mkpts1_f

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, mconf, mkpts0_c, mkpts1_c, b_ids, scale0=None, scale1=None):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        m, _, _ = conf_matrix.shape

        conf_matrix = conf_matrix.reshape(m, -1)[: len(mconf), ...]
        val, idx = torch.max(conf_matrix, dim=-1)
        idx = idx[:, None]
        idx_l, idx_r = idx // WW, idx % WW

        if self.fp16:
            grid = (
                create_meshgrid(W, W, False, conf_matrix.device, dtype=torch.float16) - W // 2 + 0.5
            )  # kornia >= 0.5.1
        else:
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        scale0 = scale
        scale1 = scale

        if torch.is_tensor(scale0) and scale0.numel() > 1:  # scale0 is a tensor
            mkpts0_f = (mkpts0_c[:, None, :] + (delta_l * scale0[: len(mconf), ...][:, None, :])).reshape(-1, 2)
            mkpts1_f = (mkpts1_c[:, None, :] + (delta_r * scale1[: len(mconf), ...][:, None, :])).reshape(-1, 2)
        else:  # scale0 is a float
            mkpts0_f = (mkpts0_c[:, None, :] + (delta_l * scale0)).reshape(-1, 2)
            mkpts1_f = (mkpts1_c[:, None, :] + (delta_r * scale1)).reshape(-1, 2)

        return idx_l, idx_r, mkpts0_f, mkpts1_f


class EfficientLoFTR(nn.Module):
    def __init__(self, config, profiler=None, model_path=None):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler

        # Modules
        self.backbone = RepVGG_8_1_align(config.backbone)
        self.loftr_coarse = LocalFeatureTransformer(config)
        self.coarse_matching = CoarseMatching(config.match_coarse)
        self.fine_preprocess = FinePreprocess(config)
        self.fine_matching = FineMatching(config)

        if model_path is not None:
            self.load_state_dict(torch.load(model_path)["state_dict"])
            self.eval()

    def forward(self, pixel_values, mask0=None, mask1=None):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        pixel_values = pixel_values[:, :, 0, :, :][:, :, None, :, :]
        image0 = pixel_values[:, 0]
        image1 = pixel_values[:, 1]

        # 1. Local Feature CNN
        bs = image0.size(0)
        hw0_i = image0.shape[2:]
        hw1_i = image1.shape[2:]

        if hw0_i == hw1_i:  # faster & better BN convergence
            ret_dict = self.backbone(torch.cat([image0, image1], dim=0))
            feats_c = ret_dict["feats_c"]
            feats_x2 = ret_dict["feats_x2"]
            feats_x1 = ret_dict["feats_x1"]
            (feat_c0, feat_c1) = feats_c.split(bs)
        else:  # handle different input shapes
            ret_dict0, ret_dict1 = self.backbone(image0), self.backbone(image1)
            feat_c0 = ret_dict0["feats_c"]
            feat_c1 = ret_dict1["feats_c"]
            feats_x2_0 = ret_dict0["feats_x2"]
            feats_x1_0 = ret_dict0["feats_x1"]
            feats_x2_1 = ret_dict1["feats_x2"]
            feats_x1_1 = ret_dict1["feats_x1"]

        mul = self.config.resolution[0] // self.config.resolution[1]
        hw0_c = feat_c0.shape[2:]
        hw1_c = feat_c1.shape[2:]
        hw0_f = [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul]
        hw1_f = [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul]

        # 2. coarse-level loftr module
        mask_c0 = mask_c1 = None  # mask is useful in training
        if mask0 is not None:
            mask_c0, mask_c1 = mask0, mask1

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")

        # 3. match coarse-level
        conf_matrix, m_bids, mkpts0_c, mkpts1_c, mconf, b_ids, i_ids, j_ids, scale0, scale1 = self.coarse_matching(
            feat_c0,
            feat_c1,
            hw0_i,
            hw1_i,
            hw0_c,
            hw1_c,
            mask_c0=mask_c0.view(mask_c0.size(0), -1) if mask_c0 is not None else mask_c0,
            mask_c1=mask_c1.view(mask_c1.size(0), -1) if mask_c1 is not None else mask_c1,
        )

        # prevent fp16 overflow during mixed precision training
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1])

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
            feat_c0, feat_c1, hw0_f, hw0_c, b_ids, hw0_i, hw1_i, feats_x2, feats_x1, i_ids, j_ids
        )

        del feat_c0, feat_c1, mask_c0, mask_c1

        # 5. match fine-level
        mkpts0_f, mkpts1_f, conf_matrix_ff, softmax_matrix_f = self.fine_matching(
            feat_f0_unfold, feat_f1_unfold, hw0_i, hw0_f, mkpts0_c, mkpts1_c, mconf, b_ids, scale0, scale1, bs
        )

        output = format_output(mkpts0_f, mkpts1_f, mconf, m_bids, hw0_i)

        return output

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

    @staticmethod
    def get_default_config():
        with initialize_config_module(
            version_base=None, config_module="polaris.configs.image_matching.semi_dense_matcher"
        ):
            config = compose(config_name="eloftr")
            return config

    @staticmethod
    def get_default(weights_path: Path) -> "EfficientLoFTR":
        default_config = EfficientLoFTR.get_default_config()
        default_config.model_path = weights_path
        return instantiate(default_config)
