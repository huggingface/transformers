import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################
# import fvcore.nn.weight_init as weight_init
# from detectron2.layers import get_norm
# from fairscale.nn.checkpoint import checkpoint_wrapper
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.seggpt.configuration_seggpt import SegGPTConfig
from transformers.utils import ModelOutput


SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST = {
    "microsoft/beit-base-patch16-224-pt22k": (
        "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k/resolve/main/config.json"
    ),
    # See all BEiT models at https://huggingface.co/models?filter=beit
}


class LayerNorm2D(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and variance normalization over the
    channel dimension for inputs that have shape (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_abs_pos(abs_pos, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Args:
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py #
    noqa B950
        attn (Tensor): attention map. q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis. rel_pos_w (Tensor): relative position
        embeddings (Lw, C) for width axis. q_size (Tuple): spatial sequence size of query q with (q_h, q_w). k_size
        (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )

    return attn


class SegGPTPatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
            padding=(0, 0),
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class SegGPTMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        out_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.embed_dim, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop(hidden_states)
        return hidden_states


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however, the original name is
    misleading as 'Drop Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    Args:
    normal distribution. The values are effectively drawn from the normal distribution :math:`\mathcal{N}(\text{mean},
    \text{std}^2)` with values outside :math:`[a, b]` redrawn until they are within the bounds. The method used for
    generating the random values works best when :math:`a \leq \text{mean} \leq b`.
        tensor: an n-dimensional `torch.Tensor` mean: the mean of the normal distribution std: the standard deviation
        of the normal distribution a: the minimum cutoff value b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5) >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class SegGPTAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        head_dim = config.embed_dim // config.num_attention_heads
        self.scale = head_dim**-0.5
        input_size = (config.image_size[0] // config.patch_size, config.image_size[1] // config.patch_size)
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            # trunc_normal_(self.rel_pos_h, std=0.02)
            # trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(
        self,
        hidden_states,
        output_attentions: Optional[bool] = False,
    ):
        B, H, W, _ = hidden_states.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(hidden_states).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        hidden_states = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        ouputs = self.proj(hidden_states)

        if output_attentions:
            return (ouputs, attn)

        return ouputs


class SegGPTBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        config,
        block_depth,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.attn = SegGPTAttention(
            config,
        )
        self.depth = block_depth
        self.reshape_mean = config.merge_index >= block_depth
        self.swap_img_tgts = config.merge_index == block_depth
        dpr_values = torch.linspace(0, config.drop_path_rate, config.num_blocks_in_group * config.num_group_blocks)
        self.drop_path_rate = dpr_values[block_depth]
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SegGPTMlp(config)

        # self.window_size = config.window_size

    def forward(
        self,
        hidden_state,
        output_attentions: Optional[bool] = None,
    ):
        shortcut = hidden_state
        hidden_state = self.norm1(hidden_state)

        hidden_state = self.attn(hidden_state, output_attentions=output_attentions)

        if output_attentions:
            hidden_state, attention_output = hidden_state

        prompt, inputs = hidden_state.split(hidden_state.shape[1] // 2, dim=1)
        if self.reshape_mean:
            num_prompts = hidden_state.shape[0] // 2
            inputs = inputs.reshape(2, num_prompts, -1)
            inputs = inputs.mean(dim=1, keepdim=True).expand_as(inputs)
            inputs = inputs.reshape(*prompt.shape)
        else:
            inputs = inputs.mean(dim=0, keepdim=True).expand_as(inputs)
        hidden_state = torch.cat([prompt, inputs], dim=1)

        hidden_state = shortcut + drop_path(hidden_state, drop_prob=self.drop_path_rate)
        hidden_state = hidden_state + drop_path(self.mlp(self.norm2(hidden_state)), drop_prob=self.drop_path_rate)

        if self.swap_img_tgts:
            hidden_state = (
                hidden_state[: hidden_state.shape[0] // 2] + hidden_state[hidden_state.shape[0] // 2 :]
            ) * 0.5

        if output_attentions:
            return hidden_state, attention_output

        return hidden_state


class SegGPTBlockGroup(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        blocks = []

        for i in range(config.num_blocks_in_group):
            block_depth = config.num_blocks_in_group * depth + i

            blocks.append(SegGPTBlock(config, block_depth))

        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        hidden_state,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        hidden_states = []
        attention_outputs = []
        for block in self.blocks:
            hidden_state = block(hidden_state, output_attentions=output_attentions)
            if output_attentions:
                hidden_state, attention = hidden_state
                attention_outputs.append(attention)
            hidden_states.append(hidden_state)

        if not return_dict:
            outputs = (hidden_state,)
            outputs = outputs + (hidden_states,) if output_hidden_states else outputs
            return outputs + (attention_outputs,) if output_attentions else outputs

        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=hidden_states, attentions=attention_outputs
        )


class SegGPTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.patch_embed = SegGPTPatchEmbed(config)
        self.num_patches = (config.image_size[0] // config.patch_size) * (config.image_size[1] // config.patch_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, config.embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, config.embed_dim))
        # token for seg types
        self.type_token_cls = nn.Parameter(torch.zeros(1, 1, 1, config.embed_dim))
        self.type_token_ins = nn.Parameter(torch.zeros(1, 1, 1, config.embed_dim))

        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (config.pretrain_img_size // config.patch_size) * (config.pretrain_img_size // config.patch_size)
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, config.embed_dim), requires_grad=True)

        # stochastic depth decay rule
        depth = config.num_group_blocks * config.num_blocks_in_group
        [x.item() for x in torch.linspace(0, config.drop_path_rate, depth)]

        self.group_blocks = nn.ModuleList()
        for i in range(config.num_group_blocks):
            block = SegGPTBlockGroup(config, depth=i)
            self.group_blocks.append(block)

        self._out_feature_channels = {config.out_feature: config.embed_dim}
        self._out_feature_strides = {config.out_feature: config.patch_size}
        self._out_features = [config.out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        imgs,
        tgts,
        seg_type=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # embed patches
        x = self.patch_embed(imgs)
        y = self.patch_embed(tgts)
        batch_size, Hp, Wp, _ = x.size()

        mask_token = self.mask_token.expand(batch_size, Hp, Wp, -1)
        # replace the masked visual tokens by mask_token

        bool_masked_pos = torch.zeros(self.num_patches)
        bool_masked_pos[self.num_patches // 2 :] = 1
        bool_masked_pos = bool_masked_pos.unsqueeze(dim=0).flatten(1).to(torch.bool)

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, Hp, Wp, 1)
        y = y * (1 - w) + mask_token * w

        # add pos embed w/o cls token
        x = x + self.segment_token_x
        y = y + self.segment_token_y
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, (x.shape[1], x.shape[2]))
            y = y + get_abs_pos(self.pos_embed, (y.shape[1], y.shape[2]))

        # add type tokens for cls and ins
        type_emb = torch.zeros(batch_size, 1, 1, self.type_token_cls.shape[-1]).to(x.device)
        type_emb[seg_type == 0] = self.type_token_cls
        type_emb[seg_type == 1] = self.type_token_ins

        x = x + type_emb
        y = y + type_emb
        hidden_state = torch.cat((x, y), dim=0)
        # apply Transformer blocks
        last_hidden_states = []
        all_hidden_states = [hidden_state]
        all_attention_outputs = []
        for idx, block_group in enumerate(self.group_blocks):
            group_block_outputs = block_group(
                hidden_state,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_state = group_block_outputs[0]
            if output_hidden_states:
                all_hidden_states.extend(group_block_outputs[1])
            if output_attentions:
                all_attention_outputs.extend(group_block_outputs[-1])
            last_hidden_states.append(self.norm(hidden_state))

        last_hidden_state = torch.cat(last_hidden_states, dim=-1)

        if not return_dict:
            outputs = (last_hidden_state,)
            outputs = outputs + (all_hidden_states,) if output_hidden_states else outputs
            return outputs + (all_attention_outputs,) if output_attentions else outputs

        return BaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=all_hidden_states, attentions=all_attention_outputs
        )


class SegGPTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.decoder_embed_dim = config.decoder_embed_dim
        self.decoder_embed = nn.Linear(
            config.embed_dim * 4, config.patch_size**2 * config.decoder_embed_dim, bias=True
        )  # decoder to patch
        self.decoder_pred = nn.Sequential(
            nn.Conv2d(
                config.decoder_embed_dim,
                config.decoder_embed_dim,
                kernel_size=3,
                padding=1,
            ),
            LayerNorm2D(config.decoder_embed_dim),
            nn.GELU(),
            nn.Conv2d(config.decoder_embed_dim, 3, kernel_size=1, bias=True),  # decoder to patch
        )

    def forward(self, hidden_states):
        hidden_states = self.decoder_embed(hidden_states)  # BxhxwxC
        p = self.patch_size
        h, w = hidden_states.shape[1], hidden_states.shape[2]
        hidden_states = hidden_states.reshape(shape=(hidden_states.shape[0], h, w, p, p, self.decoder_embed_dim))
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = hidden_states.reshape(shape=(hidden_states.shape[0], -1, h * p, w * p))

        hidden_states = self.decoder_pred(hidden_states)  # Bx3xHxW
        return hidden_states


class SegGPTPretrainedModel(PreTrainedModel):
    config_class = SegGPTConfig
    base_model_prefix = "segGPT"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SegGPTEncoder):
            factor = self.config.initializer_range
            nn.init.normal_(module.segment_token_x, mean=0.0, std=0.5 * factor)
            nn.init.normal_(module.segment_token_y, mean=0.0, std=0.5 * factor)
            nn.init.normal_(module.type_token_cls, mean=0.0, std=0.5 * factor)
            nn.init.normal_(module.type_token_ins, mean=0.0, std=0.5 * factor)
            nn.init.normal_(module.pos_embed, mean=0.0, std=0.5 * factor)


@dataclass
class SegGPTModelOutput(ModelOutput):
    """
    Class for outputs of [`MaskFormerForInstanceSegmentation`].

    This output can be directly passed to [`~MaskFormerImageProcessor.post_process_semantic_segmentation`] or or
    [`~MaskFormerImageProcessor.post_process_instance_segmentation`] or
    [`~MaskFormerImageProcessor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~MaskFormerImageProcessor] for details regarding usage.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction of the decoder.
        last_hidden_state (`torch.FloatTensor`):
            Last hidden states (final feature map) of the last stage of the encoder model .
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SegGPTModel(SegGPTPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # --------------------------------------------------------------------------
        self.encoder = SegGPTEncoder(config)

        self.decoder = SegGPTDecoder(config)
        self.num_patches = (config.image_size[0] // config.patch_size) * (config.image_size[1] // config.patch_size)

        self.apply(self._init_weights)
        self.patch_size = config.patch_size

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == 2 * imgs.shape[3] and imgs.shape[2] % p == 0

        w = imgs.shape[3] // p
        h = w * 2
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
        """
        p = self.patch_size
        w = int((x.shape[1] * 0.5) ** 0.5)
        h = w * 2
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward_decoder(self, x):
        x = self.decoder_embed(x)  # BxhxwxC
        p = self.patch_size
        h, w = x.shape[1], x.shape[2]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.decoder_embed_dim))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(x.shape[0], -1, h * p, w * p))

        x = self.decoder_pred(x)  # Bx3xHxW
        return x

    def loss(self, pred, tgts):
        """
        tgts: [N, 3, H, W] pred: [N, 3, H, W] mask: [N, L], 0 is keep, 1 is remove, valid: [N, 3, H, W]
        """

        bool_masked_pos = torch.zeros(self.num_patches)
        bool_masked_pos[self.num_patches // 2 :] = 1
        bool_masked_pos = bool_masked_pos.unsqueeze(dim=0).flatten(1).to(torch.bool)
        bool_masked_pos = bool_masked_pos[:, :, None].repeat(1, 1, self.patch_size**2 * 3)

        mask = self.unpatchify(bool_masked_pos)
        to_expand = tgts.shape[0] // mask.shape[0]
        mask = mask.repeat(to_expand, 1, 1, 1)

        target = tgts
        loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        pixel_values,
        prompt_pixel_values,
        seg_type=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # bool_masked_pos = torch.zeros(self.encoder.num_patches)
        # bool_masked_pos[self.encoder.num_patches // 2:] = 1
        # bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
        # bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        # if bool_masked_pos is None:
        #     bool_masked_pos = torch.zeros((imgs.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(
        #         imgs.device)
        # else:

        encoder_outputs = self.encoder(
            pixel_values,
            prompt_pixel_values,
            seg_type,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        decoder_output = self.decoder(encoder_outputs[0])
        logits = self.patchify(decoder_output)  # [N, L, p*p*3]
        loss = self.loss(decoder_output, prompt_pixel_values)

        if not return_dict:
            outputs = (loss, logits, encoder_outputs[0])
            outputs = outputs + (encoder_outputs[1],) if output_hidden_states else outputs
            return outputs + (encoder_outputs[-1],) if output_attentions else outputs

        return SegGPTModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if output_hidden_states else None,
            attentions=encoder_outputs[-1] if output_attentions else None,
        )


class SegGPTForInstanceSegmentation(SegGPTPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.seggpt_model = SegGPTModel(config)

    def forward(
        self,
        pixel_values,
        prompt_pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        seg_type = torch.ones([prompt_pixel_values.shape[0], 1])
        return self.seggpt_model(
            pixel_values, prompt_pixel_values, seg_type, output_attentions, output_hidden_states, return_dict
        )


class SegGPTForSemanticSegmentation(SegGPTPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.seggpt_model = SegGPTModel(config)

    def forward(
        self,
        pixel_values,
        prompt_pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        seg_type = torch.zeros([prompt_pixel_values.shape[0], 1])
        return self.seggpt_model(
            pixel_values, prompt_pixel_values, seg_type, output_attentions, output_hidden_states, return_dict
        )
