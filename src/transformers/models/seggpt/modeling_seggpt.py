import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################
# import fvcore.nn.weight_init as weight_init
# from detectron2.layers import get_norm
# from fairscale.nn.checkpoint import checkpoint_wrapper
from transformers import PreTrainedModel, add_start_docstrings
from transformers.modeling_outputs import BaseModelOutput, SemanticSegmenterOutput
from transformers.models.seggpt.configuration_seggpt import SegGPTConfig
from transformers.utils import ModelOutput, add_start_docstrings_to_model_forward, replace_return_docstrings


_CONFIG_FOR_DOC = "SegGPTConfig"

SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Raghavan/seggpt_semantic_segmentation",
]

SEGGPT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SegGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SEGGPT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size * num_prompts, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`SegGPTImageProcessor.__call__`]
            for details. Along with pixel values, the suppiled input promp(s) image needs to stitched, 
            Use SegGPTImageProcessor.pre_process_semantic_segmenation prepare the input.

        prompt_pixel_values (`torch.FloatTensor` of shape `(batch_size * num_prompts, num_channels, height, width)`):
            Prompt pixel values. Prompt pixel values can be obtained using [`AutoImageProcessor`]. See
            [`SegGPTImageProcessor.__call__`] for details. Use SegGPTImageProcessor.pre_process_semantic_segmenation 
            to prepare the input. 

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


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
            hidden_size (int):  hidden_size (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
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
        out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
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


def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == 2 * imgs.shape[3] and imgs.shape[2] % patch_size == 0

    w = imgs.shape[3] // patch_size
    h = w * 2
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size ** 2 * 3))
    return x


class SegGPTAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim ** -0.5
        input_size = (config.image_size[0] // config.patch_size, config.image_size[1] // config.patch_size)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    # copied from transformers.models.bert.modeling_sam.SamVisionAttention.get_rel_pos
    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    # copied from transformers.models.bert.modeling_sam.SamVisionAttention.add_decomposed_rel_pos
    def add_decomposed_rel_pos(
            self,
            attn: torch.Tensor,
            query: torch.Tensor,
            rel_pos_h: torch.Tensor,
            rel_pos_w: torch.Tensor,
            q_size: Tuple[int, int],
            k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn

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
            attn = self.add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

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
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SegGPTAttention(
            config,
        )
        self.depth = block_depth
        self.reshape_mean = config.merge_index >= block_depth
        self.swap_img_tgts = config.merge_index == block_depth
        dpr_values = torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        self.drop_path_rate = dpr_values[block_depth]
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SegGPTMlp(config)

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
                                   hidden_state[: hidden_state.shape[0] // 2] + hidden_state[
                                                                                hidden_state.shape[0] // 2:]
                           ) * 0.5

        if output_attentions:
            return hidden_state, attention_output

        return hidden_state


class SegGPTBlockGroup(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        blocks = []
        num_blocks_in_group = config.num_hidden_layers // config.num_group_blocks
        for i in range(num_blocks_in_group):
            block_depth = num_blocks_in_group * depth + i

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
        self.patch_embed = SegGPTPatchEmbed(config)
        self.num_patches = (config.image_size[0] // config.patch_size) * (config.image_size[1] // config.patch_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        # token for seg types
        self.type_token_cls = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.type_token_ins = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))

        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (config.pretrain_img_size // config.patch_size) * (config.pretrain_img_size // config.patch_size)
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, config.hidden_size), requires_grad=True)

        # stochastic depth decay rule

        [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]

        self.group_blocks = nn.ModuleList()
        for i in range(config.num_group_blocks):
            block = SegGPTBlockGroup(config, depth=i)
            self.group_blocks.append(block)

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        bool_masked_pos[self.num_patches // 2:] = 1
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
        self.decoder_hidden_size = config.decoder_hidden_size
        self.decoder_embed = nn.Linear(
            config.hidden_size * 4, config.patch_size ** 2 * config.decoder_hidden_size, bias=True
        )  # decoder to patch
        self.decoder_pred = nn.Sequential(
            nn.Conv2d(
                config.decoder_hidden_size,
                config.decoder_hidden_size,
                kernel_size=3,
                padding=1,
            ),
            LayerNorm2D(config.decoder_hidden_size),
            nn.GELU(),
            nn.Conv2d(config.decoder_hidden_size, 3, kernel_size=1, bias=True),  # decoder to patch
        )

    def forward(self, hidden_states):
        hidden_states = self.decoder_embed(hidden_states)  # BxhxwxC
        p = self.patch_size
        h, w = hidden_states.shape[1], hidden_states.shape[2]
        hidden_states = hidden_states.reshape(shape=(hidden_states.shape[0], h, w, p, p, self.decoder_hidden_size))
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = hidden_states.reshape(shape=(hidden_states.shape[0], -1, h * p, w * p))

        hidden_states = self.decoder_pred(hidden_states)  # Bx3xHxW
        return hidden_states


class SegGPTPreTrainedModel(PreTrainedModel):
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
    Class for outputs of [`SegGPT Base Model`].


    Args:
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

    logits: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class InstanceSegmenterOutput(SemanticSegmenterOutput):
    """
    Base class for outputs of Instance segmentation..

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """


@add_start_docstrings(
    "The bare SegGPT Model transformer outputting raw hidden-states without any specific head on top.",
    SEGGPT_START_DOCSTRING,
)
class SegGPTModel(SegGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # --------------------------------------------------------------------------
        self.encoder = SegGPTEncoder(config)

        self.apply(self._init_weights)
        self.patch_size = config.patch_size

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

    @add_start_docstrings_to_model_forward(SEGGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            pixel_values,
            prompt_pixel_values,
            seg_type=None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_outputs = self.encoder(
            pixel_values,
            prompt_pixel_values,
            seg_type,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = (encoder_outputs[0],)
            outputs = outputs + (encoder_outputs[1],) if output_hidden_states else outputs
            return outputs + (encoder_outputs[-1],) if output_attentions else outputs

        return BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if output_hidden_states else None,
            attentions=encoder_outputs[-1] if output_attentions else None,
        )


@add_start_docstrings(
    "The bare SegGPT Model transformer outputting raw hidden-states without any specific head on top.",
    SEGGPT_START_DOCSTRING,
)
class SegGPTForInstanceSegmentation(SegGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.seggpt_model = SegGPTModel(config)
        self.patch_size = config.patch_size
        self.decoder = SegGPTDecoder(config)
        self.num_patches = (config.image_size[0] // config.patch_size) * (config.image_size[1] // config.patch_size)

        self.post_init()

    @add_start_docstrings_to_model_forward(SEGGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            pixel_values,
            prompt_pixel_values,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        seg_type = torch.ones([prompt_pixel_values.shape[0], 1])
        outputs = self.seggpt_model(
            pixel_values, prompt_pixel_values, seg_type, output_attentions, output_hidden_states, return_dict
        )
        last_hidden_state = outputs[0]
        decoder_last_hidden_state = self.decoder(last_hidden_state)
        logits = patchify(decoder_last_hidden_state, self.patch_size)  # [N, L, p*p*3]

        if not return_dict:
            outputs = decoder_last_hidden_state
            outputs = outputs + (outputs[1],) if output_hidden_states else outputs
            return outputs + (outputs[-1],) if output_attentions else outputs

        return InstanceSegmenterOutput(
            logits=logits,
            hidden_states=outputs[1] if output_hidden_states else None,
            attentions=outputs[-1] if output_attentions else None,
        )


@add_start_docstrings(
    "The bare SegGPT Model transformer outputting raw hidden-states without any specific head on top.",
    SEGGPT_START_DOCSTRING,
)
class SegGPTForSemanticSegmentation(SegGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.seggpt_model = SegGPTModel(config)

        self.decoder = SegGPTDecoder(config)
        self.num_patches = (config.image_size[0] // config.patch_size) * (config.image_size[1] // config.patch_size)

        self.post_init()

    @add_start_docstrings_to_model_forward(SEGGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=InstanceSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            pixel_values,
            prompt_pixel_values,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        seg_type = torch.zeros([prompt_pixel_values.shape[0], 1])
        outputs = self.seggpt_model(
            pixel_values, prompt_pixel_values, seg_type, output_attentions, output_hidden_states, return_dict
        )

        last_hidden_state = outputs[0]
        decoder_last_hidden_state = self.decoder(last_hidden_state)
        logits = patchify(decoder_last_hidden_state, self.patch_size)  # [N, L, p*p*3]

        if not return_dict:
            outputs = decoder_last_hidden_state
            outputs = outputs + (outputs[1],) if output_hidden_states else outputs
            return outputs + (outputs[-1],) if output_attentions else outputs

        return SemanticSegmenterOutput(
            logits=logits,
            hidden_states=outputs[1] if output_hidden_states else None,
            attentions=outputs[-1] if output_attentions else None,
        )
