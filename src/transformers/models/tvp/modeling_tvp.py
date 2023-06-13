import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...activations import gelu, gelu_new
from ...activations import silu as swish
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from .configuration_tvp import TVPConfig


TVP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Intel/tvp-base",
    # See all TVP models at https://huggingface.co/models?filter=tvp
]


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Conv2D. Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def tvp_loss(logits: torch.Tensor, labels: Tuple[torch.Tensor], alpha: float = 1.0, beta: float = 0.1) -> torch.Tensor:
    """
    Labels contain:
        1. duration: torch.Tensor, size = (batch_size, 1)
        2. start_time: torch.Tensor, size = (batch_size)
        3. end_time: torch.Tensor, size = (batch_size)
    """
    duration, start_time, end_time = labels
    candidates = torch.mul(logits, duration)
    candidates_start_time, candidates_end_time = candidates[:, 0].float(), candidates[:, 1].float()
    mid_c = torch.div(torch.add(candidates_start_time, candidates_end_time), 2.0)
    mid_g = torch.div(torch.add(start_time, end_time), 2.0)

    inter = torch.min(candidates_end_time, end_time) - torch.max(candidates_start_time, start_time)
    union = torch.max(candidates_end_time, end_time) - torch.min(candidates_start_time, start_time)
    iou = inter.clamp(min=0) / union

    duration_es = torch.sub(candidates_end_time, candidates_start_time)
    duration_gt = torch.sub(end_time, start_time)

    iou_loss = 1 - iou

    d_c = torch.div(torch.max(mid_g, mid_c) - torch.min(mid_g, mid_c), duration)
    d_c = d_c.clamp(min=0.2)

    duration_diff = torch.square(torch.div(torch.sub(duration_es, duration_gt), duration))
    duration_diff = duration_diff.clamp(min=0.4)

    loss = iou_loss + alpha * d_c + beta * duration_diff

    return loss


@dataclass
class TVPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Temporal-Distance IoU loss for video grounding.
        logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Contains start_time/duration and end_time/duration. It is the time slot of the videos corresponding to the
            input texts.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class TVPVisionConv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TVPVisionFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called "weight" and "bias", "running_mean", "running_var", initialized to perform
    identity transformation.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            eps=self.eps,
        )

    def __repr__(self):
        return "TVPVisionFrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)


class TVPVisionBasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = TVPVisionConv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=TVPVisionFrozenBatchNorm2d(out_channels),
        )
        c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


class TVPVisionBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        return self


class TVPVisionBasicBlock(TVPVisionBlockBase):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = TVPVisionConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=TVPVisionFrozenBatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = TVPVisionConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=TVPVisionFrozenBatchNorm2d(out_channels),
        )

        self.conv2 = TVPVisionConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=TVPVisionFrozenBatchNorm2d(out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class TVPVisionBottleneckBlock(TVPVisionBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        dilation=1,
    ):
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = TVPVisionConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=TVPVisionFrozenBatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = TVPVisionConv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
            norm=TVPVisionFrozenBatchNorm2d(bottleneck_channels),
        )

        self.conv2 = TVPVisionConv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=TVPVisionFrozenBatchNorm2d(bottleneck_channels),
        )

        self.conv3 = TVPVisionConv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=TVPVisionFrozenBatchNorm2d(out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class TVPVisionBackbone(nn.Module):
    def __init__(self, config):
        super(TVPVisionBackbone, self).__init__()
        self.config = config
        self.stem = TVPVisionBasicStem(
            in_channels=config.resnets_stem_input_channels,
            out_channels=config.resnets_stem_out_channels,
        )
        if config.backbone_freeze_at >= 1:
            for p in self.stem.parameters():
                p.requires_grad = False

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        stages = self.build_backbone(config)

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, TVPVisionBlockBase), block
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        out_features = config.features

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        x = self.stem(x)
        for stage, name in self.stages_and_names:
            x = stage(x)

        return x

    def make_stage(self, block_class, num_blocks, first_stride, **kwargs):
        """
        Create a TVPVisionBackbone stage by creating many blocks.

        Args:
            block_class (class): A subclass of TVPVisionBlockBase.
            num_blocks (int): The number of blocks.
            first_stride (int): The stride of the first block. The other blocks will have stride=1.
                A `stride` argument will be passed to the block constructor.
            kwargs: Other arguments passed to the block constructor.

        Returns:
            list[nn.Module]: a list of block module.
        """
        blocks = []
        for i in range(num_blocks):
            blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
            kwargs["in_channels"] = kwargs["out_channels"]

        return blocks

    def build_backbone(self, config):
        """
        Return stages of TVPVisionBackbone
        """
        out_features = config.features
        depth = config.resnets_depth
        num_groups = config.resnets_num_groups
        width_per_group = config.resnets_width_per_group
        bottleneck_channels = num_groups * width_per_group
        in_channels = config.resnets_stem_out_channels
        out_channels = config.resnets_res_out_channels
        res5_dilation = config.resnets_res_dilation

        if depth not in [18, 34, 50, 101, 152]:
            raise ValueError("The depth should be in [18, 34, 50, 101, 152]")
        if res5_dilation not in [1, 2]:
            raise ValueError("The res5_dilation should be in [1, 2]")

        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]

        stages = []

        # Avoid creating variables without gradients
        # It consumes extra memory and may cause allreduce to fail
        out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
        if len(out_stage_idx) == 0:
            raise ValueError("The element in out_features should be in [res2, res3, res4, res5]")
        max_stage_idx = max(out_stage_idx)
        for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
            dilation = res5_dilation if stage_idx == 5 else 1
            first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
            stage_kargs = {
                "num_blocks": num_blocks_per_stage[idx],
                "first_stride": first_stride,
                "in_channels": in_channels,
                "out_channels": out_channels,
            }
            # Use TVPVisionBasicBlock for R18 and R34.
            if depth in [18, 34]:
                stage_kargs["block_class"] = TVPVisionBasicBlock
            else:
                stage_kargs["bottleneck_channels"] = bottleneck_channels
                stage_kargs["dilation"] = dilation
                stage_kargs["num_groups"] = num_groups
                stage_kargs["block_class"] = TVPVisionBottleneckBlock
            blocks = self.make_stage(**stage_kargs)
            in_channels = out_channels
            out_channels *= 2
            bottleneck_channels *= 2

            if config.backbone_freeze_at >= stage_idx:
                for block in blocks:
                    block.freeze()
            stages.append(blocks)

        return stages


class TVPVisionModel(nn.Module):
    def __init__(self, config):
        super(TVPVisionModel, self).__init__()
        self.in_features = config.features
        self.backbone = TVPVisionBackbone(config)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(
                config.grid_encoder_conv_input_size,
                config.grid_encoder_conv_output_size,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.input_format = config.input_format

        self.config = config

    def forward(self, pixel_values):
        bsz, n_frms, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(bsz * n_frms, c, h, w)
        if self.input_format == "BGR":
            # RGB->BGR, images are read in as RGB by default
            pixel_values = pixel_values[:, [2, 1, 0], :, :]
        grid_feat_outputs = self.backbone(pixel_values)
        grid = self.grid_encoder(grid_feat_outputs)  # (B * n_frm, C, H, W)
        new_c, new_h, new_w = grid.shape[-3:]
        grid = grid.view(bsz, n_frms, new_c, new_h, new_w)  # (B, n_frm, C, H, W)

        grid = grid.permute(0, 1, 3, 4, 2)  # (B, n_frm, H, W, C)
        return grid


class VisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """

    def __init__(self, config):
        super(VisualInputEmbedding, self).__init__()
        self.config = config

        # sequence embedding
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = nn.Embedding(config.max_grid_row_position_embeddings, config.hidden_size)
        self.col_position_embeddings = nn.Embedding(config.max_grid_col_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, grid):
        """
        Args:
            grid: (B, n_frm, H, W, C), note that #frm can be 1

        Returns:

        """
        bsz, _, _, _, hsz = grid.shape
        # temporal mean pooling
        grid = grid.mean(1)  # (B, H, W, d)
        grid = self.add_2d_positional_embeddings(grid)  # (B, H, W, d)
        # image token sequence
        visual_tokens = grid.view(bsz, -1, hsz)  # (B, H*W, d)
        visual_tokens_shape = visual_tokens.shape[:-1]  # (B, H*W)
        device = visual_tokens.device

        # image token type embeddings.
        token_type_ids = torch.zeros(visual_tokens_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (B, H*W, d)

    def add_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (B, H, W, d)

        Returns:
            grid + col_position_embeddings.view(*col_shape): (B, *, H, W, d)
        """
        height, width, hsz = grid.shape[-3:]

        # add row-wise position embeddings
        row_position_ids = torch.arange(height, dtype=torch.long, device=grid.device)  # (H, )
        row_position_embeddings = self.row_position_embeddings(row_position_ids)  # (H, d)
        row_shape = (1,) * (len(grid.shape) - 3) + (height, 1, hsz)  # (1, H, 1, d)
        grid = grid + row_position_embeddings.view(*row_shape)  # broadcast automatically

        # add column-wise position embeddings
        col_position_ids = torch.arange(width, dtype=torch.long, device=grid.device)  # (W, )
        col_position_embeddings = self.col_position_embeddings(col_position_ids)  # (W, d)
        col_shape = (1,) * (len(grid.shape) - 3) + (1, width, hsz)  # (1, 1, W, d)
        return grid + col_position_embeddings.view(*col_shape)  # broadcast automatically


class TextInputEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TVPSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: Optional[bool] = None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class TVPSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TVPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = TVPSelfAttention(config)
        self.output = TVPSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: Optional[bool] = None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TVPIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TVPOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TVPEncodeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TVPAttention(config)
        self.intermediate = TVPIntermediate(config)
        self.output = TVPOutputLayer(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: Optional[bool] = None,
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class TVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TVPEncodeLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs  # last-layer hidden state, (all hidden states), (all attentions)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_attentions if output_attentions else None,
        )


class TVPPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TVPPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = TVPConfig
    base_model_prefix = "tvp"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class TVPTransformerBase(TVPPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = TextInputEmbeddings(config)
        self.visual_embeddings = VisualInputEmbedding(config)
        self.encoder = TVPEncoder(config)
        self.pooler = TVPPooler(config)
        self.text_prompt = nn.Parameter(torch.randn([1, 10, 768]))
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        input_ids: (B, Lt) pixel_values: (B, #frame, H, W, C) attention_mask: (B, Lt) with 1 indicates valid, 0
        indicates invalid position.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        input_shape = input_ids.size()
        device = input_ids.device
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        text_embedding_output = self.embeddings(input_ids=input_ids)  # (B, Lt, D)
        visual_embedding_output = self.visual_embeddings(pixel_values)  # (B, Lv, d)
        visual_attention_mask = attention_mask.new_ones(visual_embedding_output.shape[:2])  # (B, Lv)
        pt_mask = torch.ones(attention_mask.shape[0], 10)
        attention_mask = torch.cat([pt_mask.long(), attention_mask, visual_attention_mask], dim=-1)  # (B, lt+Lv, d)
        txt_pt = self.text_prompt
        txt_pt = txt_pt.expand(text_embedding_output.shape[0], -1, -1)
        embedding_output = torch.cat([txt_pt, text_embedding_output, visual_embedding_output], dim=1)  # (B, Lt+Lv, d)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=self.get_head_mask(head_mask, self.config.num_hidden_layers),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        if not return_dict:
            return (
                last_hidden_state,
                pooled_output,
            ) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TVPTransformer(TVPPreTrainedModel):
    def __init__(self, config):
        super(TVPTransformer, self).__init__(config)
        self.config = config

        self.base = TVPTransformerBase(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, 2),
            nn.Sigmoid(),
        )
        self.init_weights()

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        labels=None,
        sample_size=-1,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.base(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,  # (B, Lt) note this mask is text only!!!
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # rank (B, 1) or ce (B, 2)
        if labels is None:
            loss = None
        else:
            loss = tvp_loss(logits, labels, self.config.alpha, self.config.beta)

        if not return_dict:
            outputs = (logits,) + outputs[2:]
            if loss:
                outputs = (loss,) + outputs
            return outputs

        return TVPOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class PadPrompter(nn.Module):
    def __init__(self, image_size, prompt_size):
        super(PadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size

        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 1, 3, image_size - pad_size * 2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 1, 3, image_size - pad_size * 2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 1, 3, self.base_size, self.base_size)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=4)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=3)
        prompt = torch.cat(x.size(1) * [prompt], dim=1)
        prompt = torch.cat(x.size(0) * [prompt])
        return prompt


class DownPadPrompter(nn.Module):
    def __init__(self, image_size, pad_size):
        super(DownPadPrompter, self).__init__()
        self.pad_size = pad_size
        self.image_size = image_size

        self.pad_down = nn.Parameter(torch.randn([1, 1, 3, pad_size, image_size]))

    def forward(self, x):
        prompt = torch.zeros([x.shape[0], x.shape[1], 3, self.image_size, self.image_size])
        start_point = self.image_size - self.pad_size
        prompt[:, :, :, start_point : self.image_size, :] = self.pad_down
        return prompt


class FrameDownPadPrompter(nn.Module):
    def __init__(self, image_size, pad_size, frame_num):
        super(FrameDownPadPrompter, self).__init__()
        self.pad_size = pad_size
        self.image_size = image_size
        self.frame_num = frame_num

        self.pad_down = nn.Parameter(torch.randn([1, frame_num, 3, pad_size, image_size]))

    def forward(self, x):
        prompt = torch.zeros([x.shape[0], x.shape[1], 3, self.image_size, self.image_size])
        start_point = self.image_size - self.pad_size
        prompt[:, :, :, start_point : self.image_size, :] = self.pad_down
        return prompt


class FramePadPrompter(nn.Module):
    def __init__(self, image_size, prompt_size, frame_num):
        super(FramePadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size
        self.frame_num = frame_num

        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, frame_num, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, frame_num, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, frame_num, 3, image_size - pad_size * 2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, frame_num, 3, image_size - pad_size * 2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, self.frame_num, 3, self.base_size, self.base_size)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=4)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=3)
        prompt = torch.cat(x.size(0) * [prompt])
        return prompt


class TVPModel(TVPPreTrainedModel):
    def __init__(self, config):
        super(TVPModel, self).__init__(config)
        self.config = config
        self.cnn = TVPVisionModel(config.vision_config)
        self.transformer = TVPTransformer(config)
        if config.vp_type == "downpad":
            self.tp = DownPadPrompter(config.max_img_size, config.pad_size)
        elif config.vp_type == "pad":
            self.tp = PadPrompter(config.max_img_size, config.pad_size)
        elif config.vp_type == "framedownpad":
            self.tp = FrameDownPadPrompter(config.max_img_size, config.pad_size, config.num_frm)
        elif config.vp_type == "framepad":
            self.tp = FramePadPrompter(config.max_img_size, config.pad_size, config.num_frm)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        labels=None,
        sample_size=-1,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        pixel_values = self.add_vision_prompt(pixel_values)
        pixel_values = self.cnn(pixel_values)
        outputs = self.transformer(
            input_ids,
            pixel_values,
            attention_mask,
            labels=labels,
            sample_size=sample_size,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False

    def add_vision_prompt(self, pixel_values):
        if self.config.vp_apply == "remove":
            pass
        else:
            vp = self.tp(pixel_values).to(pixel_values.dtype)

        if self.config.vp_apply == "add":
            pixel_values = pixel_values + vp
        elif self.config.vp_apply == "remove":
            tp_mask = torch.ones([self.config.max_img_size, self.config.max_img_size], dtype=pixel_values.dtype)
            start_point = self.config.pad_size
            end_point = self.config.max_img_size - self.config.pad_size

            if self.config.vp_type == "downpad" or self.config.vp_type == "framedownpad":
                tp_mask[end_point : self.config.max_img_size, :] = 0.0
            elif self.config.vp_type == "pad":
                tp_mask[end_point : self.config.max_img_size, :] = 0.0
                tp_mask[:start_point, :] = 0.0
                tp_mask[:, end_point : self.config.max_img_size] = 0.0
                tp_mask[:, :start_point] = 0.0

            pixel_values = pixel_values * tp_mask
        elif self.config.vp_apply == "replace":
            tp_mask = torch.ones([self.config.max_img_size, self.config.max_img_size], dtype=pixel_values.dtype)
            start_point = self.config.pad_size
            end_point = self.config.max_img_size - self.config.pad_size

            if self.config.vp_type == "downpad" or self.config.vp_type == "framedownpad":
                tp_mask[end_point : self.config.max_img_size, :] = 0.0
            elif self.config.vp_type == "pad":
                tp_mask[end_point : self.config.max_img_size, :] = 0.0
                tp_mask[:start_point, :] = 0.0
                tp_mask[:, end_point : self.config.max_img_size] = 0.0
                tp_mask[:, :start_point] = 0.0

            pixel_values = pixel_values * tp_mask + vp

        return pixel_values
