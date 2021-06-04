# coding=utf-8
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

import detectron2
from detectron2.modeling import META_ARCH_REGISTRY
from .detectron2_config import add_layoutlmv2_config
from transformers import PreTrainedModel
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMIntermediate as LayoutLMv2Intermediate
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMOutput as LayoutLMv2Output
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMPooler as LayoutLMv2Pooler
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMSelfOutput as LayoutLMv2SelfOutput
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMSelfOutput as LayoutLMv2OnlyMLMHead
from transformers.utils import logging

from .configuration_layoutlmv2 import LayoutLMv2Config


logger = logging.get_logger(__name__)

LAYOUTLMv2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutlmv2-base-uncased",
    "layoutlmv2-large-uncased",
]


LayoutLMv2LayerNorm = torch.nn.LayerNorm


class LayoutLMv2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(LayoutLMv2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayoutLMv2LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings


class LayoutLMv2SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if config.fast_qkv:
            self.qkv_linear = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        # [BSZ, NAT, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + rel_pos + rel_2d_pos
        attention_scores = attention_scores.float().masked_fill_(attention_mask.to(torch.bool), float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LayoutLMv2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv2SelfAttention(config)
        self.output = LayoutLMv2SelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

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
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class LayoutLMv2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv2Attention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class LayoutLMv2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)])

        self.rel_pos_bins = config.rel_pos_bins
        self.max_rel_pos = config.max_rel_pos
        self.rel_pos_onehot_size = config.rel_pos_bins
        self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        self.max_rel_2d_pos = config.max_rel_2d_pos
        self.rel_2d_pos_bins = config.rel_2d_pos_bins
        self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
        self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
        self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bbox=None,
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids)
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class LayoutLMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv2Config
    pretrained_model_archive_map = LAYOUTLMv2_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlmv2"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayoutLMv2LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def my_convert_sync_batchnorm(module, process_group=None):
    # same as `nn.modules.SyncBatchNorm.convert_sync_batchnorm` but allowing converting from `detectron2.layers.FrozenBatchNorm2d`
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return nn.modules.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
    module_output = module
    if isinstance(module, detectron2.layers.FrozenBatchNorm2d):
        module_output = torch.nn.SyncBatchNorm(
            num_features=module.num_features,
            eps=module.eps,
            affine=True,
            track_running_stats=True,
            process_group=process_group,
        )
        module_output.weight = torch.nn.Parameter(module.weight)
        module_output.bias = torch.nn.Parameter(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=module.running_mean.device)
    for name, child in module.named_children():
        module_output.add_module(name, my_convert_sync_batchnorm(child, process_group))
    del module
    return module_output


class VisualBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = detectron2.config.get_cfg()
        add_layoutlmv2_config(self.cfg)
        meta_arch = self.cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg)
        assert isinstance(model.backbone, detectron2.modeling.backbone.FPN)
        self.backbone = model.backbone
        if (
            config.convert_sync_batchnorm
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            self_rank = torch.distributed.get_rank()
            node_size = torch.cuda.device_count()
            world_size = torch.distributed.get_world_size()
            assert world_size % node_size == 0

            node_global_ranks = [
                list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)
            ]
            sync_bn_groups = [
                torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)
            ]
            node_rank = self_rank // node_size
            assert self_rank in node_global_ranks[node_rank]

            self.backbone = my_convert_sync_batchnorm(self.backbone, process_group=sync_bn_groups[node_rank])

        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1),
        )
        self.register_buffer("pixel_std", torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1))
        self.out_feature_key = "p2"
        if torch.is_deterministic():
            logger.warning("using `AvgPool2d` instead of `AdaptiveAvgPool2d`")
            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            self.pool = nn.AvgPool2d(
                (
                    math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]),
                    math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1]),
                )
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    def forward(self, images):
        images_input = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features


class LayoutLMv2Model(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super(LayoutLMv2Model, self).__init__(config)
        self.config = config

        self.embeddings = LayoutLMv2Embeddings(config)

        self.visual = VisualBackbone(config)
        self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = LayoutLMv2Encoder(config)
        self.pooler = LayoutLMv2Pooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        visual_embeddings = self.visual_proj(self.visual(image))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._cal_spatial_position_embeddings(bbox)
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox_x = (
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            )
            // self.config.image_feature_pool_shape[1]
        )
        visual_bbox_y = (
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            )
            // self.config.image_feature_pool_shape[0]
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(self.config.image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(self.config.image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(self.config.image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(self.config.image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))
        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        visual_attention_mask = torch.ones(visual_shape, device=device)
        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand_as(input_ids)

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=input_ids.device).repeat(
            input_shape[0], 1
        )
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        Returns:
        Examples::
            >>> from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
            >>> import torch
            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            >>> token_labels = torch.tensor([1,1,0,0]).unsqueeze(0) # batch size of 1
            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=token_labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq_length = input_ids.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LayoutLMv2ForMaskedLM(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.cls = LayoutLMv2OnlyMLMHead(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import LayoutLMv2Tokenizer, LayoutLMv2ForMaskedLM
            >>> import torch

            >>> tokenizer = LayoutLMv2Tokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMv2ForMaskedLM.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "[MASK]"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])

            >>> labels = tokenizer("Hello world", return_tensors="pt")["input_ids"]

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=labels)

            >>> loss = outputs.loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LayoutLMv2ForSequenceClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings
        
    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
        Examples::
            >>> from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification
            >>> import torch
            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMForSequenceClassification.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            >>> sequence_label = torch.tensor([1])
            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=sequence_label)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )