import copy

from torch import nn
import torch
from typing import List, Optional, Tuple, Union
import math
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    LightweightConv,
    DynamicConv,
    MultiheadAttention,
)

import transformers.models.lite_transformer.utils as utils
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from .configuration_lite_transformer import LiteTransformerConfig


class MultiBranch(nn.Module):
    def __init__(self, branches, embed_dim_list):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.embed_dim_list = embed_dim_list

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True,
                static_kv=False, attn_mask=None):
        tgt_len, bsz, embed_size = query.size()
        assert sum(self.embed_dim_list) == embed_size
        out = []
        attn = None
        start = 0
        for idx, embed_dim in enumerate(self.embed_dim_list):
            branch = self.branches[idx]
            branch_type = type(branch)

            q = query[..., start:start + embed_dim]
            if key is not None:
                assert value is not None
                k, v = key[..., start:start + embed_dim], value[..., start:start + embed_dim]
            start += embed_dim

            if branch_type == MultiheadAttention:
                x, attn = branch(q, k, v, key_padding_mask, incremental_state, need_weights, static_kv, attn_mask)
            else:
                mask = key_padding_mask
                if mask is not None:
                    q = q.masked_fill(mask.transpose(0, 1).unsqueeze(2), 0)
                x = branch(q.contiguous(), incremental_state=incremental_state)
            out.append(x)

        out = torch.cat(out, dim=-1)
        return out, attn


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        nn.init(self.weight, mean=0, std=embedding_dim ** 0.5)
        if padding_idx is not None:
            nn.init.constant_(self.weight[padding_idx], 0)

    def forward(self, input, incremental_state=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (
                (positions is None) or (self.padding_idx is None)
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = input.data.new(1, 1).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = self.make_positions(
                    input.data, self.padding_idx, onnx_trace=self.onnx_trace,
                )
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        else:
            return self.num_embeddings

    def make_positions(self, tensor, padding_idx, onnx_trace=False):
        """Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()
        return (
                       torch.cumsum(mask, dim=1).type_as(mask) * mask
               ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(bsz, 1, 1)
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = self.make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(flat_embeddings, embedding_shape)
            return embeddings
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

    def make_positions(self, tensor, padding_idx, onnx_trace=False):
        """Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()
        return (
                       torch.cumsum(mask, dim=1).type_as(mask) * mask
               ).long() + padding_idx


class LiteTransformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, vocab_size, hidden_size, pad_token_id, learned_position_embedding):
        super().__init__()
        num_embeddings = vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        if learned_position_embedding:
            if pad_token_id is not None:
                num_embeddings = vocab_size + pad_token_id + 1
            self.position_embeddings = LearnedPositionalEmbedding(num_embeddings, config.hidden_size,
                                                                  padding_idx=config.pad_token_id)
        else:
            self.position_embeddings = SinusoidalPositionalEmbedding(num_embeddings, padding_idx=pad_token_id,
                                                                     init_size=num_embeddings + pad_token_id + 1)

    def forward(self, input_ids: Optional[torch.LongTensor] = None):
        return self.word_embeddings(input_ids) + self.position_embeddings(input_ids)


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, config, index):
        super().__init__()
        self.embed_dim = config.encoder_hidden_size

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(config, 'activation_fn', 'relu')
        )

        if config.encoder_branch_type is None:
            self.self_attn = MultiheadAttention(
                self.embed_dim, config.encoder_attention_heads,
                dropout=config.attention_dropout, self_attention=True,
            )
        else:
            layers = []
            embed_dims = []
            heads = []
            for layer_type in config.encoder_branch_type:
                embed_dims.append(int(layer_type.split(':')[2]))
                heads.append(int(layer_type.split(':')[3]))
                layers.append(self.get_layer(config, index, embed_dims[-1], heads[-1], layer_type))
            assert sum(embed_dims) == self.embed_dim, (sum(embed_dims), self.embed_dim)

            self.self_attn = MultiBranch(layers, embed_dims)

        self.activation_dropout = getattr(config, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(config, 'relu_dropout', 0)

        self.normalize_before = config.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, config.encoder_ffn_embed_dim, init=config.ffn_init)
        self.fc2 = Linear(config.encoder_ffn_embed_dim, self.embed_dim, init=config.ffn_init)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def get_layer(self, config, index, out_dim, num_heads, layer_type):
        kernel_size = layer_type.split(':')[1]
        if kernel_size == 'default':
            kernel_size = config.encoder_kernel_size_list[index]
        else:
            kernel_size = int(kernel_size)
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        if 'lightweight' in layer_type:
            layer = LightweightConv(
                out_dim, kernel_size, padding_l=padding_l, weight_softmax=config.weight_softmax,
                num_heads=num_heads, weight_dropout=config.weight_dropout,
                # with_linear=config.conv_linear,
            )
        elif 'dynamic' in layer_type:
            layer = nn.ModuleList([])
            layer.append(DynamicConv(
                out_dim, kernel_size, padding_l=padding_l,
                weight_softmax=config.weight_softmax, num_heads=num_heads,
                weight_dropout=config.weight_dropout
                # with_linear=args.conv_linear,
                # glu=args.encoder_glu,
            ))
            if config.encoder_glu:
                layer.append(nn.GLU())
            if config.encoder_conv_linear:
                layer.append(Linear(out_dim,out_dim))
        elif 'attn' in layer_type:
            layer = MultiheadAttention(
                out_dim, num_heads,
                dropout=config.attention_dropout, self_attention=True,
            )
        else:
            raise NotImplementedError

        return layer

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.byte(), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, config, index, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = config.decoder_hidden_size
        kernel_size = config.decoder_kernel_size_list[index]

        self.dropout = config.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=config.activation_fn
        )
        self.activation_dropout = config.activation_dropout
        # if self.activation_dropout == 0:
        #     # for backwards compatibility with models that use args.relu_dropout
        #     self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = config.decoder_normalize_before

        if config.decoder_branch_type is None:
            self.self_attn = MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                dropout=config.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
            )
        else:
            layers = []
            embed_dims = []
            heads = []
            for layer_type in config.decoder_branch_type:
                embed_dims.append(int(layer_type.split(':')[2]))
                heads.append(int(layer_type.split(':')[3]))
                layers.append(
                    self.get_layer(config, index, embed_dims[-1], heads[-1], layer_type, add_bias_kv, add_zero_attn))
            assert sum(embed_dims) == self.embed_dim, (sum(embed_dims), self.embed_dim)

            self.self_attn = MultiBranch(layers, embed_dims)

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        # export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn = MultiheadAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            kdim=config.encoder_hidden_size,
            vdim=config.encoder_hidden_size,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, config.decoder_ffn_embed_dim, init=config.ffn_init)
        self.fc2 = Linear(config.decoder_ffn_embed_dim, self.embed_dim, init=config.ffn_init)
        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            incremental_state=None,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def get_layer(self, args, index, out_dim, num_heads, layer_type, add_bias_kv, add_zero_attn):
        kernel_size = layer_type.split(':')[1]
        if kernel_size == 'default':
            kernel_size = args.decoder_kernel_size_list[index]
        else:
            kernel_size = int(kernel_size)
        layer_type = layer_type.split(':')[0]
        if layer_type == 'lightweight':
            layer = LightweightConv(
                out_dim, kernel_size, padding_l=kernel_size-1,
                weight_softmax=args.weight_softmax, num_heads=num_heads,
                weight_dropout=args.weight_dropout, with_linear=args.conv_linear,
            )
        elif layer_type == 'dynamic':
            layer = DynamicConv(
                out_dim, kernel_size, padding_l=kernel_size-1,
                weight_softmax=args.weight_softmax, num_heads=num_heads,
                weight_dropout=args.weight_dropout
            )
        elif layer_type == 'attn':
            layer = MultiheadAttention(
                embed_dim=out_dim,
                num_heads=num_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
            )
        else:
            raise NotImplementedError
        return layer

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class LiteTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LiteTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LiteTransformerEncoder, LiteTransformerDecoder):
            module.gradient_checkpointing = value


class LiteTransformerEncoder(LiteTransformerPreTrainedModel):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, config, embeddings):
        super().__init__(config)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = config.dropout

        embed_dim = config.encoder_hidden_size

        self.padding_idx = config.encoder_pad_token_id
        self.max_source_positions = config.max_source_position_embeddings

        self.embed_tokens = embeddings
        self.embed_scale = math.sqrt(embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(config, i)
            for i in range(config.encoder_layers)
        ])

        if config.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.none = None
            self.layer_norm = self.none

    def forward(self, src_tokens):
        """
        Args:
            src_tokens (LongTensor): tokens in the ource language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x = self.input_transform(x) if self.input_transform is not None else x
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class LiteTransformerDecoder(LiteTransformerPreTrainedModel):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, config, embeddings):
        super().__init__(config)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = config.dropout
        self.share_input_output_embed = config.share_decoder_input_output_embed

        embed_dim = config.decoder_hidden_size
        self.output_embed_dim = embed_dim

        padding_idx = config.decoder_pad_token_id
        self.max_target_positions = config.max_target_position_embeddings

        self.embeddings = embeddings
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.embed_positions = PositionalEmbedding(
            config.max_target_position_embeddings, embed_dim, padding_idx,
            learned=config.decoder_learned_position_embedding,
        ) if not config.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(config, i)
            for i in range(config.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not config.tie_adaptive_weights else None

        if config.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                config.decoder_vocabdhsize,
                self.output_embed_dim,
                utils.eval_str_list(config.adaptive_softmax_cutoff, type=int),
                dropout=config.adaptive_softmax_dropout,
                adaptive_inputs=embeddings if config.tie_adaptive_weights else None,
                factor=config.adaptive_softmax_factor,
                tie_proj=config.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(config.vocab_size, self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if config.decoder_normalize_before and not config.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens)
        x = self.input_transform(x) if self.input_transform is not None else x
        x = self.embed_scale * x

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(
            0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


class LiteTransformerModel(LiteTransformerPreTrainedModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, config: LiteTransformerConfig):
        super().__init__(config)

        self.shared = LiteTransformerEmbeddings(config, config.encoder_vocab_size, config.encoder_hidden_size,
                                                config.encoder_pad_token_id, config.encoder_learned_position_embedding)
        if self.config.share_all_embeddings:
            encoder_embeddings = decoder_embeddings = self.shared
        else:
            # Since the embeddings are not shared, deepcopy the embeddings here for encoder
            # and decoder to make sure they are not tied.
            encoder_embeddings = self.shared
            decoder_embeddings = LiteTransformerEmbeddings(config, config.decoder_vocab_size,
                                                           config.decoder_hidden_size, config.decoder_pad_token_id,
                                                           config.decoder_learned_position_embedding)
            self.shared = None

        self.encoder = LiteTransformerEncoder(config, encoder_embeddings)
        self.decoder = LiteTransformerDecoder(config, decoder_embeddings)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


def Linear(in_features, out_features, bias=True, init=None):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight) if init is None else init(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
