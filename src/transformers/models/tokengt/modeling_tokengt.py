# coding=utf-8
import datetime
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    ModelOutput,
)
from .fairseq_utils import quant_noise, LayerDropModuleList
from .configuration_tokengt import TokenGTConfig
from .performer import FastAttention, gaussian_orthogonal_random_matrix_batched, ProjectionUpdater


logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "TokenGTConfig"
_TOKENIZER_FOR_DOC = "AutoTokenizer"


def init_params(module, num_layers):
    # TODO: All init must go in the PretrainedModel class 
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(num_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        #data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))
        data.normal_(mean=0.0, std=0.02)

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

class GraphFeatureTokenizer(nn.Module):
    """
    Compute node and edge features for each node and edge in the graph.
    """

    def __init__(self, config):
        super(GraphFeatureTokenizer, self).__init__()

        self.embedding_dim = config.embedding_dim

        self.atom_encoder = nn.Embedding(config.num_atoms, config.embedding_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(config.num_edges, config.embedding_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, config.embedding_dim)
        self.null_token = nn.Embedding(1, config.embedding_dim)  # this is optional

        self.rand_node_id = config.rand_node_id
        self.rand_node_id_dim = config.rand_node_id_dim
        self.orf_node_id = config.orf_node_id
        self.orf_node_id_dim = config.orf_node_id_dim
        self.lap_node_id = config.lap_node_id
        self.lap_node_id_k = config.lap_node_id_k
        self.lap_node_id_sign_flip = config.lap_node_id_sign_flip

        self.type_id = config.type_id

        if self.rand_node_id:
            self.rand_encoder = nn.Linear(2 * config.rand_node_id_dim, config.embedding_dim, bias=False)

        if self.lap_node_id:
            self.lap_encoder = nn.Linear(2 * config.lap_node_id_k, config.embedding_dim, bias=False)
            self.lap_eig_dropout = nn.Dropout2d(p=config.lap_node_id_eig_dropout) if config.lap_node_id_eig_dropout > 0 else None

        if self.orf_node_id:
            self.orf_encoder = nn.Linear(2 * config.orf_node_id_dim, config.embedding_dim, bias=False)

        if self.type_id:
            self.order_encoder = nn.Embedding(2, config.embedding_dim)

        self.apply(lambda module: init_params(module, num_layers=config.num_layers))

    @staticmethod
    def get_batch(node_feature, edge_index, edge_feature, node_num, edge_num, perturb=None):
        """
        :param node_feature: Tensor([sum(node_num), D])
        :param edge_index: LongTensor([2, sum(edge_num)])
        :param edge_feature: Tensor([sum(edge_num), D])
        :param node_num: list
        :param edge_num: list
        :param perturb: Tensor([batch_size, max(node_num), D])
        # TOOD: update comments about dimensions with more explicit names
        :return: padded_index: LongTensor([batch_size, T, 2]), padded_feature: Tensor([batch_size, T, D]), padding_mask: BoolTensor([batch_size, T])
        """
        seq_len = [n + e for n, e in zip(node_num, edge_num)] # all seq lengths
        batch_size = len(seq_len)
        d = node_feature.size(-1) # embedding dimension of node features
        max_len = max(seq_len)
        max_n = max(node_num)
        device = edge_index.device

        token_pos = torch.arange(max_len, device=device)[None, :].expand(batch_size, max_len)  # [batch_size, T]

        seq_len = torch.tensor(seq_len, dtype=torch.long, device=device)[:, None]  # [batch_size, 1]
        node_num = torch.tensor(node_num, dtype=torch.long, device=device)[:, None]  # [batch_size, 1]
        edge_num = torch.tensor(edge_num, dtype=torch.long, device=device)[:, None]  # [batch_size, 1]

        node_index = torch.arange(max_n, dtype=torch.long, device=device)[None, :].expand(batch_size, max_n)  # [batch_size, max_n]
        node_index = node_index[None, node_index < node_num].repeat(2, 1)  # [2, sum(node_num)]

        padded_node_mask = torch.less(token_pos, node_num).to(device)
        padded_edge_mask = torch.logical_and(
            torch.greater_equal(token_pos, node_num),
            torch.less(token_pos, node_num + edge_num)
        ).to(device)

        padded_index = torch.zeros(batch_size, max_len, 2, dtype=torch.long, device=device)  # [batch_size, T, 2]
        padded_index[padded_node_mask, :] = node_index.t()
        padded_index[padded_edge_mask, :] = edge_index.t()

        if perturb is not None:
            perturb_mask = padded_node_mask[:, :max_n]  # [batch_size, max_n]
            node_feature = node_feature + perturb[perturb_mask].type(node_feature.dtype)  # [sum(node_num), D]

        padded_feature = torch.zeros(batch_size, max_len, d, dtype=node_feature.dtype, device=device)  # [batch_size, T, D]
        padded_feature[padded_node_mask, :] = node_feature
        padded_feature[padded_edge_mask, :] = edge_feature

        padding_mask = torch.greater_equal(token_pos, seq_len)  # [batch_size, T]
        return padded_index, padded_feature, padding_mask, padded_node_mask, padded_edge_mask

    @staticmethod
    @torch.no_grad()
    def get_node_mask(node_num, device):
        batch_size = len(node_num)
        max_n = max(node_num)
        node_index = torch.arange(max_n, dtype=torch.long, device=device)[None, :].expand(batch_size, max_n)  # [batch_size, max_n]
        node_num = torch.tensor(node_num, dtype=torch.long, device=device)[:, None]  # [batch_size, 1]
        node_mask = torch.less(node_index, node_num)  # [batch_size, max_n]
        return node_mask

    @staticmethod
    @torch.no_grad()
    def get_random_sign_flip(eigvec, node_mask):
        batch_size, max_n = node_mask.size()
        d = eigvec.size(1)

        sign_flip = torch.rand(batch_size, d, device=eigvec.device, dtype=eigvec.dtype)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        sign_flip = sign_flip[:, None, :].expand(batch_size, max_n, d)
        sign_flip = sign_flip[node_mask]
        return sign_flip

    def handle_eigvec(self, eigvec, node_mask, sign_flip):
        if sign_flip and self.training:
            sign_flip = self.get_random_sign_flip(eigvec, node_mask)
            eigvec = eigvec * sign_flip
        else:
            pass
        return eigvec

    @staticmethod
    @torch.no_grad()
    def get_orf_batched(node_mask, dim, device, dtype):
        batch_size, max_n = node_mask.size(0), node_mask.size(1)
        orf = gaussian_orthogonal_random_matrix_batched(batch_size, dim, dim, dtype=dtype, device=device)  # [batch_size, D, D]
        orf = orf[:, None, ...].expand(batch_size, max_n, dim, dim)  # [batch_size, max(n_node), D, D]
        orf = orf[node_mask]  # [sum(n_node), D, D]
        return orf

    @staticmethod
    def get_index_embed(node_id, node_mask, padded_index):
        """
        :param node_id: Tensor([sum(node_num), D])
        :param node_mask: BoolTensor([batch_size, max_n])
        :param padded_index: LongTensor([batch_size, T, 2])
        :return: Tensor([batch_size, T, 2D])
        """
        batch_size, max_n = node_mask.size()
        max_len = padded_index.size(1)
        d = node_id.size(-1)

        padded_node_id = torch.zeros(batch_size, max_n, d, device=node_id.device, dtype=node_id.dtype)  # [batch_size, max_n, D]
        padded_node_id[node_mask] = node_id

        padded_node_id = padded_node_id[:, :, None, :].expand(batch_size, max_n, 2, d)
        padded_index = padded_index[..., None].expand(batch_size, max_len, 2, d)
        index_embed = padded_node_id.gather(1, padded_index)  # [batch_size, T, 2, D]
        index_embed = index_embed.view(batch_size, max_len, 2 * d)
        return index_embed

    def get_type_embed(self, padded_index):
        """
        :param padded_index: LongTensor([batch_size, T, 2])
        :return: Tensor([batch_size, T, D])
        """
        order = torch.eq(padded_index[..., 0], padded_index[..., 1]).long()  # [batch_size, T]
        order_embed = self.order_encoder(order)
        return order_embed

    def add_special_tokens(self, padded_feature, padding_mask):
        """
        :param padded_feature: Tensor([batch_size, T, D])
        :param padding_mask: BoolTensor([batch_size, T])
        :return: padded_feature: Tensor([batch_size, 2/3 + T, D]), padding_mask: BoolTensor([batch_size, 2/3 + T])
        """
        batch_size, _, d = padded_feature.size()

        num_special_tokens = 2
        graph_token_feature = self.graph_token.weight.expand(batch_size, 1, d)  # [1, D]
        null_token_feature = self.null_token.weight.expand(batch_size, 1, d)  # [1, D], this is optional
        special_token_feature = torch.cat((graph_token_feature, null_token_feature), dim=1)  # [batch_size, 2, D]
        special_token_mask = torch.zeros(batch_size, num_special_tokens, dtype=torch.bool, device=padded_feature.device)

        padded_feature = torch.cat((special_token_feature, padded_feature), dim=1)  # [batch_size, 2 + T, D]
        padding_mask = torch.cat((special_token_mask, padding_mask), dim=1)  # [batch_size, 2 + T]
        return padded_feature, padding_mask

    def forward(self, batched_data, perturb=None):
        (
            node_data,
            in_degree,
            out_degree,
            node_num,
            lap_eigvec,
            lap_eigval,
            edge_index,
            edge_data,
            edge_num
        ) = (
            batched_data["node_data"],
            batched_data["in_degree"],
            batched_data["out_degree"],
            batched_data["node_num"],
            batched_data["lap_eigvec"],
            batched_data["lap_eigval"],
            batched_data["edge_index"],
            batched_data["edge_data"],
            batched_data["edge_num"]
        )

        node_feature = self.atom_encoder(node_data).sum(-2)  # [sum(n_node), D]
        edge_feature = self.edge_encoder(edge_data).sum(-2)  # [sum(n_edge), D]
        device = node_feature.device
        dtype = node_feature.dtype

        padded_index, padded_feature, padding_mask, _, _ = self.get_batch(
            node_feature, edge_index, edge_feature, node_num, edge_num, perturb
        )
        node_mask = self.get_node_mask(node_num, node_feature.device)  # [batch_size, max(n_node)]

        if self.rand_node_id:
            rand_node_id = torch.rand(sum(node_num), self.rand_node_id_dim, dtype=dtype)  # [sum(n_node), D]
            rand_node_id = F.normalize(rand_node_id, p=2, dim=1)
            rand_index_embed = self.get_index_embed(rand_node_id, node_mask, padded_index)  # [batch_size, T, 2D]
            padded_feature = padded_feature + self.rand_encoder(rand_index_embed)

        if self.orf_node_id:
            batch_size, max_n = len(node_num), max(node_num)
            orf = gaussian_orthogonal_random_matrix_batched(
                batch_size, max_n, max_n, dtype=dtype
            )  # [batch_size, max(n_node), max(n_node)]
            orf_node_id = orf[node_mask]  # [sum(n_node), max(n_node)]
            if self.orf_node_id_dim > max_n:
                orf_node_id = F.pad(orf_node_id, (0, self.orf_node_id_dim - max_n), value=float('0'))  # [sum(n_node), Do]
            else:
                orf_node_id = orf_node_id[..., :self.orf_node_id_dim]  # [sum(n_node), Do]
            orf_node_id = F.normalize(orf_node_id, p=2, dim=1)
            orf_index_embed = self.get_index_embed(orf_node_id, node_mask, padded_index)  # [batch_size, T, 2Do]
            padded_feature = padded_feature + self.orf_encoder(orf_index_embed)

        if self.lap_node_id:
            lap_dim = lap_eigvec.size(-1)
            if self.lap_node_id_k > lap_dim:
                eigvec = F.pad(lap_eigvec, (0, self.lap_node_id_k - lap_dim), value=float('0'))  # [sum(n_node), Dl]
            else:
                eigvec = lap_eigvec[:, :self.lap_node_id_k]  # [sum(n_node), Dl]
            if self.lap_eig_dropout is not None:
                eigvec = self.lap_eig_dropout(eigvec[..., None, None]).view(eigvec.size())
            lap_node_id = self.handle_eigvec(eigvec, node_mask, self.lap_node_id_sign_flip)
            lap_index_embed = self.get_index_embed(lap_node_id, node_mask, padded_index)  # [batch_size, T, 2Dl]
            padded_feature = padded_feature + self.lap_encoder(lap_index_embed)

        if self.type_id:
            padded_feature = padded_feature + self.get_type_embed(padded_index)

        padded_feature, padding_mask = self.add_special_tokens(padded_feature, padding_mask)  # [batch_size, 2+T, D], [batch_size, 2+T]

        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float('0'))
        return padded_feature, padding_mask, padded_index  # [batch_size, 2+T, D], [batch_size, 2+T], [batch_size, T, 2]


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.kdim = config.kdim if config.kdim is not None else config.embedding_dim
        self.vdim = config.vdim if config.vdim is not None else config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.self_attention = config.self_attention
        self.qkv_same_dim = self.kdim == config.embedding_dim and self.vdim == config.embedding_dim
        self.head_dim = config.embedding_dim // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        # TODO: all asserts must be replaced by Exception raising
        assert self.self_attention, "Only support self attention"
        assert not self.self_attention or self.qkv_same_dim, "Self-attention requires QKV to be of the same size"
        assert self.head_dim * self.num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"

        self.attention_dropout_module = torch.nn.Dropout(p=config.attention_dropout)
        self.dropout_module = torch.nn.Dropout(p=config.dropout)
        self.k_proj = quant_noise(nn.Linear(self.kdim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.q_proj = quant_noise(nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.out_proj = quant_noise(nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.reset_parameters()
        self.onnx_trace = False

    def performer_finetune_setup(self, performer_nb_features, performer_generalized_attention):
        self.fast_attention = FastAttention(
            self.head_dim,
            performer_nb_features,
            causal=False,
            generalized_attention=performer_generalized_attention,
            kernel_fn=nn.ReLU(),
            no_projection=False
        )
        self.forward = self.forward_performer

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            attn_bias: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel # make more explicit wrt grpahs

        config:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embedding_dim = query.size()
        src_len = tgt_len
        assert embedding_dim == self.embedding_dim, f"query dim {embedding_dim} != {self.embedding_dim}"
        assert list(query.size()) == [tgt_len, bsz, embedding_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)  # [T, batch_size, D]
        k = self.k_proj(query)  # [T, batch_size, D]
        v = self.v_proj(query)  # [T, batch_size, D]
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :].to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1)  # [bsz * num_heads, tgt_len, src_len]
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embedding_dim)

        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)  # [num_heads, bsz, tgt_len, src_len]
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim: 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    def forward_performer(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            attn_bias: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        config:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        assert attn_bias is None

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embedding_dim = query.size()
        src_len = tgt_len
        assert embedding_dim == self.embedding_dim, f"query dim {embedding_dim} != {self.embedding_dim}"
        assert list(query.size()) == [tgt_len, bsz, embedding_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        assert k is not None
        assert k.size(0) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
            key_padding_mask = key_padding_mask.to(torch.bool)[:, None, :, None]

        q, k, v = map(lambda t: rearrange(t, 'n batch_size (h d) -> batch_size h n d', h=self.num_heads), (q, k, v))
        attn = self.fast_attention(q, k, v, key_padding_mask)
        attn = rearrange(attn, 'batch_size h n d -> n batch_size (h d)')

        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            raise NotImplementedError

        return attn, attn_weights

class MultiheadPerformerAttention(MultiheadAttention):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config):
        super().__init__(config)
        assert attention_dropout == 0.0
        self.fast_attention = FastAttention(config)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            attn_bias: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        config:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        return self.forward_performer(
            query,
            key,
            value,
            attn_bias,
            key_padding_mask,
            need_weights,
            attn_mask,
            before_softmax,
            need_head_weights
        )

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        # x: T x batch_size x C
        keep_prob = 1 - self.drop_prob
        random_tensor = x.new_empty(1, x.size(1), 1).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = quant_noise(nn.Linear(config.embedding_dim, config.ffn_embedding_dim), config.q_noise, config.qn_block_size)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.activation_dropout_module = torch.nn.Dropout(p=config.activation_dropout)
        self.fc2 = quant_noise(nn.Linear(config.ffn_embedding_dim, config.embedding_dim), config.q_noise, config.qn_block_size)
        self.dropout_module = torch.nn.Dropout(p=config.dropout)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class TokenGTGraphEncoderLayer(nn.Module):
    def __init__(self, config, drop_path) -> None:
        super().__init__()

        init_fn = config.init_fn

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = config.embedding_dim
        self.ffn_embedding_dim = config.ffn_embedding_dim
        self.encoder_layers = config.num_layers
        self.num_attention_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.q_noise = config.q_noise
        self.qn_block_size = config.qn_block_size
        self.layernorm_style = config.layernorm_style
        self.return_attention = config.return_attention

        self.dropout_module = torch.nn.Dropout(p=config.dropout)

        # Initialize blocks
        if config.performer:
            self.self_attn = MultiheadPerformerAttention(config)
        else:
            self.self_attn = MultiheadAttention(config)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embedding_dim)

        # drop path for stochastic depth
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.feedforward = FeedForward(config)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = torch.nn.LayerNorm(self.embedding_dim)

        # drop path for stochastic depth
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def performer_finetune_setup(self, performer_nb_features, performer_generalized_attention):
        self.self_attn.performer_finetune_setup(performer_nb_features, performer_generalized_attention)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_bias: Optional[torch.Tensor] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x batch_size x C
        if self.layernorm_style == "prenorm":
            residual = x
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                attn_bias=self_attn_bias,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                need_head_weights=self.return_attention,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = self.drop_path1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.feedforward(x)
            x = self.drop_path2(x)
            x = residual + x

        elif self.layernorm_style == "postnorm":
            residual = x
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                attn_bias=self_attn_bias,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                need_head_weights=self.return_attention,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.feedforward(x)
            x = residual + x
            x = self.final_layer_norm(x)

        else:
            raise NotImplementedError
        return x, attn


class TokenGTGraphEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dropout_module = torch.nn.Dropout(p=config.dropout)
        self.layerdrop = config.layerdrop
        self.embedding_dim = config.embedding_dim
        self.apply_graphormer_init = config.apply_graphormer_init
        self.traceable = config.traceable
        self.performer = config.performer
        self.performer_finetune = config.performer_finetune

        self.graph_feature = GraphFeatureTokenizer(config)
        self.performer_finetune = config.performer_finetune
        self.embed_scale = config.embed_scale

        if config.q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                config.q_noise,
                config.qn_block_size,
            )
        else:
            self.quant_noise = None

        if config.encoder_normalize_before:
            self.emb_layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        if config.layernorm_style == "prenorm":
            self.final_layer_norm = torch.nn.LayerNorm(self.embedding_dim)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        if config.stochastic_depth:
            assert config.layernorm_style == 'prenorm'  # only for residual nets

        self.cached_performer_options = None
        if self.performer_finetune:
            assert self.performer
            self.cached_performer_options = (
                config.performer_nb_features,
                config.performer_generalized_attention,
                config.performer_auto_check_redraw,
                config.performer_feature_redraw_interval
            )
            self.performer = False
            config.performer = False
            config.performer_nb_features = None
            config.performer_generalized_attention = False
            config.performer_auto_check_redraw = False
            config.performer_feature_redraw_interval = None

        self.layers.extend(
            [
                TokenGTGraphEncoderLayer(
                    config, 
                    drop_path=(0.1 * (layer_idx + 1) / config.num_layers) if config.stochastic_depth else 0,
                )
                for layer_idx in range(config.num_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if config.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(config.n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        if config.performer:
            # keeping track of when to redraw projections for all attention layers
            self.performer_auto_check_redraw = config.performer_auto_check_redraw
            self.performer_proj_updater = ProjectionUpdater(self.layers, config.performer_feature_redraw_interval)

    def performer_fix_projection_matrices_(self):
        self.performer_proj_updater.feature_redraw_interval = None

    def performer_finetune_setup(self):
        assert self.performer_finetune
        (
            performer_nb_features,
            performer_generalized_attention,
            performer_auto_check_redraw,
            performer_feature_redraw_interval
        ) = self.cached_performer_options

        for layer in self.layers:
            layer.performer_finetune_setup(performer_nb_features, performer_generalized_attention)

        self.performer = True
        self.performer_auto_check_redraw = performer_auto_check_redraw
        self.performer_proj_updater = ProjectionUpdater(self.layers, performer_feature_redraw_interval)

    def forward(
            self,
            batched_data,
            perturb=None,
            last_state_only: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        is_tpu = False

        if self.performer and self.performer_auto_check_redraw:
            self.performer_proj_updater.redraw_projections()

        if token_embeddings is not None:
            raise NotImplementedError
        else:
            x, padding_mask, padded_index = self.graph_feature(batched_data, perturb)

        # x: batch_size x T x C

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # batch_size x T x C -> T x batch_size x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        if attn_mask is not None:
            raise NotImplementedError

        attn_dict = {'maps': {}, 'padded_index': padded_index}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, attn = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=None)
            if not last_state_only:
                inner_states.append(x)
            attn_dict['maps'][i] = attn

        # TODO: CHECK THIS
        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), graph_rep, attn_dict
        else:
            return inner_states, graph_rep, attn_dict


class TokenGTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_nodes = config.max_nodes
        self.num_embedding_dim = config.num_layers
        self.num_attention_heads = config.num_attention_heads
        self.return_attention = config.return_attention

        self.graph_encoder = TokenGTGraphEncoder(config)

        self.share_input_output_embed = config.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(config, "remove_head", False)
        self.masked_lm_pooler = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.lm_head_transform_weight = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.layer_norm = torch.nn.LayerNorm(config.embedding_dim)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep, _ = self.graph_encoder(batched_data, perturb=perturb)

        x = inner_states[-1].transpose(0, 1)  # batch_size x T x C

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
                self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)

        if self.return_attention:
            return x, attn_dict
        else:
            return x

    def performer_finetune_setup(self):
        self.graph_encoder.performer_finetune_setup()

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict


class TokenGTDecoderHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, load_softmax, share_input_output_embed):
        super().__init__()
        """num_classes should be 1 for regression, and the number of classes for classification"""
        self.embed_out = None
        self.lm_output_learned_bias = None

        if load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
            if not share_input_output_embed:
                self.embed_out = nn.Linear(embedding_dim, num_classes, bias=False)
            else:
                raise NotImplementedError
        self.num_classes = num_classes

    def forward(self, x, **unused):
        if self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias
        return x

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

class TokenGTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    # TODO: all init must be here!! (see Graphormer code)
    """

    config_class = TokenGTConfig
    base_model_prefix = "tokengt"
    supports_gradient_checkpointing = True
    #_keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        self.apply(init_graphormer_params)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TokenGTEncoder):
            module.gradient_checkpointing = value


class TokenGTForGraphClassification(TokenGTPreTrainedModel):
    """Also works for graph regression"""
    def __init__(self, config):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self.tasks_weights = config.tasks_weights

        self.encoder = TokenGTEncoder(config)
        self.decoder = TokenGTDecoderHead(
            self.embedding_dim, 
            config.num_labels, 
            not getattr(config, "remove_head", False), 
            config.share_encoder_input_output_embed
        )

        if getattr(config, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)

        if config.performer_finetune:
            self.encoder.performer_finetune_setup()


    def forward(
        self,
        node_data,
        num_nodes,
        edge_index,
        edge_data,
        edge_num,
        in_degree,
        out_degree,
        lap_eigvec,
        lap_eigval,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        batched_data = {
            "node_data": node_data,
            "node_num": num_nodes,
            "edge_index": edge_index,
            "edge_data": edge_data,
            "edge_num": edge_num,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "lap_eigvec": lap_eigvec,
            "lap_eigval": lap_eigval,
        }
        outputs = self.encoder(batched_data)
        decoder = self.decoder_head
        head_outputs = cur_decoder(outputs)
        logits = head_outputs[:, 0, :].contiguous() # graph token

        loss = None
        if labels is not None:
            mask = ~torch.isnan(labels)

            if decoder.num_classes == 1: # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif decoder.num_classes > 1 and len(labels.shape) == 1: # One task classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, decoder.num_classes), labels[mask].view(-1))
            else: # Binary multi-task classification
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = loss_fct(logits[mask], labels[mask])

        # TODO: a specific graph classification output class will have to be created at a later stage
        return SequenceClassifierOutput(
            loss=loss, 
            logits=logits, 
            hidden_states=None, 
            attentions=None
            )

    def max_nodes(self):
        return self.encoder.max_nodes

