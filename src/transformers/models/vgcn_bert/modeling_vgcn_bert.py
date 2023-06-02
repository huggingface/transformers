# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

"""
 PyTorch VGCN-BERT model adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM) and in
 part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""


import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.configuration_utils import PretrainedConfig

from ...activations import get_activation
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_vgcn_bert import VGCNBertConfig


logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = "zhibinlu/vgcn-distilbert-base-uncased"
_CONFIG_FOR_DOC = "VGCNBertConfig"

VGCNBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "zhibinlu/vgcn-distilbert-base-uncased",
    # See all VGCN-BERT models at https://huggingface.co/models?filter=VGCNBert
]


# UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE #


def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(out, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)
    else:
        _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)


def _create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()


class VgcnParameterList(nn.ParameterList):
    def __init__(self, values=None, requires_grad=True) -> None:
        super().__init__(values)
        self.requires_grad = requires_grad

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        keys = filter(lambda x: x.startswith(prefix), state_dict.keys())
        for k in keys:
            self.append(nn.Parameter(state_dict[k], requires_grad=self.requires_grad))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        for i in range(len(self)):
            if self[i].layout is torch.sparse_coo and not self[i].is_coalesced():
                self[i] = self[i].coalesce()
            self[i].requires_grad = self.requires_grad


class VocabGraphConvolution(nn.Module):
    """Vocabulary GCN module.

    Params:
        `wgraphs`: List of vocabulary graph, normally adjacency matrix
        `wgraph_id_to_tokenizer_id_maps`: wgraph.vocabulary to tokenizer.vocabulary id-mapping
        `hid_dim`: The hidden dimension after `GCN=XAW` (GCN layer)
        `out_dim`: The output dimension after `out=Relu(XAW)W`  (GCN output)
        `activation`: The activation function in `out=act(XAW)W`
        `dropout_rate`: The dropout probabilitiy in `out=dropout(act(XAW))W`.

    Inputs:
        `X_dv`: the feature of mini batch document, can be TF-IDF (batch, vocab), or word embedding (batch, word_embedding_dim, vocab)

    Outputs:
        The graph embedding representation, dimension (batch, `out_dim`) or (batch, word_embedding_dim, `out_dim`)

    """

    def __init__(
        self,
        hid_dim: int,
        out_dim: int,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
        activation=None,
        dropout_rate=0.1,
    ):
        super().__init__()
        # self.wgraphs:VgcnParameterList = self._prepare_wgraphs(wgraphs) if wgraphs else VgcnParameterList(requires_grad=False)
        # self.gvoc_ordered_tokenizer_id_arrays, self.tokenizer_id_to_wgraph_id_arrays=VgcnParameterList(requires_grad=False),VgcnParameterList(requires_grad=False)
        # if wgraph_id_to_tokenizer_id_maps:
        #     self.gvoc_ordered_tokenizer_id_arrays, self.tokenizer_id_to_wgraph_id_arrays = self._prepare_inverted_arrays(
        #         wgraph_id_to_tokenizer_id_maps
        #     )

        self.hid_dim = hid_dim
        self.out_dim = out_dim

        # self.W_vh_list = VgcnParameterList(requires_grad=True)
        # self.W_vh_list._is_vgcn_weights = True
        # for g in self.wgraphs:
        #     self.W_vh_list.append(nn.Parameter(torch.randn(g.shape[0], hid_dim)))
        #     # self.W_vh_list.append(nn.Parameter(torch.ones(g.shape[0], hid_dim)))

        self.fc_hg = nn.Linear(hid_dim, out_dim)
        self.fc_hg._is_vgcn_linear = True
        self.activation = get_activation(activation) if activation else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        # TODO: add a Linear layer for vgcn fintune/pretrain task

        # after init.set_wgraphs, _init_weights will set again the mode (transparent,normal,uniform)
        # but if load wgraph parameters from checkpoint/pretrain, the mode will be accoding to the checkpoint
        # you can call again set_parameters to change the mode
        self.set_wgraphs(wgraphs, wgraph_id_to_tokenizer_id_maps)

    def set_parameters(self, mode="transparent"):
        """Set the parameters of the model (transparent, uniform, normal)."""
        assert mode in ["transparent", "uniform", "normal"]
        for n, p in self.named_parameters():
            if n.startswith("W"):
                nn.init.constant_(p, 1.0) if mode == "transparent" else nn.init.normal_(
                    p, mean=0.0, std=0.02
                ) if mode == "normal" else nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.fc_hg.weight.data.fill_(1.0) if mode == "transparent" else self.fc_hg.weight.data.normal_(
            mean=0.0, std=0.02
        ) if mode == "normal" else nn.init.kaiming_uniform_(self.fc_hg.weight, a=math.sqrt(5))
        self.fc_hg.bias.data.zero_()

    def set_wgraphs(
        self,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
        mode="transparent",
    ):
        assert (
            wgraphs is None
            and wgraph_id_to_tokenizer_id_maps is None
            or wgraphs is not None
            and wgraph_id_to_tokenizer_id_maps is not None
        )
        self.wgraphs: VgcnParameterList = (
            self._prepare_wgraphs(wgraphs) if wgraphs else VgcnParameterList(requires_grad=False)
        )
        self.gvoc_ordered_tokenizer_id_arrays, self.tokenizer_id_to_wgraph_id_arrays = VgcnParameterList(
            requires_grad=False
        ), VgcnParameterList(requires_grad=False)
        if wgraph_id_to_tokenizer_id_maps:
            (
                self.gvoc_ordered_tokenizer_id_arrays,
                self.tokenizer_id_to_wgraph_id_arrays,
            ) = self._prepare_inverted_arrays(wgraph_id_to_tokenizer_id_maps)
        self.W_vh_list = VgcnParameterList(requires_grad=True)
        self.W_vh_list._is_vgcn_weights = True
        for g in self.wgraphs:
            self.W_vh_list.append(nn.Parameter(torch.randn(g.shape[0], self.hid_dim)))
            # self.W_vh_list.append(nn.Parameter(torch.ones(g.shape[0], self.hid_dim)))
        self.set_parameters(mode=mode)

    # def set_wgraphs(self, wgraphs: list, wgraph_id_to_tokenizer_id_maps: List[dict]):
    #     self.wgraphs:VgcnParameterList = self._prepare_wgraphs(wgraphs)
    #     self.W_vh_list = VgcnParameterList(requires_grad=True)
    #     self.W_vh_list._is_vgcn_weights = True
    #     for g in self.wgraphs:
    #         self.W_vh_list.append(nn.Parameter(torch.randn(g.shape[0], self.hid_dim)))
    #     self.gvoc_ordered_tokenizer_id_arrays, self.tokenizer_id_to_wgraph_id_arrays = self._prepare_inverted_arrays(
    #         wgraph_id_to_tokenizer_id_maps
    #     )

    def _prepare_wgraphs(self, wgraphs: list) -> VgcnParameterList:
        # def _zero_padding_graph(adj_matrix: torch.Tensor):
        #     if adj_matrix.layout is not torch.sparse_coo:
        #         adj_matrix=adj_matrix.to_sparse_coo()
        #     indices=adj_matrix.indices()+1
        #     padded_adj= torch.sparse_coo_tensor(indices=indices, values=adj_matrix.values(), size=(adj_matrix.shape[0]+1,adj_matrix.shape[1]+1))
        #     return padded_adj.coalesce()
        glist = VgcnParameterList(requires_grad=False)
        for g in wgraphs:
            assert g.layout is torch.sparse_coo
            # g[0,:] and g[:,0] should be 0
            assert 0 not in g.indices()
            glist.append(nn.Parameter(g.coalesce(), requires_grad=False))
        return glist

    def _prepare_inverted_arrays(self, wgraph_id_to_tokenizer_id_maps: List[dict]):
        wgraph_id_to_tokenizer_id_maps = [dict(sorted(m.items())) for m in wgraph_id_to_tokenizer_id_maps]
        assert all([list(m.keys())[-1] == len(m) - 1 for m in wgraph_id_to_tokenizer_id_maps])
        gvoc_ordered_tokenizer_id_arrays = VgcnParameterList(
            [
                nn.Parameter(torch.LongTensor(list(m.values())), requires_grad=False)
                for m in wgraph_id_to_tokenizer_id_maps
            ],
            requires_grad=False,
        )

        tokenizer_id_to_wgraph_id_arrays = VgcnParameterList(
            [
                nn.Parameter(torch.zeros(max(m.values()) + 1, dtype=torch.long), requires_grad=False)
                for m in wgraph_id_to_tokenizer_id_maps
            ],
            requires_grad=False,
        )
        for m, t in zip(wgraph_id_to_tokenizer_id_maps, tokenizer_id_to_wgraph_id_arrays):
            for graph_id, tok_id in m.items():
                t[tok_id] = graph_id

        return gvoc_ordered_tokenizer_id_arrays, tokenizer_id_to_wgraph_id_arrays

    def get_subgraphs(self, adj_matrix: torch.Tensor, gx_ids: torch.LongTensor):
        device = gx_ids.device
        batch_size = gx_ids.shape[0]
        batch_masks = torch.any(
            torch.any(
                (adj_matrix.indices().view(-1) == gx_ids.unsqueeze(-1)).view(batch_size, gx_ids.shape[1], 2, -1), dim=1
            ),
            dim=1,
        )
        nnz_len = len(adj_matrix.values())

        batch_values = adj_matrix.values().unsqueeze(0).repeat(batch_size, 1)
        batch_values = batch_values.view(-1)[batch_masks.view(-1)]

        batch_positions = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, nnz_len)
        indices = torch.cat([batch_positions.view(1, -1), adj_matrix.indices().repeat(1, batch_size)], dim=0)
        indices = indices[batch_masks.view(-1).expand(3, -1)].view(3, -1)

        batch_sub_adj_matrix = torch.sparse_coo_tensor(
            indices=indices,
            values=batch_values.view(-1),
            size=(batch_size, adj_matrix.size(0), adj_matrix.size(1)),
            dtype=adj_matrix.dtype,
            device=device,
        )

        return batch_sub_adj_matrix.coalesce()

    def forward(self, word_embeddings: nn.Embedding, input_ids: torch.Tensor):  # , position_ids: torch.Tensor = None):
        if not self.wgraphs:
            raise ValueError(
                "No wgraphs is provided. There are 3 ways to initalize wgraphs:"
                " instantiate VGCN_BERT with wgraphs, or call model.vgcn_bert.set_wgraphs(),"
                " or load from_pretrained/checkpoint (make sure there is wgraphs in checkpoint"
                " or you should call set_wgraphs)."
            )
        device = input_ids.device
        batch_size = input_ids.shape[0]
        word_emb_dim = word_embeddings.weight.shape[1]

        gx_ids_list = []
        # positon_embeddings_in_gvocab_order_list=[]
        for m in self.tokenizer_id_to_wgraph_id_arrays:
            # tmp_ids is still in sentence order, but value is graph id, e.g. [0, 5, 2, 2, 0, 10,0]
            # 0 means no correspond graph id (like padding in graph), so we need to replace it with 0
            tmp_ids = input_ids.clone()
            tmp_ids[tmp_ids > len(m) - 1] = 0
            tmp_ids = m[tmp_ids]

            # # position in graph is meaningless and computationally expensive
            # if position_ids:
            #     position_ids_in_g=torch.zeros(g.shape[0], dtype=torch.LongTensor)
            #     # maybe gcn_swop_eye in original vgcn_bert preprocess is more efficient?
            #     for p_id, g_id in zip(position_ids, tmp_ids):
            #         position_ids_in_g[g_id]=p_id
            #     position_embeddings_in_g=self.position_embeddings(position_ids_in_g)
            #     position_embeddings_in_g*=position_ids_in_g>0
            #     positon_embeddings_in_gvocab_order_list.append(position_embeddings_in_g)

            gx_ids_list.append(torch.unique(tmp_ids, dim=1))

        # G_embedding=(act(V1*A1_sub*W1_vh)+act(V2*A2_sub*W2_vh)）*W_hg
        fused_H = torch.zeros((batch_size, word_emb_dim, self.hid_dim), device=device)
        for gv_ids, g, gx_ids, W_vh in zip(  # , position_in_gvocab_ev
            self.gvoc_ordered_tokenizer_id_arrays,
            self.wgraphs,
            gx_ids_list,
            self.W_vh_list,
            # positon_embeddings_in_gvocab_order_list,
        ):
            # batch_A1_sub*W1_vh, batch_A2_sub*W2_vh, ...
            sub_wgraphs = self.get_subgraphs(g, gx_ids)
            H_vh = torch.bmm(sub_wgraphs, W_vh.unsqueeze(0).expand(batch_size, *W_vh.shape))

            # V1*batch_A1_sub*W1_vh, V2*batch_A2_sub*W2_vh, ...
            gvocab_ev = word_embeddings(gv_ids).t()
            # if position_ids:
            #     gvocab_ev += position_in_gvocab_ev
            H_eh = gvocab_ev.matmul(H_vh)

            # fc -> act -> dropout
            if self.activation:
                H_eh = self.activation(H_eh)
            if self.dropout:
                H_eh = self.dropout(H_eh)

            fused_H += H_eh

        # fused_H=LayerNorm(fused_H) # embedding assemble layer will do LayerNorm
        out_ge = self.fc_hg(fused_H).transpose(1, 2)
        # self.dropout(out_ge) # embedding assemble layer will do dropout
        return out_ge


class VGCNEmbeddings(nn.Module):
    """Construct the embeddings from word, VGCN graph, position and token_type embeddings."""

    def __init__(
        self,
        config: PretrainedConfig,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
    ):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)

        self.vgcn_graph_embds_dim = config.vgcn_graph_embds_dim
        self.vgcn = VocabGraphConvolution(
            hid_dim=config.vgcn_hidden_dim,
            out_dim=config.vgcn_graph_embds_dim,
            wgraphs=wgraphs,
            wgraph_id_to_tokenizer_id_maps=wgraph_id_to_tokenizer_id_maps,
            activation=config.vgcn_activation,
            dropout_rate=config.vgcn_dropout,
        )

        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
                input_ids is mandatory in vgcn-bert.

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """

        # input_ids is mandatory in vgcn-bert
        input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        # device = input_embeds.device
        # input_lengths = (
        #     (input_ids > 0).sum(-1)
        #     if input_ids is not None
        #     else torch.ones(input_embeds.size(0), device=device, dtype=torch.int64) * input_embeds.size(1)
        # )

        seq_length = input_embeds.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)

        if self.vgcn_graph_embds_dim > 0:
            # TODO: check input_ids/position_ids donot include [CLS], [SEP][SEP]
            graph_embeds = self.vgcn(self.word_embeddings, input_ids)  # , position_ids)

            # vgcn_words_embeddings = input_embeds.clone()
            # for i in range(self.vgcn_graph_embds_dim):
            #     tmp_pos = (input_lengths - 2 - self.vgcn_graph_embds_dim + 1 + i) + torch.arange(
            #         0, input_embeds.shape[0]
            #     ).to(device) * input_embeds.shape[1]
            #     vgcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = graph_embeds[:, :, i]

            embeddings = torch.cat([embeddings, graph_embeds], dim=1)  # (bs, max_seq_length+graph_emb_dim_size, dim)

        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        # Have an even number of multi heads that divide the dimensions
        if self.dim % self.n_heads != 0:
            # Raise value errors for even multi-head attention nodes
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads: Set[int] = set()
        self.attention_head_size = self.dim // self.n_heads

    def prune_heads(self, heads: List[int]):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        self.activation = get_activation(config.activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input: torch.Tensor) -> torch.Tensor:
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # Have an even number of Configure multi-heads
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            if type(sa_output) != tuple:
                raise TypeError(f"sa_output must be a tuple but it is {type(sa_output)} type")

            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                if len(layer_outputs) != 2:
                    raise ValueError(f"The length of the layer_outputs should be 2, but it is {len(layer_outputs)}")

                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                if len(layer_outputs) != 1:
                    raise ValueError(f"The length of the layer_outputs should be 1, but it is {len(layer_outputs)}")

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )


# INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL #
# Copied from transformers.models.distilbert.modeling_distilbert.DistilBertPreTrainedModel with DistilBert->VGCNBert,distilbert->vgcn_bert
class VGCNBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VGCNBertConfig
    load_tf_weights = None
    base_model_prefix = "vgcn_bert"

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if getattr(module, "_is_vgcn_linear", False):
                if self.config.vgcn_weight_init_mode == "transparent":
                    module.weight.data.fill_(1.0)
                elif self.config.vgcn_weight_init_mode == "normal":
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                elif self.config.vgcn_weight_init_mode == "uniform":
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                else:
                    raise ValueError(f"Unknown VGCN-BERT weight init mode: {self.config.vgcn_weight_init_mode}.")
                if module.bias is not None:
                    module.bias.data.zero_()
            else:
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
        elif isinstance(module, nn.ParameterList):
            if getattr(module, "_is_vgcn_weights", False):
                for p in module:
                    if self.config.vgcn_weight_init_mode == "transparent":
                        nn.init.constant_(p, 1.0)
                    elif self.config.vgcn_weight_init_mode == "normal":
                        nn.init.normal_(p, mean=0.0, std=self.config.initializer_range)
                    elif self.config.vgcn_weight_init_mode == "uniform":
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    else:
                        raise ValueError(f"Unknown VGCN-BERT weight init mode: {self.config.vgcn_weight_init_mode}.")


VGCNBERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VGCNBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VGCNBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare VGCN-BERT encoder/transformer outputting raw hidden-states without any specific head on top.",
    VGCNBERT_START_DOCSTRING,
)
# Copied from transformers.models.distilbert.modeling_distilbert.DistilBertModel with DISTILBERT->VGCNBERT,DistilBert->VGCNBert
class VGCNBertModel(VGCNBertPreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
    ):
        super().__init__(config)

        self.embeddings = VGCNEmbeddings(config, wgraphs, wgraph_id_to_tokenizer_id_maps)  # Graph Embeddings
        self.transformer = Transformer(config)  # Encoder

        # Initialize weights and apply final processing
        self.post_init()

    def set_wgraphs(
        self,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
        mode="transparent",
    ):
        self.embeddings.vgcn.set_wgraphs(wgraphs, wgraph_id_to_tokenizer_id_maps, mode)

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight.clone()

        self.embeddings.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.dim)

        if self.config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=self.config.max_position_embeddings, dim=self.config.dim, out=self.position_embeddings.weight
            )
        else:
            with torch.no_grad():
                if num_position_embeds_diff > 0:
                    self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = nn.Parameter(
                        old_position_embeddings_weight
                    )
                else:
                    self.embeddings.position_embeddings.weight = nn.Parameter(
                        old_position_embeddings_weight[:num_position_embeds_diff]
                    )
        # move position_embeddings to correct device
        self.embeddings.position_embeddings.to(self.device)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[List[int]]]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VGCNBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
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

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embeddings = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)

        if self.embeddings.vgcn_graph_embds_dim > 0:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((input_shape[0], self.embeddings.vgcn_graph_embds_dim), device=device)],
                dim=1,
            )

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(
    """VGCNBert Model with a `masked language modeling` head on top.""",
    VGCNBERT_START_DOCSTRING,
)
# Copied from transformers.models.distilbert.modeling_distilbert.DistilBertForMaskedLM with DISTILBERT->VGCNBERT,DistilBert->VGCNBert,distilbert->vgcn_bert
class VGCNBertForMaskedLM(VGCNBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["vocab_projector.weight"]

    def __init__(
        self,
        config: PretrainedConfig,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
    ):
        super().__init__(config)

        self.activation = get_activation(config.activation)

        self.vgcn_bert = VGCNBertModel(config, wgraphs, wgraph_id_to_tokenizer_id_maps)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.vgcn_bert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.vgcn_bert.resize_position_embeddings(new_num_position_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.vocab_projector

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.vocab_projector = new_embeddings

    @add_start_docstrings_to_model_forward(VGCNBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.vgcn_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        # remove graph embedding outputs
        prediction_logits = prediction_logits[:, : input_ids.size(1), :]

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.reshape(-1, prediction_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )


@add_start_docstrings(
    """
    VGCNBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    VGCNBERT_START_DOCSTRING,
)
# Copied from transformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification with DISTILBERT->VGCNBERT,DistilBert->VGCNBert,distilbert->vgcn_bert
class VGCNBertForSequenceClassification(VGCNBertPreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.vgcn_bert = VGCNBertModel(config, wgraphs, wgraph_id_to_tokenizer_id_maps)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.vgcn_bert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.vgcn_bert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(VGCNBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vgcn_bert_output = self.vgcn_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = vgcn_bert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + vgcn_bert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=vgcn_bert_output.hidden_states,
            attentions=vgcn_bert_output.attentions,
        )


@add_start_docstrings(
    """
    VGCNBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    VGCNBERT_START_DOCSTRING,
)
# Copied from transformers.models.distilbert.modeling_distilbert.DistilBertForQuestionAnswering with DISTILBERT->VGCNBERT,DistilBert->VGCNBert,distilbert->vgcn_bert
class VGCNBertForQuestionAnswering(VGCNBertPreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
    ):
        super().__init__(config)

        self.vgcn_bert = VGCNBertModel(config, wgraphs, wgraph_id_to_tokenizer_id_maps)
        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        if config.num_labels != 2:
            raise ValueError(f"config.num_labels should be 2, but it is {config.num_labels}")

        self.dropout = nn.Dropout(config.qa_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.vgcn_bert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.vgcn_bert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(VGCNBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[QuestionAnsweringModelOutput, Tuple[torch.Tensor, ...]]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vgcn_bert_output = self.vgcn_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = vgcn_bert_output[0]  # (bs, max_query_len, dim)

        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        # remove graph embedding outputs
        logits = logits[:, : input_ids.size(1), :]

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + vgcn_bert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=vgcn_bert_output.hidden_states,
            attentions=vgcn_bert_output.attentions,
        )


@add_start_docstrings(
    """
    VGCNBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    VGCNBERT_START_DOCSTRING,
)
# Copied from transformers.models.distilbert.modeling_distilbert.DistilBertForTokenClassification with DISTILBERT->VGCNBERT,DistilBert->VGCNBert,distilbert->vgcn_bert
class VGCNBertForTokenClassification(VGCNBertPreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.vgcn_bert = VGCNBertModel(config, wgraphs, wgraph_id_to_tokenizer_id_maps)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.vgcn_bert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.vgcn_bert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(VGCNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vgcn_bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # remove graph embedding outputs
        logits = logits[:, : input_ids.size(1), :]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    VGCNBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    VGCNBERT_START_DOCSTRING,
)
# Copied from transformers.models.distilbert.modeling_distilbert.DistilBertForMultipleChoice with DISTILBERT->VGCNBERT,DistilBert->VGCNBert,distilbert->vgcn_bert
class VGCNBertForMultipleChoice(VGCNBertPreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
    ):
        super().__init__(config)

        self.vgcn_bert = VGCNBertModel(config, wgraphs, wgraph_id_to_tokenizer_id_maps)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.vgcn_bert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`)
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.vgcn_bert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(
        VGCNBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MultipleChoiceModelOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, VGCNBertForMultipleChoice
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("vgcn_bert-base-cased")
        >>> model = VGCNBertForMultipleChoice.from_pretrained("vgcn_bert-base-cased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."
        >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

        >>> encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors="pt", padding=True)
        >>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.vgcn_bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        pooled_output = self.dropout(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)  # (bs * num_choices, 1)

        reshaped_logits = logits.view(-1, num_choices)  # (bs, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
