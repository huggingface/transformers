""" PyTorch EGT model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from ...modeling_utils import PreTrainedModel
# from .configuration_egt import EGTConfig
from transformers.modeling_utils import PreTrainedModel
from configuration_egt import EGTConfig
from typing import Optional, Union, Tuple
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# from ...modeling_outputs import (
#     BaseModelOutputWithNoAttention,
#     SequenceClassifierOutput,
# )
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)

NODE_FEATURES_OFFSET = 128
NUM_NODE_FEATURES = 9
EDGE_FEATURES_OFFSET = 8
NUM_EDGE_FEATURES = 3

class EGTLayer(nn.Module):
    r"""EGTLayer for Edge-augmented Graph Transformer (EGT), as introduced in
    `Global Self-Attention as a Replacement for Graph Convolution
    Reference `<https://arxiv.org/pdf/2108.03348.pdf>`_

    Parameters
    ----------
    ndim : int
        Node embedding dimension.
    edim : int
        Edge embedding dimension.
    num_heads : int
        Number of attention heads, by which :attr: `ndim` is divisible.
    num_vns : int
        Number of virtual nodes.
    dropout : float, optional
        Dropout probability. Default: 0.0.
    attn_dropout : float, optional
        Attention dropout probability. Default: 0.0.
    activation : callable activation layer, optional
        Activation function. Default: nn.ELU().
    ffn_multiplier : float, optional
        Multiplier of the inner dimension in Feed Forward Network.
        Default: 2.0.
    edge_update : bool, optional
        Whether to update the edge embedding. Default: True.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import EGTLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> ndim, edim = 128, 32
    >>> nfeat = th.rand(batch_size, num_nodes, ndim)
    >>> efeat = th.rand(batch_size, num_nodes, num_nodes, edim)
    >>> net = EGTLayer(
            ndim=ndim,
            edim=edim,
            num_heads=8,
            num_vns=4,
        )
    >>> out = net(nfeat, efeat)
    """

    def __init__(
        self,
        ndim,
        edim,
        num_heads,
        num_vns,
        dropout=0,
        attn_dropout=0,
        activation=nn.ELU(),
        ffn_multiplier=2.0,
        edge_update=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_vns = num_vns
        self.edge_update = edge_update

        assert not (ndim % num_heads)
        self.dot_dim = ndim // num_heads
        self.mha_ln_h = nn.LayerNorm(ndim)
        self.mha_ln_e = nn.LayerNorm(edim)
        self.E = nn.Linear(edim, num_heads)
        self.QKV = nn.Linear(ndim, ndim * 3)
        self.G = nn.Linear(edim, num_heads)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.O_h = nn.Linear(ndim, ndim)
        self.mha_dropout_h = nn.Dropout(dropout)

        node_inner_dim = round(ndim * ffn_multiplier)
        self.node_ffn = nn.Sequential(
            nn.LayerNorm(ndim),
            nn.Linear(ndim, node_inner_dim),
            activation,
            nn.Linear(node_inner_dim, ndim),
            nn.Dropout(dropout),
        )

        if self.edge_update:
            self.O_e = nn.Linear(num_heads, edim)
            self.mha_dropout_e = nn.Dropout(dropout)
            edge_inner_dim = round(edim * ffn_multiplier)
            self.edge_ffn = nn.Sequential(
                nn.LayerNorm(edim),
                nn.Linear(edim, edge_inner_dim),
                activation,
                nn.Linear(edge_inner_dim, edim),
                nn.Dropout(dropout),
            )

    def forward(self, h, e, mask=None):
        """Forward computation. Note: :attr:`h` and :attr:`e` should be padded
        with embedding of virtual nodes if :attr:`num_vns` > 0, while
        :attr:`mask` should be padded with `0` values for virtual nodes.

        Parameters
        ----------
        h : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`ndim`), where N
            is the sum of maximum number of nodes and number of virtual nodes.
        e : torch.Tensor
            Edge embedding used for attention computation and self update.
            Shape: (batch_size, N, N, :attr:`edim`).
        mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where invalid positions are indicated by `-inf` and
            valid positions are indicated by `0`.
            Shape: (batch_size, N, N). Default: None.

        Returns
        -------
        h : torch.Tensor
            The output node embedding. Shape: (batch_size, N, :attr:`ndim`).
        e : torch.Tensor
            The output edge embedding. Shape: (batch_size, N, N, :attr:`edim`).
        """

        h_r1 = h
        e_r1 = e

        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)
        QKV = self.QKV(h_ln)
        E = self.E(e_ln)
        G = self.G(e_ln)
        shp = QKV.shape
        Q, K, V = QKV.view(shp[0], shp[1], -1, self.num_heads).split(
            self.dot_dim, dim=2
        )
        A_hat = torch.einsum("bldh,bmdh->blmh", Q, K)
        H_hat = A_hat.clamp(-5, 5) + E

        if mask is None:
            gates = torch.sigmoid(G)
            A_tild = F.softmax(H_hat, dim=2) * gates
        else:
            gates = torch.sigmoid(G + mask.unsqueeze(-1))
            A_tild = F.softmax(H_hat + mask.unsqueeze(-1), dim=2) * gates

        A_tild = self.attn_dropout(A_tild)
        V_attn = torch.einsum("blmh,bmkh->blkh", A_tild, V)

        # Scale the aggregated values by degree.
        degrees = torch.sum(gates, dim=2, keepdim=True)
        degree_scalers = torch.log(1 + degrees)
        degree_scalers[:, : self.num_vns] = 1.0
        V_attn = V_attn * degree_scalers

        V_attn = V_attn.reshape(shp[0], shp[1], self.num_heads * self.dot_dim)
        h = self.O_h(V_attn)

        h = self.mha_dropout_h(h)
        h.add_(h_r1)
        h_r2 = h
        h = self.node_ffn(h)
        h.add_(h_r2)

        if self.edge_update:
            e = self.O_e(H_hat)
            e = self.mha_dropout_e(e)
            e.add_(e_r1)
            e_r2 = e
            e = self.edge_ffn(e)
            e.add_(e_r2)

        return h, e


class VirtualNodes(nn.Module):
    def __init__(self, node_width, edge_width, num_virtual_nodes = 1):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_virtual_nodes = num_virtual_nodes

        self.vn_node_embeddings = nn.Parameter(torch.empty(num_virtual_nodes,
                                                           self.node_width))
        self.vn_edge_embeddings = nn.Parameter(torch.empty(num_virtual_nodes,
                                                           self.edge_width))
        nn.init.normal_(self.vn_node_embeddings)
        nn.init.normal_(self.vn_edge_embeddings)
    
    def forward(self, h, e, mask):
        
        node_emb = self.vn_node_embeddings.unsqueeze(0).expand(h.shape[0], -1, -1)
        h = torch.cat([node_emb, h], dim=1)
        
        e_shape = e.shape
        edge_emb_row = self.vn_edge_embeddings.unsqueeze(1)
        edge_emb_col = self.vn_edge_embeddings.unsqueeze(0)
        edge_emb_box = 0.5 * (edge_emb_row + edge_emb_col)
        
        edge_emb_row = edge_emb_row.unsqueeze(0).expand(e_shape[0], -1, e_shape[2], -1)
        edge_emb_col = edge_emb_col.unsqueeze(0).expand(e_shape[0], e_shape[1], -1, -1)
        edge_emb_box = edge_emb_box.unsqueeze(0).expand(e_shape[0], -1, -1, -1)
        
        e = torch.cat([edge_emb_row, e], dim=1)
        e_col_box = torch.cat([edge_emb_box, edge_emb_col], dim=1)
        e = torch.cat([e_col_box, e], dim=2)
        
        if mask is not None:
            mask = F.pad(mask, (self.num_virtual_nodes,0, self.num_virtual_nodes,0), 
                           mode='constant', value=0)
        return h, e, mask


class EGTPreTrainedModel(PreTrainedModel):
    """
    A simple interface for downloading and loading pretrained models.
    """

    config_class = EGTConfig
    base_model_prefix = "egt"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, EGTModel):
            module.gradient_checkpointing = value


class EGTModel(EGTPreTrainedModel):
    def __init__(self, config: EGTConfig):
        super().__init__(config)

        self.layer_common_kwargs = dict(
             ndim          = config.node_width            ,
             edim          = config.edge_width            ,
             num_heads           = config.num_heads             ,
             num_vns  = config.num_virtual_nodes,
             dropout    = config.dropout      ,
             attn_dropout        = config.attn_dropout          ,
             activation          = config.activation            ,
             ffn_multiplier = config.ffn_multiplier   ,
        )        
        
        self.EGT_layers = nn.ModuleList([EGTLayer(**self.layer_common_kwargs, 
                                                   edge_update=(not config.egt_simple))
                                         for _ in range(config.model_height-1)])
    
        self.EGT_layers.append(EGTLayer(**self.layer_common_kwargs, edge_update = False))

        self.upto_hop          = config.upto_hop
        self.num_virtual_nodes = config.num_virtual_nodes
        self.svd_encodings     = config.svd_encodings
        
        self.nodef_embed = nn.Embedding(NUM_NODE_FEATURES*NODE_FEATURES_OFFSET+1,
                                        config.node_width, padding_idx=0)
        if self.svd_encodings:
            self.svd_embed = nn.Linear(self.svd_encodings*2, config.node_width)
        
        self.dist_embed = nn.Embedding(self.upto_hop+2, config.edge_width)
        self.featm_embed = nn.Embedding(NUM_EDGE_FEATURES*EDGE_FEATURES_OFFSET+1,
                                        config.edge_width, padding_idx=0)
        
        if self.num_virtual_nodes > 0:
            self.vn_layer = VirtualNodes(config.node_width, config.edge_width, 
                                         self.num_virtual_nodes)
        
        self.final_ln_h = nn.LayerNorm(config.node_width)
        mlp_dims = [config.node_width * max(self.num_virtual_nodes, 1)]\
                    +[round(config.node_width*r) for r in config.mlp_ratios]\
                        +[config.output_dim]
        self.mlp_layers = nn.ModuleList([nn.Linear(mlp_dims[i],mlp_dims[i+1])
                                         for i in range(len(mlp_dims)-1)])
        self.mlp_fn = config.activation
     
    def input_block(self, nodef, featm, dm, nodem, svd_encodings):
        dm = dm.long().clamp(min=0, max=self.upto_hop+1)  # (b,i,j)
        
        h = self.nodef_embed(nodef).sum(dim=2)      # (b,i,w,h) -> (b,i,h)
        
        if self.svd_encodings:
            h = h + self.svd_embed(svd_encodings)
        
        e = self.dist_embed(dm)\
              + self.featm_embed(featm).sum(dim=3)  # (b,i,j,f,e) -> (b,i,j,e)
        
        mask = (nodem[:,:,None] * nodem[:,None,:] - 1)*1e9
        
        if self.num_virtual_nodes > 0:
            h, e, mask = self.vn_layer(h, e, mask)
        return h, e, mask
    
    def final_embedding(self, h, attn_mask):
        h = self.final_ln_h(h)
        if self.num_virtual_nodes > 0:
            h = h[:,:self.num_virtual_nodes].reshape(h.shape[0],-1)
        else:
            nodem = attn_mask.float().unsqueeze(dim=-1)
            h = (h*nodem).sum(dim=1)/(nodem.sum(dim=1)+1e-9)
        return h
    
    def output_block(self, h):
        h = self.mlp_layers[0](h)
        for layer in self.mlp_layers[1:]:
            h = layer(self.mlp_fn(h))
        return h

    def forward(
        self,
        node_feat: torch.LongTensor,
        featm: torch.LongTensor,
        dm: torch.LongTensor,
        attn_mask: torch.LongTensor,
        svd_encodings: torch.Tensor,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> torch.Tensor:
        h, e, mask = self.input_block(node_feat, featm, dm, attn_mask, svd_encodings)
        
        for layer in self.EGT_layers:
            h, e = layer(h, e, mask)
        
        h = self.final_embedding(h, attn_mask)
        
        outputs = self.output_block(h)
        
        if not return_dict:
            return tuple(x for x in [outputs] if x is not None)
        return BaseModelOutputWithNoAttention(last_hidden_state=outputs)


class EGTForGraphClassification(EGTPreTrainedModel):
    """
    This model can be used for graph-level classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_classes to the number of classes); there should be one integer
      label per graph
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list
      of integer labels for each graph.
    """

    def __init__(self, config: EGTConfig):
        super().__init__(config)
        self.model = EGTModel(config)
        self.num_classes = config.output_dim

    def forward(
        self,
        input_nodes: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_mask: torch.LongTensor,
        svd_pe: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits = self.model(
            input_nodes,
            attn_edge_type,
            spatial_pos,
            attn_mask,
            svd_pe,
            return_dict,
        )['last_hidden_state']

        loss = None
        print(logits)
        print(labels)
        if labels is not None:
            mask = ~torch.isnan(labels)

            if self.num_classes == 1:  # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif self.num_classes > 1 and len(labels.shape) == 1:  # One task classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, self.num_classes), labels[mask].view(-1))
            else:  # Binary multi-task classification
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = loss_fct(logits[mask], labels[mask])

        if not return_dict:
            return tuple(x for x in [loss, logits] if x is not None)
        return SequenceClassifierOutput(loss=loss, logits=logits, attentions=None)
