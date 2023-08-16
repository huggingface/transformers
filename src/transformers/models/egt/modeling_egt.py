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
# from dgl.nn import EGTLayer

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
    feat_size : int
        Node feature size.
    edge_feat_size : int
        Edge feature size.
    num_heads : int
        Number of attention heads, by which :attr: `feat_size` is divisible.
    num_virtual_nodes : int
        Number of virtual nodes.
    dropout : float, optional
        Dropout probability. Default: 0.0.
    attn_dropout : float, optional
        Attention dropout probability. Default: 0.0.
    activation : callable activation layer, optional
        Activation function. Default: nn.ELU().
    edge_update : bool, optional
        Whether to update the edge embedding. Default: True.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import EGTLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> feat_size, edge_feat_size = 128, 32
    >>> nfeat = th.rand(batch_size, num_nodes, feat_size)
    >>> efeat = th.rand(batch_size, num_nodes, num_nodes, edge_feat_size)
    >>> net = EGTLayer(
            feat_size=feat_size,
            edge_feat_size=edge_feat_size,
            num_heads=8,
            num_virtual_nodes=4,
        )
    >>> out = net(nfeat, efeat)
    """

    def __init__(
        self,
        feat_size,
        edge_feat_size,
        num_heads,
        num_virtual_nodes,
        dropout=0,
        attn_dropout=0,
        activation=nn.ELU(),
        edge_update=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_virtual_nodes = num_virtual_nodes
        self.edge_update = edge_update

        assert (
            feat_size % num_heads == 0
        ), "feat_size must be divisible by num_heads"
        self.dot_dim = feat_size // num_heads
        self.mha_ln_h = nn.LayerNorm(feat_size)
        self.mha_ln_e = nn.LayerNorm(edge_feat_size)
        self.edge_input = nn.Linear(edge_feat_size, num_heads)
        self.qkv_proj = nn.Linear(feat_size, feat_size * 3)
        self.gate = nn.Linear(edge_feat_size, num_heads)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.node_output = nn.Linear(feat_size, feat_size)
        self.mha_dropout_h = nn.Dropout(dropout)

        self.node_ffn = nn.Sequential(
            nn.LayerNorm(feat_size),
            nn.Linear(feat_size, feat_size),
            activation,
            nn.Linear(feat_size, feat_size),
            nn.Dropout(dropout),
        )

        if self.edge_update:
            self.edge_output = nn.Linear(num_heads, edge_feat_size)
            self.mha_dropout_e = nn.Dropout(dropout)
            self.edge_ffn = nn.Sequential(
                nn.LayerNorm(edge_feat_size),
                nn.Linear(edge_feat_size, edge_feat_size),
                activation,
                nn.Linear(edge_feat_size, edge_feat_size),
                nn.Dropout(dropout),
            )

    def forward(self, nfeat, efeat, mask=None):
        """Forward computation. Note: :attr:`nfeat` and :attr:`efeat` should be
        padded with embedding of virtual nodes if :attr:`num_virtual_nodes` > 0,
        while :attr:`mask` should be padded with `0` values for virtual nodes.
        The padding should be put at the beginning.

        Parameters
        ----------
        nfeat : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where N
            is the sum of the maximum number of nodes and the number of virtual nodes.
        efeat : torch.Tensor
            Edge embedding used for attention computation and self update.
            Shape: (batch_size, N, N, :attr:`edge_feat_size`).
        mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where valid positions are indicated by `0` and
            invalid positions are indicated by `-inf`.
            Shape: (batch_size, N, N). Default: None.

        Returns
        -------
        nfeat : torch.Tensor
            The output node embedding. Shape: (batch_size, N, :attr:`feat_size`).
        efeat : torch.Tensor, optional
            The output edge embedding. Shape: (batch_size, N, N, :attr:`edge_feat_size`).
            It is returned only if :attr:`edge_update` is True.
        """
        nfeat_r1 = nfeat
        efeat_r1 = efeat

        nfeat_ln = self.mha_ln_h(nfeat)
        efeat_ln = self.mha_ln_e(efeat)
        qkv = self.qkv_proj(nfeat_ln)
        e_bias = self.edge_input(efeat_ln)
        gates = self.gate(efeat_ln)
        bsz, N, _ = qkv.shape
        q_h, k_h, v_h = qkv.view(bsz, N, -1, self.num_heads).split(
            self.dot_dim, dim=2
        )
        attn_hat = torch.einsum("bldh,bmdh->blmh", q_h, k_h)
        attn_hat = attn_hat.clamp(-5, 5) + e_bias

        if mask is None:
            gates = torch.sigmoid(gates)
            attn_tild = F.softmax(attn_hat, dim=2) * gates
        else:
            gates = torch.sigmoid(gates + mask.unsqueeze(-1))
            attn_tild = F.softmax(attn_hat + mask.unsqueeze(-1), dim=2) * gates

        attn_tild = self.attn_dropout(attn_tild)
        v_attn = torch.einsum("blmh,bmkh->blkh", attn_tild, v_h)

        # Scale the aggregated values by degree.
        degrees = torch.sum(gates, dim=2, keepdim=True)
        degree_scalers = torch.log(1 + degrees)
        degree_scalers[:, : self.num_virtual_nodes] = 1.0
        v_attn = v_attn * degree_scalers

        v_attn = v_attn.reshape(bsz, N, self.num_heads * self.dot_dim)
        nfeat = self.node_output(v_attn)

        nfeat = self.mha_dropout_h(nfeat)
        nfeat.add_(nfeat_r1)
        nfeat_r2 = nfeat
        nfeat = self.node_ffn(nfeat)
        nfeat.add_(nfeat_r2)

        if self.edge_update:
            efeat = self.edge_output(attn_hat)
            efeat = self.mha_dropout_e(efeat)
            efeat.add_(efeat_r1)
            efeat_r2 = efeat
            efeat = self.edge_ffn(efeat)
            efeat.add_(efeat_r2)

            return nfeat, efeat

        return nfeat


class VirtualNodes(nn.Module):
    """
    Generate node and edge features for virtual nodes in the graph
    and pad the corresponding matrices.
    """

    def __init__(self, feat_size, edge_feat_size, num_virtual_nodes = 1):
        super().__init__()
        self.feat_size = feat_size
        self.edge_feat_size = edge_feat_size
        self.num_virtual_nodes = num_virtual_nodes

        self.vn_node_embeddings = nn.Parameter(torch.empty(num_virtual_nodes,
                                                           self.feat_size))
        self.vn_edge_embeddings = nn.Parameter(torch.empty(num_virtual_nodes,
                                                           self.edge_feat_size))
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
    """The EGT model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    EGTForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in EGTForGraphClassification.
    """
    def __init__(self, config: EGTConfig):
        super().__init__(config)

        self.layer_common_kwargs = dict(
             feat_size          = config.feat_size            ,
             edge_feat_size          = config.edge_feat_size           ,
             num_heads           = config.num_heads             ,
             num_virtual_nodes  = config.num_virtual_nodes,
             dropout    = config.dropout      ,
             attn_dropout        = config.attn_dropout          ,
             activation          = config.activation            ,
        )        
        self.edge_update = not config.egt_simple
        
        self.EGT_layers = nn.ModuleList([EGTLayer(**self.layer_common_kwargs, 
                                                   edge_update=self.edge_update)
                                         for _ in range(config.num_layers-1)])
    
        self.EGT_layers.append(EGTLayer(**self.layer_common_kwargs, edge_update = False))

        self.upto_hop          = config.upto_hop
        self.num_virtual_nodes = config.num_virtual_nodes
        self.svd_pe_size     = config.svd_pe_size
        
        self.nodef_embed = nn.Embedding(NUM_NODE_FEATURES*NODE_FEATURES_OFFSET+1,
                                        config.feat_size, padding_idx=0)
        if self.svd_pe_size:
            self.svd_embed = nn.Linear(self.svd_pe_size*2, config.feat_size)
        
        self.dist_embed = nn.Embedding(self.upto_hop+2, config.edge_feat_size)
        self.featm_embed = nn.Embedding(NUM_EDGE_FEATURES*EDGE_FEATURES_OFFSET+1,
                                        config.edge_feat_size, padding_idx=0)
        
        if self.num_virtual_nodes > 0:
            self.vn_layer = VirtualNodes(config.feat_size, config.edge_feat_size, 
                                         self.num_virtual_nodes)
        
        self.final_ln_h = nn.LayerNorm(config.feat_size)
        mlp_dims = [config.feat_size * max(self.num_virtual_nodes, 1)]\
                    +[round(config.feat_size*r) for r in config.mlp_ratios]\
                        +[config.num_classes]
        self.mlp_layers = nn.ModuleList([nn.Linear(mlp_dims[i],mlp_dims[i+1])
                                         for i in range(len(mlp_dims)-1)])
        self.mlp_fn = config.activation
     
    def input_block(self, nodef, featm, dm, nodem, svd_pe):
        dm = dm.long().clamp(min=0, max=self.upto_hop+1)  # (b,i,j)
        
        h = self.nodef_embed(nodef).sum(dim=2)      # (b,i,w,h) -> (b,i,h)
        
        if self.svd_pe_size:
            h = h + self.svd_embed(svd_pe)
        
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
        svd_pe: torch.Tensor,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> torch.Tensor:
        h, e, mask = self.input_block(node_feat, featm, dm, attn_mask, svd_pe)

        for layer in self.EGT_layers[:-1]:
            if self.edge_update:
                h, e = layer(h, e, mask)
            else:
                h = layer(h, e, mask)

        h = self.EGT_layers[-1](h, e, mask)
        
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
        self.num_classes = config.num_classes

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
