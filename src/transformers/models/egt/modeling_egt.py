""" PyTorch EGT model."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import is_dgl_available, requires_backends
from .configuration_egt import EGTConfig


if is_dgl_available():
    from dgl.nn import EGTLayer


NODE_FEATURES_OFFSET = 128
EDGE_FEATURES_OFFSET = 8


_CHECKPOINT_FOR_DOC = "dgl-egt"
_CONFIG_FOR_DOC = "EGTConfig"


EGT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Zhiteng/dgl-egt",
]


class VirtualNodes(nn.Module):
    """
    Generate node and edge features for virtual nodes in the graph and pad the corresponding matrices.
    """

    def __init__(self, feat_size, edge_feat_size, num_virtual_nodes=1):
        super().__init__()
        self.feat_size = feat_size
        self.edge_feat_size = edge_feat_size
        self.num_virtual_nodes = num_virtual_nodes

        self.vn_node_embeddings = nn.Parameter(torch.empty(num_virtual_nodes, self.feat_size))
        self.vn_edge_embeddings = nn.Parameter(torch.empty(num_virtual_nodes, self.edge_feat_size))
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
            mask = F.pad(mask, (self.num_virtual_nodes, 0, self.num_virtual_nodes, 0), mode="constant", value=0)
        return h, e, mask


class EGTPreTrainedModel(PreTrainedModel):
    """
    A simple interface for downloading and loading pretrained models.
    """

    config_class = EGTConfig
    base_model_prefix = "egt"
    supports_gradient_checkpointing = True
    main_input_name_nodes = "node_feat"
    main_input_name_edges = "featm"

    def __init__(self, config, **kwargs):
        requires_backends(self, "dgl")
        super().__init__(config)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, EGTModel):
            module.gradient_checkpointing = value


class EGTModel(EGTPreTrainedModel):
    """The EGT model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    EGTForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine this
    model with a downstream model of your choice, following the example in EGTForGraphClassification.
    """

    def __init__(self, config: EGTConfig):
        super().__init__(config)

        self.activation = getattr(nn, config.activation)()

        self.layer_common_kwargs = {
            "feat_size": config.feat_size,
            "edge_feat_size": config.edge_feat_size,
            "num_heads": config.num_heads,
            "num_virtual_nodes": config.num_virtual_nodes,
            "dropout": config.dropout,
            "attn_dropout": config.attn_dropout,
            "activation": self.activation,
        }
        self.edge_update = not config.egt_simple

        self.EGT_layers = nn.ModuleList(
            [EGTLayer(**self.layer_common_kwargs, edge_update=self.edge_update) for _ in range(config.num_layers - 1)]
        )

        self.EGT_layers.append(EGTLayer(**self.layer_common_kwargs, edge_update=False))

        self.upto_hop = config.upto_hop
        self.num_virtual_nodes = config.num_virtual_nodes
        self.svd_pe_size = config.svd_pe_size

        self.nodef_embed = nn.Embedding(config.num_atoms * NODE_FEATURES_OFFSET + 1, config.feat_size, padding_idx=0)
        if self.svd_pe_size:
            self.svd_embed = nn.Linear(self.svd_pe_size * 2, config.feat_size)

        self.dist_embed = nn.Embedding(self.upto_hop + 2, config.edge_feat_size)
        self.featm_embed = nn.Embedding(
            config.num_edges * EDGE_FEATURES_OFFSET + 1, config.edge_feat_size, padding_idx=0
        )

        if self.num_virtual_nodes > 0:
            self.vn_layer = VirtualNodes(config.feat_size, config.edge_feat_size, self.num_virtual_nodes)

        self.final_ln_h = nn.LayerNorm(config.feat_size)
        mlp_dims = (
            [config.feat_size * max(self.num_virtual_nodes, 1)]
            + [round(config.feat_size * r) for r in config.mlp_ratios]
            + [config.num_classes]
        )
        self.mlp_layers = nn.ModuleList([nn.Linear(mlp_dims[i], mlp_dims[i + 1]) for i in range(len(mlp_dims) - 1)])
        self.mlp_fn = self.activation

        self._backward_compatibility_gradient_checkpointing()

    def input_block(self, nodef, featm, dm, nodem, svd_pe):
        dm = dm.long().clamp(min=0, max=self.upto_hop + 1)  # (b,i,j)

        h = self.nodef_embed(nodef).sum(dim=2)  # (b,i,w,h) -> (b,i,h)

        if self.svd_pe_size:
            h = h + self.svd_embed(svd_pe)

        e = self.dist_embed(dm) + self.featm_embed(featm).sum(dim=3)  # (b,i,j,f,e) -> (b,i,j,e)

        mask = (nodem[:, :, None] * nodem[:, None, :] - 1) * 1e9

        if self.num_virtual_nodes > 0:
            h, e, mask = self.vn_layer(h, e, mask)
        return h, e, mask

    def final_embedding(self, h, attn_mask):
        h = self.final_ln_h(h)
        if self.num_virtual_nodes > 0:
            h = h[:, : self.num_virtual_nodes].reshape(h.shape[0], -1)
        else:
            nodem = attn_mask.float().unsqueeze(dim=-1)
            h = (h * nodem).sum(dim=1) / (nodem.sum(dim=1) + 1e-9)
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        self._backward_compatibility_gradient_checkpointing()

    def forward(
        self,
        node_feat: torch.LongTensor,
        featm: torch.LongTensor,
        dm: torch.LongTensor,
        attn_mask: torch.LongTensor,
        svd_pe: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits = self.model(
            node_feat,
            featm,
            dm,
            attn_mask,
            svd_pe,
            return_dict=True,
        )["last_hidden_state"]

        loss = None
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
