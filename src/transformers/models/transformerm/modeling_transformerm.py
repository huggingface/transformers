import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import (
    GraphormerLayer,
    DegreeEncoder,
    SpatialEncoder,
)


def gaussian(x, mean, std):
    """compute gaussian basis kernel function"""
    const_pi = 3.14159
    a = (2 * const_pi) ** 0.5
    return th.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


"""Temporarily define spatialencoder3d here for api change in DGL"""
class SpatialEncoder3d(nn.Module):
    r"""3D Spatial Encoder, as introduced in
    `One Transformer Can Understand Both 2D & 3D Molecular Data
    <https://arxiv.org/pdf/2210.01765.pdf>`__

    This module encodes pair-wise relation between atom pair :math:`(i,j)` in
    the 3D geometric space, according to the Gaussian Basis Kernel function:

    :math:`\psi _{(i,j)} ^k = \frac{1}{\sqrt{2\pi} \lvert \sigma^k \rvert}
    \exp{\left ( -\frac{1}{2} \left( \frac{\gamma_{(i,j)} \lvert \lvert r_i -
    r_j \rvert \rvert + \beta_{(i,j)} - \mu^k}{\lvert \sigma^k \rvert} \right)
    ^2 \right)}，k=1,...,K,`

    where :math:`K` is the number of Gaussian Basis kernels.
    :math:`r_i` is the Cartesian coordinate of atom :math:`i`.
    :math:`\gamma_{(i,j)}, \beta_{(i,j)}` are learnable scaling factors of
    the Gaussian Basis kernels.

    Parameters
    ----------
    num_kernels : int
        Number of Gaussian Basis Kernels to be applied. Each Gaussian Basis 
        Kernel contains a learnable kernel center and a learnable scaling 
        factor.
    num_heads : int, optional
        Number of attention heads if multi-head attention mechanism is applied.
        Default : 1.
    max_node_type : int, optional
        Maximum number of node types. Default : 100.

    Examples
    --------
    >>> import torch as th
    >>> import dgl
    >>> from dgl.nn import SpatialEncoder3d

    >>> coordinate = th.rand(1, 4, 3)
    >>> node_type = th.tensor([[1, 0, 2, 1]])
    >>> spatial_encoder = SpatialEncoder3d(num_kernels=4,
    ...                                    num_heads=8,
    ...                                    max_node_type=3)
    >>> out = spatial_encoder(coordinate, node_type=node_type)
    >>> print(out.shape)
    torch.Size([1, 4, 4, 8])
    """

    def __init__(self, num_kernels, num_heads=1, max_node_type=100):
        super().__init__()
        self.num_kernels = num_kernels
        self.num_heads = num_heads
        self.max_node_type = max_node_type
        self.means = nn.Parameter(th.empty(num_kernels))
        self.stds = nn.Parameter(th.empty(num_kernels))
        self.linear_layer_1 = nn.Linear(num_kernels, num_kernels)
        self.linear_layer_2 = nn.Linear(num_kernels, num_heads)
        # There are 2 * max_node_type + 3 pairs of gamma and beta parameters:
        # 1. Parameters at position 0 are for default gamma/beta when no node
        #    type is given
        # 2. Parameters at position 1 to max_node_type+1 are for src node types.
        #    (position 1 is for padded unexisting nodes)
        # 3. Parameters at position max_node_type+2 to 2*max_node_type+2 are
        #    for tgt node types. (position max_node_type+2 is for padded)
        #    unexisting nodes)
        self.gamma = nn.Embedding(2 * max_node_type + 3, 1, padding_idx=0)
        self.beta = nn.Embedding(2 * max_node_type + 3, 1, padding_idx=0)

        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.stds, 0, 3)
        nn.init.constant_(self.gamma.weight, 1)
        nn.init.constant_(self.beta.weight, 0)

    def forward(self, coord, node_type=None):
        """
        Parameters
        ----------
        coord : torch.Tensor
            3D coordinates of nodes in shape :math:`(B, N, 3)`, where :math:`B`
            is the batch size, :math:`N`: is the maximum number of nodes.
        node_type : torch.Tensor, optional
            Node type ids of nodes. Default : None.

            * If specified, :attr:`node_type` should be a tensor in shape
              :math:`(B, N,)`. The scaling factors in gaussian kernels of each
              pair of nodes are determined by their node types.
            * Otherwise, :attr:`node_type` will be set to zeros of the same
              shape by default.

        Returns
        -------
        torch.Tensor
            Return attention bias as 3D spatial encoding of shape
            :math:`(B, N, N, H)`, where :math:`H` is :attr:`num_heads`.
        """
        bsz, N = coord.shape[:2]
        euc_dist = th.cdist(coord, coord, p=2.0)  # shape: [B, n, n]
        if node_type is None:
            node_type = th.zeros([bsz, N, N, 2], device=coord.device).long()
        else:
            src_node_type = node_type.unsqueeze(-1).repeat(1, 1, N)
            tgt_node_type = node_type.unsqueeze(1).repeat(1, N, 1)
            node_type = th.stack(
                [src_node_type + 2, tgt_node_type + self.max_node_type + 3],
                dim=-1,
            )  # shape: [B, n, n, 2]

        # scaled euclidean distance
        gamma = self.gamma(node_type).sum(dim=-2)  # shape: [B, n, n, 1]
        beta = self.beta(node_type).sum(dim=-2)  # shape: [B, n, n, 1]
        euc_dist = gamma * euc_dist.unsqueeze(-1) + beta  # shape: [B, n, n, 1]
        # gaussian basis kernel
        euc_dist = euc_dist.expand(-1, -1, -1, self.num_kernels)
        gaussian_kernel = gaussian(
            euc_dist, self.means, self.stds.abs() + 1e-2
        )  # shape: [B, n, n, K]
        # linear projection
        encoding = self.linear_layer_1(gaussian_kernel)
        encoding = F.gelu(encoding)
        encoding = self.linear_layer_2(encoding)  # shape: [B, n, n, H]

        return encoding


class PathEncoder(nn.Module):
    def __init__(self, max_len, num_edges, feat_dim, num_heads=1):
        super().__init__()
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.edge_emb = nn.Embedding(num_edges + 1, feat_dim)
        self.attn_map_weights = nn.Parameter(th.randn(max_len, feat_dim, num_heads))
    
    def forward(self, dist, path_data):
        """
        dist in [B, N, N] th.long
        path_data in [B, N, N, L, d] th.long
        return [B, N, N, H] th.float
        """
        # assume the path data has been truncated to max_len
        max_len = path_data.size(-2)
        assert max_len <= self.max_len, f"Input max length {max_len} exceeds" \
            f"the maximum {self.max_len}."
        path_data = self.edge_emb(path_data).mean(dim=-2) # [B, N, N, L, d']
        shortest_distance = th.clamp(dist, min=1, max=max_len)
        weight = self.attn_map_weights[:max_len]
        path_encoding = th.einsum("bxyld,ldh->bxylh", path_data, weight)
        path_encoding = path_encoding.sum(-2) / dist.unsqueeze(-1)
        
        return path_encoding


def get_activation(activation="relu"):
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    return ValueError("Activation function {} not supported".format(activation))


class TransformerM(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 1 for padding
        self.atom_encoder = nn.Embedding(
            args.num_atoms + 1, args.embed_dim, padding_idx=0
        )
        # one additional embedding for the virtual graph token
        self.graph_token_emb = nn.Parameter(torch.randn(args.embed_dim))

        # degree encoder
        self.degree_encoder = DegreeEncoder(
            max_degree=args.max_degree,
            embedding_dim=args.embed_dim
        )

        # spatial encoder
        self.spatial_encoder = SpatialEncoder(
            max_dist=args.num_spatial,
            num_heads=args.num_heads
        )
        # embedding for the virtual graph token distance
        self.graph_token_virtual_dist = nn.Parameter(
            torch.randn(args.num_heads)
        )

        # path encoder
        self.path_encoder = PathEncoder(
            max_len=args.max_path_len,
            num_edges=args.num_edges,
            feat_dim=args.edge_dim,
            num_heads=args.num_heads,
        )

        # spatial encoder 3d
        """
         default 0
         src 1 (pad), 2, ..., 512
         tgt 513 (pad), 514, ..., 1024
        """
        self.spatial_encoder_3d = SpatialEncoder3d(
            num_kernels=args.num_kernels,
            num_heads=args.num_heads,
            max_node_type=args.max_node_type,
        )

        # transformer layers
        self.encoder = nn.ModuleList([
            GraphormerLayer(
                feat_size=args.embed_dim,
                hidden_size=args.ffn_embed_dim,
                num_heads=args.num_heads,
                dropout=args.dropout,
                attn_dropout=args.attn_dropout,
                activation=get_activation(args.activation),
            ) for _ in range(args.num_layers)
        ])


        self.post_process = None
        self.mlp = nn.Sequential(
            nn.Linear(args.embed_dim, args.embed_dim),
            get_activation(args.activation),
            nn.LayerNorm(args.embed_dim),
            nn.Linear(args.embed_dim, args.num_classes),
        )

    def forward(self, data_dict):

        return None


class TransformerMGraphEncoder(nn.Module):
    def __init__(self, config: TransformerMConfig):
        super().__init__()

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.layerdrop = config.layerdrop
        self.embed_dim = config.embed_dim
        self.apply_transformerm_init = config.apply_transformerm_init
        self.traceable = config.traceable

        self.embed_scale = config.embed_scale

        if config.q_noise > 0:
            self.quant_noise = quant_noise(
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                config.q_noise,
                config.qn_block_size,
            )
        else:
            self.quant_noise = None
        
        # 1 for padding
        self.atom_encoder = nn.Embedding(
            args.num_atoms + 1, args.embed_dim, padding_idx=0
        )
        # one additional embedding for the virtual graph token
        self.graph_token_emb = nn.Parameter(torch.randn(args.embed_dim))

        # degree encoder
        self.degree_encoder = DegreeEncoder(
            max_degree=args.max_degree,
            embedding_dim=args.embed_dim
        )

        # spatial encoder
        self.spatial_encoder = SpatialEncoder(
            max_dist=args.num_spatial,
            num_heads=args.num_heads
        )
        # embedding for the virtual graph token distance
        self.graph_token_virtual_dist = nn.Parameter(
            torch.randn(args.num_heads)
        )

        # path encoder
        self.path_encoder = PathEncoder(
            max_len=args.max_path_len,
            num_edges=args.num_edges,
            feat_dim=args.edge_dim,
            num_heads=args.num_heads,
        )

        # spatial encoder 3d
        """
         default 0
         src 1 (pad), 2, ..., 512
         tgt 513 (pad), 514, ..., 1024
        """
        self.spatial_encoder_3d = SpatialEncoder3d(
            num_kernels=args.num_kernels,
            num_heads=args.num_heads,
            max_node_type=args.max_node_type,
        )

        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([GraphormerLayer(
                feat_size=config.embed_dim,
                hidden_size=config.ffn_embed_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                activation=get_activation(config.activation),
            ) for _ in range(config.num_hidden_layers)
        ])

        # Apply initialization of model params after building the model
        if config.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(config.num_trans_layers_to_freeze):
            m = self.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(
        self,
        node_feat: torch.LongTensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        dist: torch.LongTensor,
        path_data: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        node_type: torch.LongTensor,
        last_state_only: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.LongTensor]], torch.Tensor]:
        # encode input node feature
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(th.stack((in_degree, out_degree)))
        node_feat = self.atom_encoder(node_feat)
        node_feat = node_feat + deg_emb
        graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
            num_graphs, 1, 1
        )
        x = th.cat([graph_token_feat, node_feat], dim=1)

        attn_bias = th.zeros(
            num_graphs,
            max_num_nodes + 1,
            max_num_nodes + 1,
            self.num_heads,
            device=dist.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        spatial3d_encoding = self.spatial_encoder_3d(spatial_pos, node_type)

        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding + \
            spatial3d_encoding

        # spatial encoding of the virtual node
        t = self.graph_token_virtual_dist.weight.reshape(1, 1, self.num_heads)
        # Since the virtual node comes first, the spatial encodings between it
        # and other nodes will fill the 1st row and 1st column (omit num_graphs
        # and num_heads dimensions) of attn_bias matrix by broadcasting.
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(x)

        graph_rep = x[:, 0, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep


class TransformerMDecoderHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        """num_classes should be 1 for regression, or the number of classes for classification"""
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.num_classes = num_classes

    def forward(self, input_nodes: torch.Tensor, **unused) -> torch.Tensor:
        input_nodes = self.classifier(input_nodes)
        input_nodes = input_nodes + self.lm_output_learned_bias
        return input_nodes


class TransformerMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TransformerMConfig
    base_model_prefix = "transformerm"
    supports_gradient_checkpointing = True
    main_input_name_nodes = "input_nodes"
    main_input_name_edges = "input_edges"

    def normal_(self, data: torch.Tensor):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    def init_transformerm_params(self, module: Union[nn.Linear, nn.Embedding, TransformerMMultiheadAttention]):
        """
        Initialize the weights of the Transformer-M Model.
        """
        if isinstance(module, nn.Linear):
            self.normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            self.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            self.normal_(module.q_proj.weight.data)
            self.normal_(module.k_proj.weight.data)
            self.normal_(module.v_proj.weight.data)

    def _init_weights(
        self,
        module: Union[
            nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm,
            TransformerMGraphEncoder
        ],
    ):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # update the initialization part
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # elif isinstance(module, GraphormerMultiheadAttention):
        #     module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        #     module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        #     module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        #     module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module,TransformerMGraphEncoder):
            if module.apply_graphormer_init:
                module.apply(self.init_transformerm_params)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GraphormerModel):
            module.gradient_checkpointing = value


class TransformerMForGraphClassification(TransformerMPreTrainedModel):
    """
    This model can be used for graph-level classification or regression tasks.
    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one
      float-type label per graph
    - one task classification (by setting config.num_classes to the number of
      classes); there should be one integer label per graph
    - binary multi-task classification (by setting config.num_classes to the
      number of labels); there should be a list of integer labels for each graph.
    """

    def __init__(self, config):
        super().__init__(config)
        self.encoder = TransformerMModel(config)
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.classifier = TransformerMDecoderHead(self.embedding_dim, self.num_classes)
        self.is_encoder_decoder = True

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        node_feat: torch.LongTensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        dist: torch.LongTensor,
        path_data: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        node_type: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        encoder_outputs = self.encoder(
            node_feat,
            in_degree,
            out_degree,
            dist,
            path_data,
            spatial_pos,
            node_type,
            attn_mask = attn_mask,
        )
        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]

        head_outputs = self.classifier(outputs)
        logits = head_outputs[:, 0, :].contiguous()

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
            return (loss, logits, hidden_states)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=None)
