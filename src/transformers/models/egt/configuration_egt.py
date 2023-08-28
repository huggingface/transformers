from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

EGT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # pcqm4mv1 now deprecated
    "graphormer-base": "https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/main/config.json",
    # See all Graphormer models at https://huggingface.co/models?filter=graphormer
}


class EGTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~EGTModel`]. It is used to instantiate an EGT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the EGT
    [graphormer-base-pcqm4mv1](https://huggingface.co/graphormer-base-pcqm4mv1) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_atoms (`int`, *optional*, defaults to 9):
            Number of node types in the graphs.
        num_edges (`int`, *optional*, defaults to 3):
            Number of edges types in the graphs.
        feat_size (`int`, *optional*, defaults to 768):
            Node feature size.
        edge_feat_size (`int`, *optional*, defaults to 64):
            Edge feature size.
        num_heads (`int`, *optional*, defaults to 32):
            Number of attention heads, by which :attr: `feat_size` is divisible.
        num_layers (`int`, *optional*, defaults to 30):
            Number of layers.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability.
        attn_dropout (`float`, *optional*, defaults to 0.3):
            Attention dropout probability.
        activation (`str`, *optional*, defaults to 'ELU'):
            Activation function.
        egt_simple (`bool`, *optional*, defaults to False):
            If `False`, update the edge embedding.
        upto_hop (`int`, *optional*, defaults to 16):
            Maximum distance between nodes in the distance matrices.
        mlp_ratios (`List[float]`, *optional*, defaults to [1., 1.]):
            Ratios of inner dimensions with respect to the input dimension in MLP output block.
        num_virtual_nodes (`int`, *optional*, defaults to 4):
            Number of virtual nodes in EGT model, aggregated to graph embedding in the readout function.
        svd_pe_size (`int`, *optional*, defaults to 8):
            SVD positional encoding size.
        num_classes (`int`, *optional*, defaults to 1):
            Number of target classes or labels, set to n for binary classification of n tasks.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        traceable (`bool`, *optional*, defaults to `False`):
            Changes return value of the encoder's inner_state to stacked tensors.

        Example:
            ```python
            >>> from transformers import EGTForGraphClassification, EGTConfig  # doctest: +SKIP

            >>> # Initializing a EGT graphormer-base-pcqm4mv2 style configuration
            >>> configuration = EGTConfig()  # doctest: +SKIP

            >>> # Initializing a model from the graphormer-base-pcqm4mv1 style configuration
            >>> model = EGTForGraphClassification(configuration)  # doctest: +SKIP

            >>> # Accessing the model configuration
            >>> configuration = model.config  # doctest: +SKIP
            ```
    """
    model_type = "egt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_atoms: int = 9,
        num_edges: int = 3,
        feat_size: int = 768,
        edge_feat_size: int = 64,
        num_heads: int = 32,
        num_layers: int = 30,
        dropout: float = 0.0,
        attn_dropout: float = 0.3,
        activation: str = "ELU",
        egt_simple: bool = False,
        upto_hop: int = 16,
        mlp_ratios: List[float] = [1.0, 1.0],
        num_virtual_nodes: int = 4,
        svd_pe_size: int = 8,
        num_classes: int = 1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.num_atoms = num_atoms
        self.num_edges = num_edges
        self.feat_size = feat_size
        self.edge_feat_size = edge_feat_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.activation = activation
        self.egt_simple = egt_simple
        self.upto_hop = upto_hop
        self.mlp_ratios = mlp_ratios
        self.num_virtual_nodes = num_virtual_nodes
        self.svd_pe_size = svd_pe_size
        self.num_classes = num_classes

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
