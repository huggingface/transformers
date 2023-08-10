# from ...configuration_utils import PretrainedConfig
# from ...utils import logging
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import List
import torch.nn as nn


logger = logging.get_logger(__name__)

# GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     # pcqm4mv1 now deprecated
#     "graphormer-base": "https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/main/config.json",
#     # See all Graphormer models at https://huggingface.co/models?filter=graphormer
# }


class EGTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~EGTModel`]. It is used to instantiate an
    EGT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the EGT
    [graphormer-base-pcqm4mv1](https://huggingface.co/graphormer-base-pcqm4mv1) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_classes (`int`, *optional*, defaults to 1):
            Number of target classes or labels, set to n for binary classification of n tasks.
        num_atoms (`int`, *optional*, defaults to 512*9):
            Number of node types in the graphs.
        num_edges (`int`, *optional*, defaults to 512*3):
            Number of edges types in the graph.
        num_in_degree (`int`, *optional*, defaults to 512):
            Number of in degrees types in the input graphs.
        num_out_degree (`int`, *optional*, defaults to 512):
            Number of out degrees types in the input graphs.
        num_edge_dis (`int`, *optional*, defaults to 128):
            Number of edge dis in the input graphs.
        multi_hop_max_dist (`int`, *optional*, defaults to 20):
            Maximum distance of multi hop edges between two nodes.
        spatial_pos_max (`int`, *optional*, defaults to 1024):
            Maximum distance between nodes in the graph attention bias matrices, used during preprocessing and
            collation.
        edge_type (`str`, *optional*, defaults to multihop):
            Type of edge relation chosen.
        max_nodes (`int`, *optional*, defaults to 512):
            Maximum number of nodes which can be parsed for the input graphs.
        share_input_output_embed (`bool`, *optional*, defaults to `False`):
            Shares the embedding layer between encoder and decoder - careful, True is not implemented.
        num_layers (`int`, *optional*, defaults to 12):
            Number of layers.
        embedding_dim (`int`, *optional*, defaults to 768):
            Dimension of the embedding layer in encoder.
        ffn_embedding_dim (`int`, *optional*, defaults to 768):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in the encoder.
        self_attention (`bool`, *optional*, defaults to `True`):
            Model is self attentive (False not implemented).
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention weights.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        bias (`bool`, *optional*, defaults to `True`):
            Uses bias in the attention module - unsupported at the moment.
        embed_scale(`float`, *optional*, defaults to None):
            Scaling factor for the node embeddings.
        num_trans_layers_to_freeze (`int`, *optional*, defaults to 0):
            Number of transformer layers to freeze.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Normalize features before encoding the graph.
        pre_layernorm (`bool`, *optional*, defaults to `False`):
            Apply layernorm before self attention and the feed forward network. Without this, post layernorm will be
            used.
        apply_graphormer_init (`bool`, *optional*, defaults to `False`):
            Apply a custom graphormer initialisation to the model before training.
        freeze_embeddings (`bool`, *optional*, defaults to `False`):
            Freeze the embedding layer, or train it along the model.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Apply the layer norm before each encoder block.
        q_noise (`float`, *optional*, defaults to 0.0):
            Amount of quantization noise (see "Training with Quantization Noise for Extreme Model Compression"). (For
            more detail, see fairseq's documentation on quant_noise).
        qn_block_size (`int`, *optional*, defaults to 8):
            Size of the blocks for subsequent quantization with iPQ (see q_noise).
        kdim (`int`, *optional*, defaults to None):
            Dimension of the key in the attention, if different from the other values.
        vdim (`int`, *optional*, defaults to None):
            Dimension of the value in the attention, if different from the other values.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        traceable (`bool`, *optional*, defaults to `False`):
            Changes return value of the encoder's inner_state to stacked tensors.

        Example:
            ```python
            >>> from transformers import EGTForGraphClassification, EGTConfig

            >>> # Initializing a EGT graphormer-base-pcqm4mv2 style configuration
            >>> configuration = EGTConfig()

            >>> # Initializing a model from the graphormer-base-pcqm4mv1 style configuration
            >>> model = EGTForGraphClassification(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
            ```
    """
    model_type = "egt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        node_width: int = 128,
        edge_width: int = 32,
        num_heads: int = 8,
        model_height: int = 4,
        dropout: float = 0.,
        attn_dropout: float = 0.,
        activation: nn.Module = nn.ELU(),
        ffn_multiplier: float = 1.,
        egt_simple: bool = False,
        upto_hop: int = 16,
        mlp_ratios: List[float] = [1., 1.],
        num_virtual_nodes: int = 4,
        svd_encodings: int = 8,
        output_dim: int = 1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_heads = num_heads
        self.model_height = model_height
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.activation = activation
        self.ffn_multiplier = ffn_multiplier
        self.egt_simple = egt_simple
        self.upto_hop = upto_hop
        self.mlp_ratios = mlp_ratios
        self.num_virtual_nodes = num_virtual_nodes
        self.svd_encodings = svd_encodings
        self.output_dim = output_dim

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
