# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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

"""Ernie model configuration"""
import copy
from typing import Union

from transformers import PretrainedConfig

from ...modeling_rope_utils import rope_config_validation


__all__ = [
    "ERNIE_PRETRAINED_INIT_CONFIGURATION",
    "Ernie4_5_Config",
    "Ernie4_5_MoEConfig",
    "Ernie4_5_VLMoEConfig",
]


class DFNRopeVisionTransformerConfig(PretrainedConfig):
    """
    Configuration class for DFNRopeVisionTransformer model.
    This class inherits from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "DFNRope_vision_transformer"
    base_model_tp_plan = {}

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        attn_implementation="eager",  # new added
        pp_data_balance=False,
        recompute=False,
        attn_sep=False,
        vit_first_fwd_bsz=128,
        vit_num_recompute_layers=10000,
        **kwargs,
    ):
        """
        Initialize DFNRopeVisionTransformer model configuration with default or specified parameters.

        Args:
            depth (int): Number of transformer layers in the model.
            embed_dim (int): Dimensionality of the embedding layer.
            hidden_size (int): Dimensionality of the feedforward network.
            hidden_act (str): Activation function for the feedforward network.
            mlp_ratio (float): Ratio between the number of input features and
                the number of output features in the feedforward network.
            num_heads (int): Number of attention heads in each attention layer.
            in_channels (int): Number of channels in the input image.
            patch_size (int):
                Size of patches in the input image. Defaults to 14.
            spatial_merge_size (int):
                Spatial merge size for the spatial transformer module. Defaults to 2.
            attn_implementation (str): Attention implementation type. Defaults to "eager".
            pp_data_balance (bool): Whether to balance data during preprocessing. Defaults to False.
            recompute (bool): Whether to use recompute. Defaults to False.
            attn_sep (bool): Whether to separate attention computation into two stages. Defaults to False.
            vit_first_fwd_bsz (int): First forward batch size for ViT. Defaults to 128.
            vit_num_recompute_layers (int): Number of recomputed layers for ViT. Defaults to
        """
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.attn_implementation = attn_implementation
        self.pp_data_balance = pp_data_balance
        self.recompute = recompute
        self.attn_sep = attn_sep
        self.vit_first_fwd_bsz = vit_first_fwd_bsz
        self.vit_num_recompute_layers = vit_num_recompute_layers

    def get(self, key, default=None):
        """get config value by key"""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default


ERNIE_PRETRAINED_INIT_CONFIGURATION = {
    "ernie/tiny-random-ernie": {
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "model_type": "ernie",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": False,
        "recompute": False,
        "use_flash_attn": True,
        "use_pure_fp16": False,
    },
}


class Ernie4_5_Config(PretrainedConfig):
    """
    Configuration class for ERNIE model.

    This class stores the configuration of an ERNIE model, defining the model architecture.
    It inherits from PretrainedConfig and can be used to control model outputs.
    """

    model_type = "ernie"
    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION
    base_model_tp_plan = {}

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=11008,
        max_position_embeddings=32768,
        num_hidden_layers=2,
        num_attention_heads=2,
        initializer_range=0.02,  # no use
        rms_norm_eps=1e-6,
        use_cache=False,
        use_flash_attention=True,
        use_sparse_flash_attn=True,
        use_var_len_flash_attn=False,
        recompute=False,
        recompute_granularity="core_attn",
        recompute_use_reentrant=False,
        use_rmsnorm=True,
        fuse_rms_norm=False,
        fuse_ln=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        fuse_swiglu=False,
        use_bias=False,
        rope_theta=10000,
        fuse_rope=False,
        fuse_softmax_mask=False,
        use_fast_ln=False,
        weight_share_add_bias=True,
        fuse_linear=False,
        max_sequence_length=1024,
        ignored_index=-100,
        add_tail_layers=False,
        use_recompute_lm_head=False,
        use_recompute_loss_fn=False,
        refined_recompute=dict(),
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        compression_ratio: float = 1.0,
        num_key_value_heads=None,
        use_sparse_head_and_loss_fn=False,
        micro_batch_size=-1,
        use_ep_comm_overlap=False,
        use_fused_head_and_loss_fn=False,
        token_balance_loss=False,
        token_balance_seqlen=False,  # calculated based on batchsize and seqlen
        cachekv_quant: bool = False,
        pp_seg_method="layer:ErnieDecoderLayer|EmptyLayer",
        **kwargs,
    ):
        """
        Initialize ERNIE model configuration with default or specified parameters.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens)
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer
            intermediate_size (int): Dimensionality of the "intermediate" (feed-forward) layer
            max_position_embeddings (int): Maximum sequence length the model can handle
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder
            num_attention_heads (int): Number of attention heads for each attention layer
            rms_norm_eps (float): The epsilon used by the RMS normalization layers
            use_cache (bool): Whether to use caching for faster generation (decoding)
            use_flash_attention (bool): Whether to use FlashAttention for optimized attention computation
            use_sparse_flash_attn (bool): Whether to use sparse FlashAttention
            use_var_len_flash_attn (bool): Whether to use variable-length FlashAttention
            recompute (bool): Whether to use gradient checkpointing to save memory
            recompute_granularity (str): Granularity of recomputation ("core_attn", "full", etc.)
            recompute_use_reentrant (bool): Whether to use reentrant checkpointing
            use_rmsnorm (bool): Whether to use RMSNorm instead of LayerNorm
            fuse_rms_norm (bool): Whether to fuse RMSNorm operations for optimization
            fuse_ln (bool): Whether to fuse LayerNorm operations
            pad_token_id (int): Token ID used for padding sequences
            bos_token_id (int): Token ID used for beginning-of-sequence
            eos_token_id (int): Token ID used for end-of-sequence
            fuse_swiglu (bool): Whether to fuse SwiGLU operations
            use_bias (bool): Whether to use bias terms in linear layers
            rope_theta (float): The base period of the RoPE embeddings
            fuse_rope (bool): Whether to fuse RoPE operations
            use_fast_ln (bool): Whether to use optimized LayerNorm implementation
            weight_share_add_bias (bool): Whether to share bias weights in certain layers
            fuse_linear (bool): Whether to fuse linear operations
            max_sequence_length (int): Maximum sequence length for positional embeddings
            ignored_index (int): Target value that is ignored during loss computation
            add_tail_layers (bool): Whether to add additional layers at the end
            use_recompute_lm_head (bool): Whether to recompute gradients for language model head
            use_recompute_loss_fn (bool): Whether to recompute gradients for loss function
            refined_recompute (dict): Dictionary specifying refined recomputation settings
            attention_probs_dropout_prob (float): Dropout probability for attention weights
            hidden_dropout_prob (float): Dropout probability for hidden layers
            compression_ratio (float): Ratio for KV cache compression (1.0 = no compression)
            num_key_value_heads (int): Number of key/value heads (for Grouped Query Attention)
            use_sparse_head_and_loss_fn (bool): Whether to use sparse attention head and loss function
            micro_batch_size (int): Size of micro batches (-1 for automatic)
            use_ep_comm_overlap (bool): Whether to overlap communication with computation
            use_fused_head_loss_fn (bool): Whether to use fused head and loss function
            token_balance_loss (bool): Whether to balance loss by token count
            token_balance_seqlen (bool): Whether to balance sequence lengths
            cachekv_quant (bool): Whether to quantize key-value cache
            pp_seg_method (str): Method for pipeline parallel segmentation
            **kwargs: Additional keyword arguments passed to parent class
        """

        # Set default for tied embeddings if not specified.
        if "tie_word_embeddings" not in kwargs:
            kwargs["tie_word_embeddings"] = False
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.recompute = recompute
        self.recompute_granularity = recompute_granularity
        self.use_flash_attention = use_flash_attention
        self.use_sparse_flash_attn = use_sparse_flash_attn
        self.recompute_use_reentrant = recompute_use_reentrant
        self.use_var_len_flash_attn = use_var_len_flash_attn
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.fuse_swiglu = fuse_swiglu
        self.fuse_rms_norm = fuse_rms_norm
        self.fuse_ln = fuse_ln
        self.use_rmsnorm = use_rmsnorm
        self.micro_batch_size = micro_batch_size

        self.max_sequence_length = max_sequence_length
        self.use_bias = use_bias
        self.weight_share_add_bias = weight_share_add_bias
        self.rope_theta = rope_theta
        self.fuse_rope = fuse_rope
        self.fuse_softmax_mask = fuse_softmax_mask
        self.use_fast_ln = use_fast_ln

        self.fuse_linear = fuse_linear
        self.ignored_index = ignored_index
        self.add_tail_layers = add_tail_layers
        self.use_recompute_lm_head = use_recompute_lm_head
        self.use_recompute_loss_fn = use_recompute_loss_fn

        self.refined_recompute = refined_recompute
        self.skip_recompute_ops = dict()
        """
            `refined_recompute` is a dictionary that specifies fine-grained gradient recomputation settings,
            which currently only takes effect in Pipeline Parallel (PP) mode.

            In PP mode, this dictionary populates `self.skip_recompute_ops` with the following structure:
            - Key (`op_name`): The operation name to configure, with possible values:
            * "mlp_row_ln" - MLP row-wise layer normalization
            * "flash_attn" - Flash attention operation
            * "attention_row_ln" - Attention row-wise layer normalization
            * "attention_column_ln" - Attention column-wise layer normalization
            * "mlp_column_ln" - MLP column-wise layer normalization

            - Value (`skip_num`): Controls how many times to skip recomputation:
            * 0: Never skip recomputation (minimum memory usage)
            * -1: Always skip recomputation (maximum memory usage)
            * [0,1,...,12]: Skip recomputation for specified number of times
            * ≥12: Equivalent to -1 (always skip recomputation)

            This allows precise control over memory/computation tradeoffs for different operations.
        """
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.compression_ratio = compression_ratio
        self.num_key_value_heads = num_key_value_heads
        self.use_sparse_head_and_loss_fn = use_sparse_head_and_loss_fn
        self.use_ep_comm_overlap = use_ep_comm_overlap
        self.use_fused_head_and_loss_fn = use_fused_head_and_loss_fn
        self.token_balance_loss = token_balance_loss
        self.token_balance_seqlen = token_balance_seqlen
        self.cachekv_quant = cachekv_quant
        self.pp_seg_method = pp_seg_method

    def get(self, key, default=None):
        """get config value by key"""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default


class Ernie4_5_MoEConfig(Ernie4_5_Config):
    r"""
    Configuration class for ErnieMoE model architecture.

    This class stores the configuration for a [`~ErnieModel`] and is used to instantiate
    an ErnieMoE model according to the specified arguments. Inherits from [`PretrainedConfig`]
    and can control model outputs.

    Attributes:
        Inherits all attributes from Ernie4_5_Config and adds MoE-specific configurations.
    """

    model_type = "ernie"
    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }
    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION
    base_model_tp_plan = {}

    def __init__(
        self,
        moe_num_experts: Union[int, list] = 0,
        use_recompute_moe=False,
        moe_capacity=(),
        moe_layer_interval=2,
        moe_layer_start_index=0,
        moe_layer_end_index=-1,
        moe_aux_loss_lambda=1e-2,
        moe_z_loss_lambda=1e-4,
        moe_orthogonal_loss_lambda=1e-2,
        sinkhorn_2gate=True,
        sinkhorn_temp=3e-2,
        global_aux_loss=False,
        moe_dropout_prob=0.0,
        moe_group="world",
        moe_gate="top2",
        moe_intermediate_size: Union[int, list] = 0,
        moe_num_shared_experts: int = 0,
        moe_reverse_token_drop: bool = False,
        moe_gate_act: str = "softmax",
        moe_norm_gate_logits=True,
        moe_all_to_all_dropout: float = 0.0,
        moe_k=2,
        moe_use_aux_free: bool = False,
        # `moe_group_experts` must be used with `moe_use_hard_gate=True`
        moe_group_experts: bool = False,
        moe_group_orthogonal_loss: bool = True,
        enable_delay_scale_loss: bool = True,
        num_acc_steps: int = 1,
        fuse_gate_detach_matmul: bool = False,
        dpo_config=None,
        moe_multimodal_dispatch_use_allgather: str = "",
        moe_use_hard_gate=False,
        moe_dense_experts_token_type_id=3,
        **kwargs,
    ):
        """
        Initialize ErnieMoE configuration with MoE-specific parameters.

        Args:
            moe_num_experts: Number of experts in MoE layers
            use_recompute_moe: Whether to use recomputation for MoE layers
            moe_capacity: Capacity configuration for MoE layers
            moe_layer_interval: Interval between MoE layers
            moe_layer_start_index: Starting layer index for MoE
            moe_layer_end_index: Ending layer index for MoE (-1 means last layer)
            moe_aux_loss_lambda: Weight for auxiliary loss
            moe_z_loss_lambda: Weight for z-loss
            moe_orthogonal_loss_lambda: Weight for orthogonal loss
            sinkhorn_2gate: Whether to use sinkhorn 2-gate routing
            sinkhorn_temp: Temperature for sinkhorn routing
            global_aux_loss: Whether to use global auxiliary loss
            moe_dropout_prob: Dropout probability for MoE layers
            moe_group: Group configuration for MoE experts
            moe_gate: Type of gating mechanism ('top2', etc.)
            moe_intermediate_size: Intermediate size for MoE layers
            moe_num_shared_experts: Number of shared experts
            moe_reverse_token_drop: Whether to use reverse token dropping
            moe_gate_act: Activation function for gating
            moe_norm_gate_logits: Whether to normalize gate logits
            moe_all_to_all_dropout: Dropout for all-to-all communication
            moe_k: Number of experts to route to
            moe_use_aux_free: Whether to use auxiliary-free routing
            moe_group_experts: Whether to group experts (requires hard gating)
            moe_group_orthogonal_loss: Whether to use group orthogonal loss
            enable_delay_scale_loss: Whether to enable delayed loss scaling
            num_acc_steps: Number of accumulation steps
            fuse_gate_detach_matmul: Whether to fuse gate detach matmul
            **kwargs: Additional base model configuration parameters

        Note:
            When use_recompute_moe is True, recompute_granularity will be changed to full_attn.
        """

        if use_recompute_moe:
            logger.warning(
                "set `use_recompute_moe`=True, disabling `recompute_granularity=full`, change to full_attn."
            )
            if kwargs["recompute"] and kwargs["recompute_granularity"] == "full":
                kwargs["recompute_granularity"] = "full_attn"
        super().__init__(**kwargs)

        self.moe_num_experts = moe_num_experts
        self.use_recompute_moe = use_recompute_moe
        self.moe_capacity = moe_capacity
        self.moe_aux_loss_lambda = moe_aux_loss_lambda
        self.moe_z_loss_lambda = moe_z_loss_lambda
        self.moe_orthogonal_loss_lambda = moe_orthogonal_loss_lambda
        self.global_aux_loss = global_aux_loss
        self.sinkhorn_2gate = sinkhorn_2gate
        self.sinkhorn_temp = sinkhorn_temp
        self.moe_layer_interval = moe_layer_interval
        self.moe_dropout_prob = moe_dropout_prob
        self.moe_group = moe_group
        self.moe_gate = moe_gate
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_reverse_token_drop = moe_reverse_token_drop
        self.moe_k = moe_k
        self.moe_all_to_all_dropout = moe_all_to_all_dropout
        self.moe_group_experts = moe_group_experts
        self.moe_group_orthogonal_loss = moe_group_orthogonal_loss
        self.enable_delay_scale_loss = enable_delay_scale_loss
        self.num_acc_steps = num_acc_steps
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_layer_end_index = (
            self.num_hidden_layers - 1
            if moe_layer_end_index == -1
            else moe_layer_end_index
        )
        self.moe_gate_act = moe_gate_act
        self.moe_norm_gate_logits = moe_norm_gate_logits
        self.moe_use_aux_free = moe_use_aux_free
        self.fuse_gate_detach_matmul = fuse_gate_detach_matmul
        self.dpo_config = dpo_config
        self.moe_multimodal_dispatch_use_allgather = (
            moe_multimodal_dispatch_use_allgather
        )
        self.moe_use_hard_gate = moe_use_hard_gate
        self.moe_dense_experts_token_type_id = moe_dense_experts_token_type_id

    @property
    def multimodel_experts(self) -> bool:
        """multimodel experts."""
        return (
            isinstance(self.moe_num_experts, (tuple, list))
            and len(self.moe_num_experts) > 1
        )

    @property
    def use_moe(self) -> bool:
        """
        Check if model is using MoE architecture.

        Returns:
            bool: True if moe_num_experts > 0, False otherwise
        """
        return self.moe_num_experts > 0


class Ernie4_5_VLMoEConfig(Ernie4_5_MoEConfig):
    """
    This is the configuration class to store the configuration of a [`~ErnieModel`]. It is used to instantiate an Ernie
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Ernie-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Ernie model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ErnieModel`] or [`~TFErnieModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
    """

    model_type = "ernie4_5_moe_vl"
    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }
    base_model_tp_plan = {
        "ernie.layers.*.self_attn.q_proj": "colwise_rep",
        "ernie.layers.*.self_attn.k_proj": "colwise_rep",
        "ernie.layers.*.self_attn.v_proj": "colwise_rep",
        "ernie.layers.*.self_attn.o_proj": "rowwise_rep",
        "ernie.layers.*.mlp.experts.*.gate_proj": "colwise",
        "ernie.layers.*.mlp.experts.*.up_proj": "colwise",
        "ernie.layers.*.mlp.experts.*.down_proj": "rowwise",
        "ernie.layers.*.mlp_text.experts.*.gate_proj": "colwise",
        "ernie.layers.*.mlp_text.experts.*.up_proj": "colwise",
        "ernie.layers.*.mlp_text.experts.*.down_proj": "rowwise",
        "ernie.layers.*.mlp.gate_proj": "colwise",
        "ernie.layers.*.mlp.up_proj": "colwise",
        "ernie.layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vision_config=None,
        im_patch_id=None,
        pixel_hidden_size=None,
        modality_detach=False,
        temporal_conv_size=2,
        spatial_conv_size=2,
        mm_vocab_size=0,  # vocab for mm specialtokens
        max_text_id=None,
        use_temporal_conv=True,
        moe_use_size_all2all=False,
        moe_num_attn_experts=False,
        moe_dense_experts_token_type_id: int = 3,
        moe_use_hard_gate: bool = True,
        moe_fuse_experts: bool = False,
        moe_use_token_type_bias: bool = False,
        disable_ffn_model_parallel=False,
        fuse_attn_ffn=True,
        rope_3d=True,
        freq_allocation=20,
        using_precision_check=False,
        use_recompute_resampler=False,
        resampler_fuse_rms_norm=False,
        moe_layer_feed_fake_token=False,
        tensor_parallel_degree=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(vision_config, dict):
            self.vision_config = DFNRopeVisionTransformerConfig(**vision_config)
        else:
            self.vision_config = DFNRopeVisionTransformerConfig()
        self.im_patch_id = im_patch_id
        self.pixel_hidden_size = pixel_hidden_size
        self.modality_detach = modality_detach
        self.temporal_conv_size = temporal_conv_size
        self.spatial_conv_size = spatial_conv_size
        self.mm_vocab_size = mm_vocab_size
        self.max_text_id = max_text_id
        self.use_temporal_conv = use_temporal_conv

        self.moe_use_size_all2all = moe_use_size_all2all
        self.moe_num_attn_experts = moe_num_attn_experts
        self.moe_dense_experts_token_type_id = moe_dense_experts_token_type_id
        self.moe_use_hard_gate = moe_use_hard_gate
        self.moe_fuse_experts = moe_fuse_experts
        self.moe_use_token_type_bias = moe_use_token_type_bias
        self.disable_ffn_model_parallel = disable_ffn_model_parallel

        self.fuse_attn_ffn = fuse_attn_ffn
        self.rope_3d = rope_3d
        self.freq_allocation = freq_allocation
        # TODO: proper integration
        self.rope_scaling = {"rope_type": "ernie_3d", "freq_allocation": freq_allocation}
        rope_config_validation(self)
        self.using_precision_check = using_precision_check
        self.use_recompute_resampler = use_recompute_resampler
        self.resampler_fuse_rms_norm = resampler_fuse_rms_norm
        self.moe_layer_feed_fake_token = moe_layer_feed_fake_token

        self.tensor_parallel_degree = tensor_parallel_degree

    @property
    def multimodel_experts(self) -> bool:
        """Check if model is using more than 1 multimodel experts."""
        return (
            isinstance(self.moe_num_experts, (tuple, list))
            and len(self.moe_num_experts) > 1
        )

    @property
    def use_moe(self) -> bool:
        """
        Check if model is using MoE architecture.

        Returns:
            bool: True if moe_num_experts > 0, False otherwise
        """
        return (
            sum(self.moe_num_experts) > 0
            if self.multimodel_experts
            else self.moe_num_experts > 0
        )

    def to_dict(self, saving_file=False):
        """to_dict"""
        output = copy.deepcopy(self.__dict__)
        if self.vision_config:
            output["vision_config"] = (
                self.vision_config.to_dict()
                if isinstance(self.vision_config, (DFNRopeVisionTransformerConfig))
                else self.vision_config
            )

        output["model_type"] = self.__class__.model_type
        return output
