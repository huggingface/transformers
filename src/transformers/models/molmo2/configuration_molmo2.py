"""
Molmo2 configuration
"""

from typing import Optional, Any

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging

logger = logging.get_logger(__name__)


class Molmo2VitConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Molmo2VisionTransformer`].
    It is used to instantiate a `Molmo2VisionTransformer` according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Example:
    ```python
    >>> from transformers import Molmo2VitConfig, Molmo2VisionTransformer

    >>> # Initializing a Molmo2VitConfig
    >>> configuration = Molmo2VitConfig()

    >>> # Initializing a Molmo2VisionTransformer (with random weights)
    >>> model = Molmo2VisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmo2"
    base_config_key = "vit_config"

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        image_default_input_size: tuple[int, int] = (378, 378),
        image_patch_size: int = 14,
        image_num_pos: int = 577,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        initializer_range: float = 0.02,
        float32_attention: bool = True,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.attn_implementation = attn_implementation
        super().__init__(
            attn_implementation=attn_implementation,
            **kwargs
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.image_default_input_size = image_default_input_size
        self.image_patch_size = image_patch_size
        self.image_num_pos = image_num_pos
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.initializer_range = initializer_range
        self.float32_attention = float32_attention

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


class Molmo2AdapterConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of Molmo2Adapter. With Molmo2VitConfig,
    It is used to instantiate an Molmo2VisionBackbone according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import Molmo2VitConfig, Molmo2AdapterConfig, Molmo2VisionBackbone

    >>> # Initializing a Molmo2VitConfig and a Molmo2AdapterConfig
    >>> vit_config = Molmo2VitConfig()
    >>> adapter_config = Molmo2AdapterConfig()

    >>> # Initializing a Molmo2VisionBackbone (with random weights)
    >>> model = Molmo2VisionBackbone(vit_config, adapter_config)

    >>> # Accessing the model configuration
    >>> vit_configuration = model.vit_config
    >>> adapter_configuration = model.adapter_config
    ```"""

    model_type = "molmo2"
    base_config_key = "adapter_config"

    def __init__(
        self,
        vit_layers: tuple = (-3, -9),
        pooling_attention_mask: bool = False,
        hidden_size: int = 1152,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        float32_attention: bool = True,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        hidden_act: str = "silu",
        intermediate_size: int = 18944,
        text_hidden_size: int = 3584,
        image_feature_dropout: float = 0.0,
        initializer_range: float = 0.02,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.attn_implementation = attn_implementation
        super().__init__(
            attn_implementation=attn_implementation,
            **kwargs
        )
        self.vit_layers = vit_layers
        self.pooling_attention_mask = pooling_attention_mask
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.float32_attention = float32_attention
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.text_hidden_size = text_hidden_size
        self.image_feature_dropout = image_feature_dropout
        self.initializer_range = initializer_range


class Molmo2TextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Molmo2TextModel`]. It is used to instantiate a
    `Molmo2TextModel` according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Example:
    ```python
    >>> from transformers import Molmo2TextConfig, Molmo2TextModel

    >>> # Initializing a Molmo2TextConfig
    >>> configuration = Molmo2TextConfig()

    >>> # Initializing a Molmo2TextModel (with random weights)
    >>> model = Molmo2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmo2_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "blocks.*.self_attn.att_proj": "colwise",
        "blocks.*.self_attn.attn_out": "rowwise",
        "blocks.*.mlp.ff_proj": "colwise",
        "blocks.*.mlp.ff_out": "rowwise",
    }
    base_model_pp_plan = {
        "wte": (["input_ids"], ["inputs_embeds"]),
        "blocks": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "ln_f": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: Optional[int] = 4,
        head_dim: int = 128,
        vocab_size: int = 152064,
        additional_vocab_size: int = 128,
        qkv_bias: bool = True,
        num_hidden_layers: int = 48,
        intermediate_size: int = 18944,
        hidden_act: str = "silu",
        embedding_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1000000.0,
        rope_scaling: dict[str, Any] = None,
        rope_scaling_layers: Optional[list[int]] = None,
        use_qk_norm: bool = False,
        qk_norm_type: str = "olmo",
        layer_norm_eps: int = 1e-6,
        norm_after: bool = False,
        initializer_range: float = 0.02,
        use_cache=True,
        tie_word_embeddings=False,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.attn_implementation = attn_implementation
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            attn_implementation=attn_implementation,
            **kwargs
        )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.qkv_bias = qkv_bias
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_scaling_layers = rope_scaling_layers
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.layer_norm_eps = layer_norm_eps
        self.norm_after = norm_after
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)


class Molmo2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Molmo2ForConditionalGeneration`].
    It is used to instantiate an Molmo2 model according to the specified arguments, defining the model architecture.

    Example:

    ```python
    >>> from transformers import Molmo2Config, Molmo2VitConfig, Molmo2AdapterConfig, Molmo2TextConfig

    >>> # Initializing a Molmo2VitConfig
    >>> vit_config = Molmo2VitConfig()

    >>> # Initializing a Molmo2AdapterConfig
    >>> adapter_config = Molmo2AdapterConfig()

    >>> # Initializing a Molmo2TextConfig
    >>> text_config = Molmo2TextConfig()

    >>> # Initializing a Molmo2Config
    >>> configuration = Molmo2Config(
    >>>     vit_config=vit_config,
    >>>     adapter_config=adapter_config,
    >>>     text_config=text_config,
    >>>     image_start_token_id=151936,
    >>>     image_end_token_id=151937,
    >>>     image_patch_id=151938,
    >>>     image_col_id=151939,
    >>>     low_res_image_start_token_id=151940,
    >>>     image_low_res_id=151942,
    >>>     frame_start_token_id=151943,
    >>>     frame_end_token_id=151944,
    >>> )

    >>> # Initializing a model
    >>> model = Molmo2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "molmo2"
    sub_configs = {
        "text_config": Molmo2TextConfig,
        "vit_config": Molmo2VitConfig,
        "adapter_config": Molmo2AdapterConfig,
    }

    def __init__(
        self,
        vit_config: Optional[Molmo2VitConfig] = None,
        adapter_config: Optional[Molmo2AdapterConfig] = None,
        text_config: Optional[Molmo2TextConfig] = None,
        image_start_token_id: Optional[int] = None,
        low_res_image_start_token_id: Optional[int] = None,
        image_end_token_id: Optional[int] = None,
        image_low_res_id: Optional[int] = None,
        image_patch_id: Optional[int] = None,
        image_col_id: Optional[int] = None,
        frame_start_token_id: Optional[int] = None,
        frame_end_token_id: Optional[int] = None,
        use_frame_special_tokens: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vit_config is None:
            self.vit_config = Molmo2VitConfig()
        elif isinstance(vit_config, dict):
            self.vit_config = Molmo2VitConfig(**vit_config)
        else:
            self.vit_config = vit_config
        if adapter_config is None:
            self.adapter_config = Molmo2AdapterConfig()
        elif isinstance(adapter_config, dict):
            self.adapter_config = Molmo2AdapterConfig(**adapter_config)
        else:
            self.adapter_config = adapter_config
        if text_config is None:
            self.text_config = Molmo2TextConfig()
        elif isinstance(text_config, dict):
            self.text_config = Molmo2TextConfig(**text_config)
        else:
            self.text_config = text_config
        self.image_start_token_id = image_start_token_id
        self.low_res_image_start_token_id = low_res_image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_low_res_id = image_low_res_id
        self.image_high_res_id = image_patch_id
        self.image_patch_id = image_patch_id
        self.image_col_id = image_col_id
        self.frame_start_token_id = frame_start_token_id
        self.frame_end_token_id = frame_end_token_id
        self.use_frame_special_tokens = use_frame_special_tokens
        self.initializer_range = initializer_range

    @property
    def image_num_patch(self):
        assert self.vit_config is not None
        return self.vit_config.image_num_patch
    
    @property
    def num_attention_heads(self):
        return self.text_config.num_attention_heads
    
    @property
    def num_key_value_heads(self):
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self):
        return self.text_config.head_dim

    @property
    def num_hidden_layers(self):
        return self.text_config.num_hidden_layers
    
    @property
    def hidden_size(self):
        return self.text_config.hidden_size
    
    @property
    def vocab_size(self):
        return self.text_config.vocab_size
    
    @property
    def max_position_embeddings(self):
        return self.text_config.max_position_embeddings
