from collections.abc import Sequence
from typing import Any

from transformers.configuration_utils import PretrainedConfig


class StepRoboticsVisionEncoderConfig(PretrainedConfig):
    model_type = "perception_encoder"

    def __init__(
        self,
        width=1536,
        layers=47,
        heads=16,
        num_channels=3,
        image_size=728,
        mlp_ratio=8960 / 1536,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        ues_cls_token=False,
        use_cls_token: bool | None = None,
        use_ln_pre=True,
        use_ln_post=False,
        use_abs_posemb=True,
        use_rope2d=True,
        ls_init_value=0.1,
        **kwargs,
    ):
        self.width = width
        self.hidden_size = width
        self.layers = layers
        self.num_hidden_layers = layers
        self.heads = heads
        self.num_attention_heads = heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.intermediate_size = int(width * mlp_ratio)
        self.attention_dropout = 0.0
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        if use_cls_token is None:
            use_cls_token = ues_cls_token
        self.ues_cls_token = use_cls_token
        self.use_cls_token = use_cls_token
        self.use_ln_pre = use_ln_pre
        self.ls_init_value = ls_init_value
        self.use_ln_post = use_ln_post
        self.use_abs_posemb = use_abs_posemb
        self.use_rope2d = use_rope2d
        super().__init__(**kwargs)


class Step3p7TextConfig(PretrainedConfig):
    model_type = "step3p5"
    architectures = ["Step3p5ForCausalLM"]

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11264,
        num_attention_heads: int = 64,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 45,
        max_seq_len: int = 128000,
        vocab_size: int = 128815,
        rms_norm_eps: float = 1e-5,
        moe_intermediate_size: int = 1280,
        moe_num_experts: int = 288,
        moe_top_k: int = 8,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 128000,
        share_expert_dims: int = 1280,
        share_expert_dim: int | None = None,
        head_dim: int = 128,
        norm_expert_weight: bool = True,
        layer_types: list[str] = None,
        sliding_window: int | None = None,
        pad_token_id: int = 1,
        attention_dropout: float = 0.0,
        use_head_wise_attn_gate: bool = False,
        use_moe_router_bias: bool = False,
        moe_router_activation: str = "softmax",
        moe_router_scaling_factor: float = 1.0,
        need_fp32_gate: bool = False,
        attention_other_setting: dict[str, Any] | None = None,
        swiglu_limits: list[float | None] | None = None,
        swiglu_limits_shared: list[float | None] | None = None,
        use_rope_layers: list[bool] | None = None,
        yarn_only_types: list[str] | None = None,
        moe_layers_enum: tuple[int] = (
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
        ),
        **kwargs,
    ) -> None:
        torch_dtype = kwargs.get("torch_dtype")
        trim_layer_types = _normalize_per_layer_values(layer_types, num_hidden_layers)
        if isinstance(rope_scaling, dict):
            rope_scaling = dict(rope_scaling)
        if share_expert_dim is None:
            share_expert_dim = share_expert_dims
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.share_expert_dim = share_expert_dim
        self.head_dim = head_dim
        self.norm_expert_weight = norm_expert_weight
        self.moe_layers_enum = moe_layers_enum
        self.layer_types = trim_layer_types
        self.sliding_window = sliding_window
        self.pad_token_id = pad_token_id
        self.attention_dropout = attention_dropout
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.use_moe_router_bias = use_moe_router_bias
        self.moe_router_activation = moe_router_activation
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.need_fp32_gate = need_fp32_gate
        self.attention_other_setting = attention_other_setting
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared
        self.use_rope_layers = use_rope_layers
        self.yarn_only_types = yarn_only_types
        super().__init__(**kwargs)
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        self.layer_types = layer_types

    def to_dict(self):
        output = super().to_dict()
        torch_dtype = getattr(self, "torch_dtype", None)
        if torch_dtype is not None:
            output["torch_dtype"] = torch_dtype
        return output


def _normalize_per_layer_values(
    values: Sequence[Any] | None,
    num_hidden_layers: int,
) -> list[Any] | None:
    if values is None:
        return None
    normalized = list(values)
    if not normalized:
        return normalized
    if len(normalized) < num_hidden_layers:
        normalized.extend([normalized[-1]] * (num_hidden_layers - len(normalized)))
    # Some checkpoints keep MTP/spec layer entries after the decoder layers.
    # This config only builds num_hidden_layers decoder layers, and HF strict
    # validation requires per-layer fields to match that decoder count.
    return normalized[:num_hidden_layers]


class Step3p7Config(PretrainedConfig):
    # This loader is a compatibility shim for original Step VL checkpoints
    # whose top-level config model_type is `step3p7`.
    model_type = "step3p7"

    def __init__(
        self,
        vision_config: dict | StepRoboticsVisionEncoderConfig | None = None,
        text_config: dict | Step3p7TextConfig | None = None,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        image_token_id: int = 151679,
        **kwargs,
    ) -> None:
        shared_rope_scaling = kwargs.get("rope_scaling")
        if isinstance(shared_rope_scaling, dict):
            shared_rope_scaling = dict(shared_rope_scaling)

        if vision_config is None:
            vision_config = StepRoboticsVisionEncoderConfig()
        elif isinstance(vision_config, dict):
            vision_config = StepRoboticsVisionEncoderConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Step3p7TextConfig(rope_scaling=shared_rope_scaling)
        elif isinstance(text_config, dict):
            text_config = dict(text_config)
            if shared_rope_scaling is not None and "rope_scaling" not in text_config:
                text_config["rope_scaling"] = shared_rope_scaling
            text_config = Step3p7TextConfig(**text_config)
        elif shared_rope_scaling is not None and text_config.rope_scaling is None:
            text_config.rope_scaling = dict(shared_rope_scaling)
        self.text_config = text_config

        rope_scaling = kwargs.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            kwargs["rope_scaling"] = dict(rope_scaling)

        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.hidden_size = text_config.hidden_size
        self.max_position_embeddings = text_config.max_position_embeddings
        self.image_token_id = image_token_id
        # Help Auto classes find the correct implementation when saving/loading.
        super().__init__(**kwargs)
