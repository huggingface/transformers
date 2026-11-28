from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring
@strict
class Chronos2Config(PreTrainedConfig):
    r"""
    context_length (`int`):
        Length of the input context used for forecasting.

    input_patch_size (`int`):
        Number of input values in each patch.

    input_patch_stride (`int`):
        Step between consecutive input patches.

    output_patch_size (`int`):
        Number of forecast values produced per output patch.

    quantiles (`list[float]`):
        Quantiles predicted by the model.
    """

    model_type = "chronos2"

    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "d_kv",
    }

    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8

    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 0.05
    feed_forward_proj: str = "relu"
    vocab_size: int = 2
    pad_token_id: int = 0
    rope_theta: float = 10000.0

    context_length: int
    input_patch_size: int
    input_patch_stride: int
    output_patch_size: int
    quantiles: list[float]

    use_reg_token: bool = False
    use_arcsinh: bool = False
    max_output_patches: int = 1
    time_encoding_scale: int | None = None

    def __post_init__(self, **kwargs):
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if self.is_gated_act:
            raise ValueError("Gated activation is not supported.")

        kwargs["attn_implementation"] = kwargs.get("attn_implementation") or "sdpa"

        super().__post_init__(**kwargs)


__all__ = ["Chronos2Config"]
