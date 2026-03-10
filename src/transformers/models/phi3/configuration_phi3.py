# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""Phi-3 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/Phi-3-mini-4k-instruct")
class Phi3Config(PreTrainedConfig):
    r"""
    original_max_position_embeddings (`int`, *optional*, defaults to 4096):
        The maximum sequence length that this model was trained with. This is used to determine the size of the
        original RoPE embeddings when using long scaling.

    Example:

    ```python
    >>> from transformers import Phi3Model, Phi3Config

    >>> # Initializing a Phi-3 style configuration
    >>> configuration = Phi3Config.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    >>> # Initializing a model from the configuration
    >>> model = Phi3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "phi3"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.qkv_proj": "colwise_gather_output",  # we need to replicate here due to the slicing of qkv
        "layers.*.self_attn.o_proj": "rowwise_split_input",  # input is replicated due to the slicing of qkv
        "layers.*.mlp.gate_up_proj": "colwise_gather_output",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_split_input",  # input is replicated due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 32064,
        hidden_size: int | None = 3072,
        intermediate_size: int | None = 8192,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        resid_pdrop: float | None = 0.0,
        embd_pdrop: float | None = 0.0,
        attention_dropout: float | None = 0.0,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096,
        original_max_position_embeddings: int | None = 4096,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 32000,
        pad_token_id: int | None = 32000,
        sliding_window: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 1.0)  # assign default for BC
        self.sliding_window = sliding_window

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)

    def convert_rope_params_to_dict(
        self, default_theta: int | float = 10_000.0, ignore_keys: set | None = None, **kwargs
    ):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", default_theta))
        self.rope_parameters.setdefault("partial_rotary_factor", kwargs["partial_rotary_factor"])
        self.standardize_rope_params()

        # For backward compatibility if previous version used "su" or "yarn"
        rope_parameters_type = self.rope_parameters.get("rope_type", None)
        if rope_parameters_type is not None and rope_parameters_type in ["su", "yarn"]:
            self.rope_parameters["rope_type"] = "longrope"
        self.validate_rope(ignore_keys=ignore_keys)
        return kwargs

    def validate_rope(self, ignore_keys: set | None = None):
        """
        Validate the `rope_parameters` configuration.
        """
        super().validate_rope(ignore_keys=ignore_keys)

        # Run Phi3 specific validation
        if not isinstance(self.rope_parameters, dict):
            raise ValueError(f"`rope_parameters` must be a dictionary but got {self.rope_parameters}")
        rope_parameters_type = self.rope_parameters.get("rope_type", None)
        rope_parameters_short_factor = self.rope_parameters.get("short_factor", None)
        rope_parameters_long_factor = self.rope_parameters.get("long_factor", None)
        rotary_ndims = int(
            self.hidden_size // self.num_attention_heads * self.rope_parameters["partial_rotary_factor"]
        )
        if rope_parameters_type not in ["default", "longrope"]:
            raise ValueError(f"`rope_parameters`'s type field must be one of ['longrope'], got {rope_parameters_type}")

        if rope_parameters_short_factor is not None:
            if not (
                isinstance(rope_parameters_short_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_parameters_short_factor)
            ):
                raise ValueError(
                    f"`rope_parameters`'s short_factor field must be a list of numbers, got {rope_parameters_short_factor}"
                )
            if not len(rope_parameters_short_factor) == rotary_ndims // 2:
                raise ValueError(
                    f"`rope_parameters`'s short_factor field must have length {rotary_ndims // 2}, got {len(rope_parameters_short_factor)}"
                )

        if rope_parameters_long_factor is not None:
            if not (
                isinstance(rope_parameters_long_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_parameters_long_factor)
            ):
                raise ValueError(
                    f"`rope_parameters`'s long_factor field must be a list of numbers, got {rope_parameters_long_factor}"
                )
            if not len(rope_parameters_long_factor) == rotary_ndims // 2:
                raise ValueError(
                    f"`rope_parameters`'s long_factor field must have length {rotary_ndims // 2}, got {len(rope_parameters_long_factor)}"
                )


__all__ = ["Phi3Config"]
