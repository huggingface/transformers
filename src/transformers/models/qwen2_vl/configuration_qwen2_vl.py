# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen2VL model configuration"""

import inspect

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Qwen2-VL-7B-Instruct")
class Qwen2VLVisionConfig(PreTrainedConfig):
    model_type = "qwen2_vl"
    base_config_key = "vision_config"

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
        temporal_patch_size=2,
        initializer_range=0.02,
        **kwargs,
    ):
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
        self.temporal_patch_size = temporal_patch_size
        self.initializer_range = initializer_range


@auto_docstring(checkpoint="Qwen2-VL-7B-Instruct")
class Qwen2VLTextConfig(PreTrainedConfig):
    r"""
    max_window_layers (`int`, *optional*, defaults to 80):
        The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
        additional layer afterwards will use SWA (Sliding Window Attention).

    ```python
    >>> from transformers import Qwen2VLTextModel, Qwen2VLConfig

    >>> # Initializing a Qwen2VL style configuration
    >>> configuration = Qwen2VLConfig()

    >>> # Initializing a model from the Qwen2-VL-7B style configuration
    >>> model = Qwen2VLTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0
    # Default tensor parallel plan for base model `Qwen2VL`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 152064,
        hidden_size: int | None = 8192,
        intermediate_size: int | None = 29568,
        num_hidden_layers: int | None = 80,
        num_attention_heads: int | None = 64,
        num_key_value_heads: int | None = 8,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-05,
        use_cache: bool | None = True,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = 4096,
        max_window_layers: int | None = 80,
        layer_types: list[str] | None = None,
        attention_dropout: float | None = 0.0,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        bos_token_id: int | None = 151643,
        eos_token_id: int | None = 151645,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        super().__init__(
            ignore_keys_at_rope_validation={"mrope_section"},
            **kwargs,
        )

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation: set | None = None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", self.default_theta))
        if self.rope_parameters.get("rope_type", self.rope_parameters.get("type")) == "mrope":
            self.rope_parameters["rope_type"] = "default"
        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs


@auto_docstring(checkpoint="Qwen2-VL-7B-Instruct")
class Qwen2VLConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig

    >>> # Initializing a Qwen2VL style configuration
    >>> configuration = Qwen2VLConfig()

    >>> # Initializing a model from the Qwen2-VL-7B style configuration
    >>> model = Qwen2VLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_vl"
    sub_configs = {"vision_config": Qwen2VLVisionConfig, "text_config": Qwen2VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # Hub configs are saved as flat dicts so we pop some of kwargs to init `TextConfig`
            text_params = inspect.signature(self.sub_configs["text_config"].__init__).parameters.keys()
            text_params = list(text_params) + ["rope_scaling", "rope_theta"]
            text_config = {key: kwargs.pop(key) for key in text_params if key in kwargs}
            text_config["dtype"] = kwargs.get("torch_dtype", kwargs.get("dtype"))  # don't pop the dtype
            self.text_config = self.sub_configs["text_config"](**text_config)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["Qwen2VLConfig", "Qwen2VLTextConfig"]
