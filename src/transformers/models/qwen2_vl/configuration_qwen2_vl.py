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
from ...modeling_rope_utils import RopeParameters, RotaryEmbeddingConfigMixin
from ...utils import logging


logger = logging.get_logger(__name__)


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


class Qwen2VLTextConfig(PreTrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2VLTextModel`]. It is used to instantiate a
    Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the Qwen2VL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2VLModel`]
        hidden_size (`int`, *optional*, defaults to 8192):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 29568):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 80):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        bos_token_id (`int`, *optional*, defaults to 151643):
            The id of the _beginning-of-stream_ token.
        eos_token_id (`int`, *optional*, defaults to 151645):
            The id of the _end-of-stream_ token.
        pad_token_id (`int`, *optional*):
            The id of the _padding_ token.

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


class Qwen2VLConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2VLModel`]. It is used to instantiate a
    Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2VLTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen2VLVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The token index to denote start of vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The token index to denote end of vision input.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

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
