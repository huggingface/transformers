# coding=utf-8
# Copyright 2024 The Convai Innovations Authors and The HuggingFace Inc. team. All rights reserved.
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
""" Hindi Causal LM model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "convaiinnovations/hindi-foundational-model-base": "https://huggingface.co/convaiinnovations/hindi-foundational-model-base/resolve/main/config.json",
    # Add other checkpoints here if needed
}


class HindiCausalLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HindiCausalLMModel`]. It is used to instantiate a
    Hindi Causal LM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    `convaiinnovations/hindi-foundational-model-base` architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 16000):
            Vocabulary size of the Hindi Causal LM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HindiCausalLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder. Must divide hidden_size.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"swish"` and `"gelu_new"` are supported. Checkpoints might use "swiglu" or "geglu" which maps to silu/gelu.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period for the Rotary Position Embedding.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the Rotary Positional Embedding. Currently supports
            `type` ("linear" or "dynamic") and `factor` (float).
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the RMS normalization layers. Note: original config.json used layer_norm_eps, mapping to rms_norm_eps.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The index of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 1):
            The index of the beginning of sequence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 2):
            The index of the end of sequence token in the vocabulary.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the weights of the input embeddings and the output embeddings.


    Example:

    ```
    >>> from transformers import HindiCausalLMModel, HindiCausalLMConfig

    >>> # Initializing a Hindi Causal LM style configuration
    >>> configuration = HindiCausalLMConfig()

    >>> # Initializing a model from the configuration
    >>> model = HindiCausalLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "hindi_causal_lm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=16000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=3072,
        hidden_act="silu", # From config.json
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-12, # From config.json layer_norm_eps
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        # Add other params from config.json if needed, mapping names
        hidden_dropout_prob=0.1, # Added from original config.json
        attention_probs_dropout_prob=0.1, # Added from original config.json
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        # Added dropout from original config.json for compatibility
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        # Map legacy LayerNorm name if present in kwargs
        if "layer_norm_eps" in kwargs and rms_norm_eps == 1e-12: # Avoid overwriting if explicitly set
             self.rms_norm_eps = kwargs.pop("layer_norm_eps")
             logger.warning("Mapping legacy `layer_norm_eps` to `rms_norm_eps`.")

        # Handle potential activation function name variations
        # The config.json states "silu", README mentions "swiglu". Treat them as equivalent (silu) here.
        if self.hidden_act in ["swiglu", "geglu"]:
             logger.info(f"Activation function '{self.hidden_act}' mapped to standard name.")
             # The specific gated mechanism is handled in the modeling file's MLP layer

        # Handle normalization layer type from config.json if needed
        # Here, we assume RMSNorm based on config and README.
        self.normalization_layer = kwargs.pop("normalization_layer", "rmsnorm")
        if self.normalization_layer != "rmsnorm":
            logger.warning(f"Config specifies `normalization_layer={self.normalization_layer}`, but implementation uses RMSNorm based on `rms_norm_eps`.")


        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # Copied from LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with keys `type` and `factor`, " f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

