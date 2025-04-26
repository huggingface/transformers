# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
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
"""HindiCausalLM configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

# Archive map might point to your fine-tuned model if uploaded
HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "convaiinnovations/hindi-foundational-model-base": "https://huggingface.co/convaiinnovations/hindi-foundational-model-base/resolve/main/config.json",
    # Add path to your fine-tuned config if uploaded, e.g.:
    # "your-username/hindi-lm-output-new": "https://huggingface.co/your-username/hindi-lm-output-new/resolve/main/config.json",
}


class HindiCausalLMConfig(PretrainedConfig):
    """
    Configuration class for HindiCausalLM model, adapted to match the structure
    used in the training script (hindi_language_model.py).

    This is the configuration class to store the configuration of a [`HindiCausalLMModel`].
    It is used to instantiate a Hindi causal language model according to the specified arguments.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 16000): Vocabulary size.
        hidden_size (`int`, *optional*, defaults to 768): Dimension of hidden layers.
        num_hidden_layers (`int`, *optional*, defaults to 12): Number of hidden layers.
        num_attention_heads (`int`, *optional*, defaults to 12): Number of attention heads. (Adjusted default)
        intermediate_size (`int`, *optional*, defaults to 3072): Dimension of the FFN layer.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function. **Set to "gelu"** to match the hardcoded nn.GELU()
            in the original training code's TransformerBlock. Other values will be ignored by the
            modeling code refactored to match the original implementation.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1): Dropout probability for FFN layers.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1): Dropout ratio for attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512): Maximum sequence length.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12): Epsilon for layer normalization.
        normalization_layer (`str`, *optional*, defaults to `"layernorm"`):
             Normalization layer type. **Set to "layernorm"** as the original code uses nn.LayerNorm.
             Other values are not supported by the refactored modeling code.
        positional_encoding_type (`str`, *optional*, defaults to `"absolute"`):
             Positional encoding type. **Set to "absolute" or "learned"** as the original code uses
             standard nn.Embedding for positions. RoPE is not implemented in the refactored model code.
        pad_token_id (`int`, *optional*, defaults to 0): Padding token ID.
        bos_token_id (`int`, *optional*, defaults to 1): BOS token ID.
        eos_token_id (`int`, *optional*, defaults to 2): EOS token ID.
        unk_token_id (`int`, *optional*, defaults to 3): UNK token ID.
        tie_word_embeddings (`bool`, *optional*, defaults to True): Whether to tie word embeddings.
    """

    model_type = "hindi_causal_lm"
    keys_to_ignore_at_inference = ["past_key_values"]  # Standard key to ignore

    def __init__(
        self,
        vocab_size=16000,  # Match your trained tokenizer vocab size if different
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,  # Default from original notebook code (12) might differ from training script arg (None -> auto)
        intermediate_size=3072,
        # --- Set defaults based on original script's implementation ---
        hidden_act="gelu",  # GELU was hardcoded in original FFN
        normalization_layer="layernorm",  # Only LayerNorm used in original
        positional_encoding_type="absolute",  # Or "learned"; RoPE not implemented in original
        # --- End implementation-based defaults ---
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
        tie_word_embeddings=True,
        **kwargs,
    ):
        # Ensure superclass gets necessary token IDs and tie_word_embeddings
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.normalization_layer = normalization_layer
        self.positional_encoding_type = positional_encoding_type

        # Store original training script args if passed, but they don't drive the structure now
        self.activation_function = kwargs.pop("activation_function", hidden_act)
        self.norm_type = kwargs.pop("norm_type", normalization_layer)
        self.pos_encoding_type = kwargs.pop("pos_encoding_type", positional_encoding_type)

        # Validate hidden_size vs num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            valid_heads = [h for h in [16, 12, 8, 6, 4, 2, 1] if self.hidden_size > 0 and self.hidden_size % h == 0]
            if valid_heads:
                original_heads = self.num_attention_heads
                self.num_attention_heads = valid_heads[0]
                logger.warning(
                    f"Overriding num_attention_heads from {original_heads} to {self.num_attention_heads} "
                    f"to be divisible by hidden_size {self.hidden_size}."
                )
            else:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads "
                    f"({self.num_attention_heads}). No suitable alternative found."
                )

        # Validate hardcoded implementation choices
        if self.hidden_act != "gelu":
            logger.warning(
                f"Config hidden_act is '{self.hidden_act}', but the adapted modeling code uses hardcoded GELU "
                f"to match the original training script. This config value will be ignored by the model structure."
            )
        if self.normalization_layer != "layernorm":
            logger.warning(
                f"Config normalization_layer is '{self.normalization_layer}', but the adapted modeling code uses nn.LayerNorm "
                f"to match the original training script. This config value will be ignored by the model structure."
            )
        if self.positional_encoding_type == "rope":
            logger.warning(
                "Config positional_encoding_type is 'rope', but the adapted modeling code uses standard embeddings "
                "to match the original training script. RoPE is not implemented."
            )
            # Force to absolute/learned to match implementation
            self.positional_encoding_type = "absolute"
