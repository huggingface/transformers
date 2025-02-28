# coding=utf-8
# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Evolla model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class EvollaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EvollaModel`]. It is used to instantiate an
    Evolla model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Evolla-10B.

    e.g. [westlake-repl/Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the Evolla llama model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`EvollaModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the llama layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimensionality of the intermediate layers in the llama model.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the llama model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the llama model.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value pairs for each attention layer in the llama model.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the llama model. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"silu"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for the RMS-norm layer in the llama model.
        pretraining_tp (`int`, *optional*, defaults to 1):
            The pretraining task. 1 for language modeling, 2 for protein sequence modeling.
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The threshold value for the RoPE layer in the llama model.
        rope_scaling (`float`, *optional*):
            The scaling factor for the RoPE layer in the llama model.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention layer.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP layer.
        aligner_ffn_mult (`int`, *optional*, defaults to 4):
            The FFN multiplier for the aligner layer.
        aligner_enable_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the aligner layer.
        aligner_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities in the aligner layer.
        aligner_num_add_layers (`int`, *optional*, defaults to 8):
            The number of additional layers for the aligner layer.
        protein_vocab_size (`int`, *optional*, defaults to 446):
            Vocabulary size of the protein sequence model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`EvollaModel`].
        protein_mask_token_id (`int`, *optional*, defaults to 4):
            The id of the *mask* token in the protein sequence model.
        protein_pad_token_id (`int`, *optional*, defaults to 1):
            The id of the *padding* token in the protein sequence model.
        protein_hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the protein sequence model layers and the pooler layer.
        protein_num_hidden_layers (`int`, *optional*, defaults to 33):
            Number of hidden layers in the protein sequence model.
        protein_num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the protein sequence model.
        protein_intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the intermediate layers in the protein sequence model.
        protein_hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the hidden layers in the protein sequence model.
        protein_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities in the protein sequence model.
        protein_max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that the protein sequence model might ever be used with. Typically set this to
            something large just in case (e.g., 512 or 1024 or 2048).
        protein_layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for the layer normalization layer in the protein sequence model.
        protein_position_embedding_type (`str`, *optional*, defaults to `"rotary"`):
            The type of position embedding to use in the protein sequence model. Currently only `"rotary"` is supported.
        protein_emb_layer_norm_before (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization before the position embedding in the protein sequence model.
        protein_token_dropout (`bool`, *optional*, defaults to `True`):
            Whether to apply dropout to the tokens in the protein sequence model.
        resampler_depth (`int`, *optional*, defaults to 6):
            The depth of the resampler layer in the llama model.
        resampler_dim_head (`int`, *optional*, defaults to 64):
            The dimension of the heads in the resampler layer in the llama model.
        resampler_heads (`int`, *optional*, defaults to 8):
            The number of heads in the resampler layer in the llama model.
        resampler_num_latents (`int`, *optional*, defaults to 64):
            The number of latents in the resampler layer in the llama model.
        resampler_ff_mult (`int`, *optional*, defaults to 4):
            The FFN multiplier for the resampler layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 128000):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 128009):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the input and output word embeddings.

    Example:

    ```python
    >>> from transformers import EvollaModel, EvollaConfig

    >>> # Initializing a Evolla evolla-10b style configuration
    >>> configuration = EvollaConfig()

    >>> # Initializing a model from the evolla-10b style configuration
    >>> model = EvollaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "EvollaModel"
    # sub_configs = {"protein_config": EvollaProteinConfig, "llm_config": EvollaLLMConfig}

    def __init__(
        self,
        vocab_size=128256,  # llama vocab size
        hidden_size=4096,  # llama hidden size
        intermediate_size=14336,  # llama intermediate size
        num_hidden_layers=32,  # llama num layers
        num_attention_heads=32,  # llama num heads
        num_key_value_heads=8,  # llama num key-value heads
        hidden_act="silu",  # llama activation function
        max_position_embeddings=8192,  # llama rope max length
        rms_norm_eps=1e-05,
        pretraining_tp=1,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        # head_dim=None,
        aligner_ffn_mult=4,
        aligner_enable_bias=True,
        aligner_attention_probs_dropout_prob=0.1,
        aligner_num_add_layers=8,
        protein_vocab_size=446,
        protein_mask_token_id=4,
        protein_pad_token_id=1,
        protein_hidden_size=1280,
        protein_num_hidden_layers=33,
        protein_num_attention_heads=20,
        protein_intermediate_size=5120,
        protein_hidden_dropout_prob=0.1,
        protein_attention_probs_dropout_prob=0.1,
        protein_max_position_embeddings=1026,
        protein_layer_norm_eps=1e-05,
        protein_position_embedding_type="rotary",
        protein_emb_layer_norm_before=False,
        protein_token_dropout=True,
        resampler_depth=6,
        resampler_dim_head=64,
        resampler_heads=8,
        resampler_num_latents=64,
        resampler_ff_mult=4,
        # protein_config=None,
        # llm_config=None,
        initializer_range=0.02,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128009,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        tie_word_embeddings=False,
        # max_new_tokens=512,
        # do_sample=True,
        # temperature=0.6,
        # top_p=0.9,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # self.output_attentions = output_attentions
        # self.output_hidden_states = output_hidden_states
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        # self.pad_token_id = pad_token_id
        # self.bos_token_id = bos_token_id
        # self.eos_token_id = eos_token_id
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.aligner_ffn_mult = aligner_ffn_mult
        self.aligner_enable_bias = aligner_enable_bias
        self.aligner_attention_probs_dropout_prob = aligner_attention_probs_dropout_prob
        self.aligner_num_add_layers = aligner_num_add_layers

        self.protein_vocab_size = protein_vocab_size
        self.protein_mask_token_id = protein_mask_token_id
        self.protein_pad_token_id = protein_pad_token_id
        self.protein_hidden_size = protein_hidden_size
        self.protein_num_hidden_layers = protein_num_hidden_layers
        self.protein_num_attention_heads = protein_num_attention_heads
        self.protein_intermediate_size = protein_intermediate_size
        self.protein_hidden_dropout_prob = protein_hidden_dropout_prob
        self.protein_attention_probs_dropout_prob = protein_attention_probs_dropout_prob
        self.protein_max_position_embeddings = protein_max_position_embeddings
        self.protein_layer_norm_eps = protein_layer_norm_eps
        self.protein_position_embedding_type = protein_position_embedding_type
        self.protein_emb_layer_norm_before = protein_emb_layer_norm_before
        self.protein_token_dropout = protein_token_dropout

        self.resampler_depth = resampler_depth
        self.resampler_dim_head = resampler_dim_head
        self.resampler_heads = resampler_heads
        self.resampler_num_latents = resampler_num_latents
        self.resampler_ff_mult = resampler_ff_mult

        # self.max_new_tokens = max_new_tokens
        # self.do_sample = do_sample
        # self.temperature = temperature
        # self.top_p = top_p
        # print(return_dict, self.return_dict)

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # if protein_config is None:
        #     self.protein_config = EvollaProteinConfig(protein_text_hidden_size=self.protein_text_hidden_size)
        # elif isinstance(protein_config, dict):
        #     protein_config.update({"protein_text_hidden_size": self.protein_text_hidden_size})
        #     self.protein_config = EvollaProteinConfig(**protein_config)
        # elif isinstance(protein_config, EvollaProteinConfig):
        #     self.protein_config = protein_config

        # if llm_config is None:
        #     self.llm_config = EvollaLLMConfig(protein_text_hidden_size=self.protein_text_hidden_size)
        # elif isinstance(llm_config, dict):
        #     llm_config.update({"protein_text_hidden_size": self.protein_text_hidden_size})
        #     self.llm_config = EvollaLLMConfig(**llm_config)
        # elif isinstance(llm_config, EvollaLLMConfig):
        #     self.llm_config = llm_config

        # if self.protein_config.resampler_config.output_repr_dim is None:
        #     self.protein_config.resampler_config.output_repr_dim = self.llm_config.llama_config.hidden_size

        # if self.llm_config.sequence_aligner_config.protein_encoder_dim is None:
        #     self.llm_config.sequence_aligner_config.protein_encoder_dim = self.protein_config.resampler_config.output_repr_dim

        self.initializer_range = initializer_range
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
            # max_new_tokens=max_new_tokens,
            # do_sample=do_sample,
            # temperature=temperature,
            # top_p=top_p,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # IMPORTANT: Do not do any __init__ args-based checks in the constructor, since
        # PretrainedConfig.from_dict first instantiates the class with the config dict and only then
        # updates the config object with `kwargs` from from_pretrained, so during the instantiation
        # of this object many attributes have default values and haven't yet been overridden.
        # Do any required checks inside `from_pretrained` once the superclass' `from_pretrained` was run.


__all__ = ["EvollaConfig"]
