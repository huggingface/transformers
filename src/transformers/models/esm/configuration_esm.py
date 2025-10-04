# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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
"""ESM model configuration"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.type_validation import interval, is_divisible_by


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass
class EsmFoldConfig(PretrainedConfig):
    esm_type: Optional[str] = None
    fp16_esm: Optional[bool] = True
    use_esm_attn_map: Optional[bool] = False
    esm_ablate_pairwise: Optional[bool] = False
    esm_ablate_sequence: Optional[bool] = False
    esm_input_dropout: Optional[float] = 0.0
    embed_aa: Optional[bool] = True
    bypass_lm: Optional[bool] = False
    lddt_head_hid_dim: Optional[int] = 128
    trunk: Optional[Union[dict, "TrunkConfig"]] = None

    def __post_init__(self):
        if self.trunk is None:
            self.trunk = TrunkConfig()
        elif isinstance(self.trunk, dict):
            self.trunk = TrunkConfig(**self.trunk)


@strict(accept_kwargs=True)
@dataclass
class TrunkConfig(PretrainedConfig):
    num_blocks: Optional[int] = 48
    sequence_state_dim: Optional[int] = 1024
    pairwise_state_dim: Optional[int] = is_divisible_by(divisor=2)(default=128)
    sequence_head_width: Optional[int] = 32
    pairwise_head_width: Optional[int] = 32
    position_bins: Optional[int] = 32
    dropout: Optional[float] = interval(max=0.4)(default=0.0)
    layer_drop: Optional[float] = 0.0
    cpu_grad_checkpoint: Optional[bool] = False
    max_recycles: Optional[int] = interval(min=0)(default=4)
    chunk_size: Optional[int] = 128
    structure_module: Optional[Union[dict, "StructureModuleConfig"]] = None

    def __post_init__(self):
        if self.structure_module is None:
            self.structure_module = StructureModuleConfig()
        elif isinstance(self.structure_module, dict):
            self.structure_module = StructureModuleConfig(**self.structure_module)

        if self.sequence_state_dim % self.sequence_state_dim != 0:
            raise ValueError(
                "`sequence_state_dim` should be a round multiple of `sequence_state_dim`, got"
                f" {self.sequence_state_dim} and {self.sequence_state_dim}."
            )
        if self.pairwise_state_dim % self.pairwise_state_dim != 0:
            raise ValueError(
                "`pairwise_state_dim` should be a round multiple of `pairwise_state_dim`, got"
                f" {self.pairwise_state_dim} and {self.pairwise_state_dim}."
            )

        sequence_num_heads = self.sequence_state_dim // self.sequence_head_width
        pairwise_num_heads = self.pairwise_state_dim // self.pairwise_head_width

        if self.sequence_state_dim != sequence_num_heads * self.sequence_head_width:
            raise ValueError(
                "`sequence_state_dim` should be equal to `sequence_num_heads * sequence_head_width, got"
                f" {self.sequence_state_dim} != {sequence_num_heads} * {self.sequence_head_width}."
            )
        if self.pairwise_state_dim != pairwise_num_heads * self.pairwise_head_width:
            raise ValueError(
                "`pairwise_state_dim` should be equal to `pairwise_num_heads * pairwise_head_width, got"
                f" {self.pairwise_state_dim} != {pairwise_num_heads} * {self.pairwise_head_width}."
            )


@strict(accept_kwargs=True)
@dataclass
class StructureModuleConfig(PretrainedConfig):
    """
    Args:
        sequence_dim:
            Single representation channel dimension
        pairwise_dim:
            Pair representation channel dimension
        ipa_dim:
            IPA hidden channel dimension
        resnet_dim:
            Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
        num_heads_ipa:
            Number of IPA heads
        num_qk_points:
            Number of query/key points to generate during IPA
        num_v_points:
            Number of value points to generate during IPA
        dropout_rate:
            Dropout rate used throughout the layer
        num_blocks:
            Number of structure module blocks
        num_transition_layers:
            Number of layers in the single representation transition (Alg. 23 lines 8-9)
        num_resnet_blocks:
            Number of blocks in the angle resnet
        num_angles:
            Number of angles to generate in the angle resnet
        trans_scale_factor:
            Scale of single representation transition hidden dimension
        epsilon:
            Small number used in angle resnet normalization
        inf:
            Large number used for attention masking
    """

    sequence_dim: Optional[int] = 384
    pairwise_dim: Optional[int] = 128
    ipa_dim: Optional[int] = 16
    resnet_dim: Optional[int] = 128
    num_heads_ipa: Optional[int] = 12
    num_qk_points: Optional[int] = 4
    num_v_points: Optional[int] = 8
    dropout_rate: Optional[float] = 0.1
    num_blocks: Optional[int] = 8
    num_transition_layers: Optional[int] = 1
    num_resnet_blocks: Optional[int] = 2
    num_angles: Optional[int] = 7
    trans_scale_factor: Optional[int] = 10
    epsilon: Optional[float] = 1e-8
    inf: Optional[float] = 1e5


@strict(accept_kwargs=True)
@dataclass(repr=False)
class EsmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ESMModel`]. It is used to instantiate a ESM model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ESM
    [facebook/esm-1b](https://huggingface.co/facebook/esm-1b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the ESM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ESMModel`].
        mask_token_id (`int`, *optional*):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary. This must be included in the config because certain parts
            of the ESM code use this instead of the attention mask.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query", "rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        emb_layer_norm_before (`bool`, *optional*):
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout (`bool`, defaults to `False`):
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.

    Examples:

    ```python
    >>> from transformers import EsmModel, EsmConfig

    >>> # Initializing a ESM facebook/esm-1b style configuration
    >>> configuration = EsmConfig(vocab_size=33)

    >>> # Initializing a model from the configuration
    >>> model = EsmModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "esm"

    vocab_size: Optional[int] = None
    mask_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    hidden_size: Optional[int] = 768
    num_hidden_layers: Optional[int] = 12
    num_attention_heads: Optional[int] = 12
    intermediate_size: Optional[int] = 3072
    hidden_dropout_prob: Optional[float] = 0.1
    attention_probs_dropout_prob: Optional[float] = 0.1
    max_position_embeddings: Optional[int] = 1026
    initializer_range: Optional[float] = 0.02
    layer_norm_eps: Optional[float] = 1e-12
    position_embedding_type: Optional[str] = "absolute"
    use_cache: Optional[bool] = True
    emb_layer_norm_before: Optional[bool] = None
    token_dropout: Optional[bool] = False
    is_folding_model: Optional[bool] = False
    esmfold_config: Optional[Union[dict, EsmFoldConfig]] = None
    vocab_list: Optional[Sequence[str]] = None

    def __post_init__(self, **kwargs):
        if self.is_folding_model:
            if self.esmfold_config is None:
                logger.info("No esmfold_config supplied for folding model, using default values.")
                self.esmfold_config = EsmFoldConfig()
            elif isinstance(self.esmfold_config, dict):
                self.esmfold_config = EsmFoldConfig(**self.esmfold_config)

            if self.vocab_list is None:
                logger.warning("No vocab_list supplied for folding model, assuming the ESM-2 vocabulary!")
                self.vocab_list = get_default_vocab_list()
        else:
            self.esmfold_config = None
            self.vocab_list = None

        if self.esmfold_config is not None and getattr(self.esmfold_config, "use_esm_attn_map", False):
            raise ValueError("The HuggingFace port of ESMFold does not support use_esm_attn_map at this time!")

        super().__post_init__(**kwargs)


def get_default_vocab_list():
    return (
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
        "<null_1>",
        "<mask>",
    )


__all__ = ["EsmConfig"]
