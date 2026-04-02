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

from typing import Union

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ...utils.type_validators import interval, is_divisible_by


logger = logging.get_logger(__name__)


@strict
class StructureModuleConfig(PreTrainedConfig):
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

    sequence_dim: int | None = 384
    pairwise_dim: int | None = 128
    ipa_dim: int | None = 16
    resnet_dim: int | None = 128
    num_heads_ipa: int | None = 12
    num_qk_points: int | None = 4
    num_v_points: int | None = 8
    dropout_rate: float | None = 0.1
    num_blocks: int | None = 8
    num_transition_layers: int | None = 1
    num_resnet_blocks: int | None = 2
    num_angles: int | None = 7
    trans_scale_factor: int | None = 10
    epsilon: float | None = 1e-8
    inf: float | None = 1e5


@strict
class TrunkConfig(PreTrainedConfig):
    sub_configs = {"structure_module": StructureModuleConfig}

    num_blocks: int | None = 48
    sequence_state_dim: int | None = 1024
    pairwise_state_dim: int | None = is_divisible_by(divisor=2)(default=128)
    sequence_head_width: int | None = 32
    pairwise_head_width: int | None = 32
    position_bins: int | None = 32
    dropout: float | int | None = interval(max=0.4)(default=0.0)
    layer_drop: float | int | None = 0.0
    cpu_grad_checkpoint: bool | None = False
    max_recycles: int | None = interval(min=0)(default=4)
    chunk_size: int | None = 128
    structure_module: Union[dict, "StructureModuleConfig"] | None = None

    def __post_init__(self, **kwargs):
        if self.structure_module is None:
            self.structure_module = StructureModuleConfig()
        elif isinstance(self.structure_module, dict):
            self.structure_module = StructureModuleConfig(**self.structure_module)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
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


@strict
class EsmFoldConfig(PreTrainedConfig):
    sub_configs = {"trunk": TrunkConfig}

    esm_type: str | None = None
    fp16_esm: bool | None = True
    use_esm_attn_map: bool | None = False
    esm_ablate_pairwise: bool | None = False
    esm_ablate_sequence: bool | None = False
    esm_input_dropout: float | int | None = 0.0
    embed_aa: bool | None = True
    bypass_lm: bool | None = False
    lddt_head_hid_dim: int | None = 128
    trunk: Union[dict, "TrunkConfig"] | None = None

    def __post_init__(self, **kwargs):
        if self.trunk is None:
            self.trunk = TrunkConfig()
        elif isinstance(self.trunk, dict):
            self.trunk = TrunkConfig(**self.trunk)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/esm-1b")
@strict
class EsmConfig(PreTrainedConfig):
    r"""
    mask_token_id (`int`, *optional*):
        The index of the mask token in the vocabulary. This must be included in the config because of the
        "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose either `"absolute"` or "rotary"`.
    emb_layer_norm_before (`bool`, *optional*):
        Whether to apply layer normalization after embeddings but before the main stem of the network.
    token_dropout (`bool`, defaults to `False`):
        When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.
    is_folding_model (`bool`, defaults to `False`):
        When this is enabled, ESMFold model will be initialized.
    esmfold_config (`dict`, *optional*):
        Configuration to initiate the ESMFold module.
    vocab_list (`list`, *optional*):
        List of the vocabulary items.

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
    sub_configs = {"esmfold_config": EsmFoldConfig}

    vocab_size: int | None = None
    mask_token_id: int | None = None
    pad_token_id: int | None = None
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float | None = 0.1
    attention_probs_dropout_prob: float | None = 0.1
    max_position_embeddings: int = 1026
    initializer_range: float = 0.02
    layer_norm_eps: float | None = 1e-12
    position_embedding_type: str | None = "absolute"
    use_cache: bool = True
    emb_layer_norm_before: bool | None = None
    token_dropout: bool | None = False
    is_folding_model: bool | None = False
    esmfold_config: dict | EsmFoldConfig | None = None
    vocab_list: list[str] | tuple[str, ...] | None = None
    is_decoder: bool | None = False
    add_cross_attention: bool | None = False
    tie_word_embeddings: bool = True

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
