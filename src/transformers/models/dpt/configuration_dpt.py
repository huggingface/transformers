# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""DPT model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="Intel/dpt-large")
@strict
class DPTConfig(PreTrainedConfig):
    r"""
    is_hybrid (`bool`, *optional*, defaults to `False`):
        Whether to use a hybrid backbone. Useful in the context of loading DPT-Hybrid models.
    backbone_out_indices (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
        Indices of the intermediate hidden states to use from backbone.
    readout_type (`str`, *optional*, defaults to `"project"`):
        The readout type to use when processing the readout token (CLS token) of the intermediate hidden states of
        the ViT backbone. Can be one of [`"ignore"`, `"add"`, `"project"`].
        - "ignore" simply ignores the CLS token.
        - "add" passes the information from the CLS token to all other tokens by adding the representations.
        - "project" passes information to the other tokens by concatenating the readout to all other tokens before
          projecting the
        representation to the original feature dimension D using a linear layer followed by a GELU non-linearity.
    reassemble_factors (`list[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
        The up/downsampling factors of the reassemble layers.
    neck_hidden_sizes (`list[str]`, *optional*, defaults to `[96, 192, 384, 768]`):
        The hidden sizes to project to for the feature maps of the backbone.
    fusion_hidden_size (`int`, *optional*, defaults to 256):
        The number of channels before fusion.
    head_in_index (`int`, *optional*, defaults to -1):
        The index of the features to use in the heads.
    use_batch_norm_in_fusion_residual (`bool`, *optional*, defaults to `False`):
        Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
    use_bias_in_fusion_residual (`bool`, *optional*, defaults to `True`):
        Whether to use bias in the pre-activate residual units of the fusion blocks.
    add_projection (`bool`, *optional*, defaults to `False`):
        Whether to add a projection layer before the depth estimation head.
    use_auxiliary_head (`bool`, *optional*, defaults to `True`):
        Whether to use an auxiliary head during training.
    auxiliary_loss_weight (`float`, *optional*, defaults to 0.4):
        Weight of the cross-entropy loss of the auxiliary head.
    semantic_classifier_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the semantic classification head.
    backbone_featmap_shape (`list[int]`, *optional*, defaults to `[1, 1024, 24, 24]`):
        Used only for the `hybrid` embedding type. The shape of the feature maps of the backbone.
    neck_ignore_stages (`list[int]`, *optional*, defaults to `[0, 1]`):
        Used only for the `hybrid` embedding type. The stages of the readout layers to ignore.
    pooler_output_size (`int`, *optional*):
        Dimensionality of the pooler layer. If None, defaults to `hidden_size`.
    pooler_act (`str`, *optional*, defaults to `"tanh"`):
        The activation function to be used by the pooler.

    Example:

    ```python
    >>> from transformers import DPTModel, DPTConfig

    >>> # Initializing a DPT dpt-large style configuration
    >>> configuration = DPTConfig()

    >>> # Initializing a model from the dpt-large style configuration
    >>> model = DPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dpt"
    sub_configs = {"backbone_config": AutoConfig}

    # NOTE: some values are typed as `None` on purpose
    # DPT creates one of: backbone or the general model only
    # so official checkpoint saved them as `None`
    hidden_size: int = 768
    num_hidden_layers: None | int = 12
    num_attention_heads: int | None = 12
    intermediate_size: int | None = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int | None = 0.0
    attention_probs_dropout_prob: float | int | None = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float | None = 1e-12
    image_size: int | list[int] | tuple[int, int] | None = 384
    patch_size: int | list[int] | tuple[int, int] | None = 16
    num_channels: int | None = 3
    is_hybrid: bool = False
    qkv_bias: bool | None = True
    backbone_out_indices: list[int] | tuple[int, ...] | None = (2, 5, 8, 11)
    readout_type: str = "project"
    reassemble_factors: list[int | float] | tuple[int | float, ...] = (4, 2, 1, 0.5)
    neck_hidden_sizes: list[int] | tuple[int, ...] = (96, 192, 384, 768)
    fusion_hidden_size: int = 256
    head_in_index: int = -1
    use_batch_norm_in_fusion_residual: bool | None = False
    use_bias_in_fusion_residual: bool | None = None
    add_projection: bool = False
    use_auxiliary_head: bool | None = True
    auxiliary_loss_weight: float = 0.4
    semantic_loss_ignore_index: int = 255
    semantic_classifier_dropout: float | int = 0.1
    backbone_featmap_shape: list[int] | tuple[int, ...] | None = (1, 1024, 24, 24)
    neck_ignore_stages: list[int] | tuple[int, ...] = (0, 1)
    backbone_config: dict | PreTrainedConfig | None = None
    pooler_output_size: int | None = None
    pooler_act: str = "tanh"

    def __post_init__(self, **kwargs):
        if self.readout_type not in ["ignore", "add", "project"]:
            raise ValueError("Readout_type must be one of ['ignore', 'add', 'project']")

        if self.is_hybrid:
            if isinstance(self.backbone_config, dict):
                self.backbone_config.setdefault("model_type", "bit")

            self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
                backbone_config=self.backbone_config,
                default_config_type="bit",
                default_config_kwargs={
                    "global_padding": "same",
                    "layer_type": "bottleneck",
                    "depths": [3, 4, 9],
                    "out_features": ["stage1", "stage2", "stage3"],
                    "embedding_dynamic_padding": True,
                },
                **kwargs,
            )
            if self.readout_type != "project":
                raise ValueError("Readout type must be 'project' when using `DPT-hybrid` mode.")
        elif kwargs.get("backbone") is not None or self.backbone_config is not None:
            self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
                backbone_config=self.backbone_config,
                **kwargs,
            )
            self.backbone_out_indices = None

        self.backbone_featmap_shape = self.backbone_featmap_shape if self.is_hybrid else None
        self.neck_ignore_stages = self.neck_ignore_stages if self.is_hybrid else []
        self.pooler_output_size = self.pooler_output_size if self.pooler_output_size else self.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["DPTConfig"]
