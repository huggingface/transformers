# Copyright 2022 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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
"""OneFormer model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="shi-labs/oneformer_ade20k_swin_tiny")
@strict
class OneFormerConfig(PreTrainedConfig):
    r"""
    ignore_value (`int`, *optional*, defaults to 255):
        Values to be ignored in GT label while calculating loss.
    num_queries (`int`, *optional*, defaults to 150):
        Number of object queries.
    contrastive_weight (`float`, *optional*, defaults to 0.5):
        Weight for contrastive loss.
    contrastive_temperature (`float`, *optional*, defaults to 0.07):
        Initial value for scaling the contrastive logits.
    train_num_points (`int`, *optional*, defaults to 12544):
        Number of points to sample while calculating losses on mask predictions.
    oversample_ratio (`float`, *optional*, defaults to 3.0):
        Ratio to decide how many points to oversample.
    importance_sample_ratio (`float`, *optional*, defaults to 0.75):
        Ratio of points that are sampled via importance sampling..
    is_training (`bool`, *optional*, defaults to `False`):
        Whether to run in training or inference mode.
    output_auxiliary_logits (`bool`, *optional*, defaults to `True`):
        Whether to return intermediate predictions from transformer decoder.
    strides (`list`, *optional*, defaults to `[4, 8, 16, 32]`):
        List containing the strides for feature maps in the encoder.
    task_seq_len (`int`, *optional*, defaults to 77):
        Sequence length for tokenizing text list input.
    text_encoder_width (`int`, *optional*, defaults to 256):
        Hidden size for text encoder.
    text_encoder_context_length (`int`, *optional*, defaults to 77):
        Input sequence length for text encoder.
    text_encoder_num_layers (`int`, *optional*, defaults to 6):
        Number of layers for transformer in text encoder.
    text_encoder_vocab_size (`int`, *optional*, defaults to 49408):
        Vocabulary size for tokenizer.
    text_encoder_proj_layers (`int`, *optional*, defaults to 2):
        Number of layers in MLP for project text queries.
    text_encoder_n_ctx (`int`, *optional*, defaults to 16):
        Number of learnable text context queries.
    conv_dim (`int`, *optional*, defaults to 256):
        Feature map dimension to map outputs from the backbone.
    mask_dim (`int`, *optional*, defaults to 256):
        Dimension for feature maps in pixel decoder.
    hidden_dim (`int`, *optional*, defaults to 256):
        Dimension for hidden states in transformer decoder.
    encoder_feedforward_dim (`int`, *optional*, defaults to 1024):
        Dimension for FFN layer in pixel decoder.
    norm (`str`, *optional*, defaults to `"GN"`):
        Type of normalization.
    use_task_norm (`bool`, *optional*, defaults to `True`):
        Whether to normalize the task token.
    dim_feedforward (`int`, *optional*, defaults to 2048):
        Dimension for FFN layer in transformer decoder.
    pre_norm (`bool`, *optional*, defaults to `False`):
        Whether to normalize hidden states before attention layers in transformer decoder.
    enforce_input_proj (`bool`, *optional*, defaults to `False`):
        Whether to project hidden states in transformer decoder.
    query_dec_layers (`int`, *optional*, defaults to 2):
        Number of layers in query transformer.
    common_stride (`int`, *optional*, defaults to 4):
        Common stride used for features in pixel decoder.

    Examples:
    ```python
    >>> from transformers import OneFormerConfig, OneFormerModel

    >>> # Initializing a OneFormer shi-labs/oneformer_ade20k_swin_tiny configuration
    >>> configuration = OneFormerConfig()
    >>> # Initializing a model (with random weights) from the shi-labs/oneformer_ade20k_swin_tiny style configuration
    >>> model = OneFormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "oneformer"
    sub_configs = {"backbone_config": AutoConfig}
    attribute_map = {"hidden_size": "hidden_dim", "num_hidden_layers": "decoder_layers"}

    backbone_config: dict | PreTrainedConfig | None = None
    ignore_value: int = 255
    num_queries: int = 150
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    contrastive_weight: float = 0.5
    contrastive_temperature: float = 0.07
    train_num_points: int = 12544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    layer_norm_eps: float = 1e-05
    is_training: bool = False
    use_auxiliary_loss: bool = True
    output_auxiliary_logits: bool = True
    strides: list[int] | tuple[int, ...] = (4, 8, 16, 32)
    task_seq_len: int = 77
    text_encoder_width: int = 256
    text_encoder_context_length: int = 77
    text_encoder_num_layers: int = 6
    text_encoder_vocab_size: int = 49408
    text_encoder_proj_layers: int = 2
    text_encoder_n_ctx: int = 16
    conv_dim: int = 256
    mask_dim: int = 256
    hidden_dim: int = 256
    encoder_feedforward_dim: int = 1024
    norm: str = "GN"
    encoder_layers: int = 6
    decoder_layers: int = 10
    use_task_norm: bool = True
    num_attention_heads: int = 8
    dropout: float | int = 0.1
    dim_feedforward: int = 2048
    pre_norm: bool = False
    enforce_input_proj: bool = False
    query_dec_layers: int = 2
    common_stride: int = 4

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="swin",
            default_config_kwargs={
                "drop_path_rate": 0.3,
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            **kwargs,
        )

        super().__post_init__(**kwargs)


__all__ = ["OneFormerConfig"]
