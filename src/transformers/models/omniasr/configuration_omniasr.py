# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""

Example of config from OmniASR-CTC-300M model

Wav2Vec2AsrConfig(
    encoder_config=Wav2Vec2EncoderConfig(
        model_dim=1024, 
        max_seq_len=4096, 
        feature_dim=512, 
        use_fbank=False, 
        first_pass_dropout_p=0.0, 
        layer_norm_features=False, 
        feature_extractor_layer_descs=[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)], 
        feature_extractor_bias=True, 
        feature_extractor_layer_norm_convs=True, 
        feature_grad_scale=0.1, 
        num_fbank_channels=0, 
        fbank_stride=0, 
        sample_fbank_every_k=0, 
        pos_encoder_type='conv', 
        pos_encoder_depth=1, 
        pos_conv_kernel_size=128, 
        num_pos_conv_groups=16, 
        use_conformer=False, 
        num_encoder_layers=24, 
        num_encoder_attn_heads=16, 
        ffn_inner_dim=4096, 
        dropout_p=0.0, 
        attn_dropout_p=0.0, 
        ffn_inner_dropout_p=0.1, 
        layer_drop_p=0.1, 
        norm_order=<TransformerNormOrder.PRE: 1>, 
        depthwise_conv_kernel_size=0
    ), 
    target_vocab_size=10288, 
    final_dropout_p=0.0, 
    use_masking=False, 
    temporal_mask_span_len=10, 
    max_temporal_mask_prob=0.0, 
    min_num_temporal_mask_spans=2, 
    spatial_mask_span_len=64, 
    max_spatial_mask_prob=0.0, 
    min_num_spatial_mask_spans=2
)

"""


from typing import Union

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class OmniASREncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniASREncoderConfig`]. It is used to instantiate
    a `OmniASREncoder` model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    TODO
    - docstrings for each parameter
    - rename to Transformer convention

    """

    model_type = "omniasr_encoder"

    def __init__(
        self,
        model_dim=1024, 
        max_seq_len=4096, 
        feature_dim=512, 
        use_fbank=False, 
        first_pass_dropout_p=0.0, 
        layer_norm_features=False, 

        # feature_extractor_layer_descs=[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)], 
        # feature_extractor_bias=True,
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_bias=True,

        feature_extractor_layer_norm_convs=True, 
        feature_grad_scale=0.1, 
        num_fbank_channels=0, 
        fbank_stride=0, 
        sample_fbank_every_k=0, 
        pos_encoder_type='conv', 
        pos_encoder_depth=1, 
        pos_conv_kernel_size=128, 
        num_pos_conv_groups=16, 
        use_conformer=False, 
        num_encoder_layers=24, 
        num_encoder_attn_heads=16, 
        ffn_inner_dim=4096, 
        dropout_p=0.0, 
        attn_dropout_p=0.0, 
        ffn_inner_dropout_p=0.1, 
        layer_drop_p=0.1, 
        layer_norm_pre=True, 
        depthwise_conv_kernel_size=0,
        initializer_range=0.02,
        **kwargs,
    ):
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.use_fbank = use_fbank
        self.first_pass_dropout_p = first_pass_dropout_p
        self.layer_norm_features = layer_norm_features

        # self.feature_extractor_layer_descs = feature_extractor_layer_descs
        # self.feature_extractor_bias = feature_extractor_bias
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias


        self.feature_extractor_layer_norm_convs = feature_extractor_layer_norm_convs
        self.feature_grad_scale = feature_grad_scale
        self.num_fbank_channels = num_fbank_channels
        self.fbank_stride = fbank_stride
        self.sample_fbank_every_k = sample_fbank_every_k
        self.pos_encoder_type = pos_encoder_type
        self.pos_encoder_depth = pos_encoder_depth
        self.pos_conv_kernel_size = pos_conv_kernel_size
        self.num_pos_conv_groups = num_pos_conv_groups
        self.use_conformer = use_conformer
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_attn_heads = num_encoder_attn_heads
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout_p = dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.ffn_inner_dropout_p = ffn_inner_dropout_p
        self.layer_drop_p = layer_drop_p
        # Whether layer normalization is applied at the beginning of each layer or after each layer's residuation connection: https://github.com/facebookresearch/fairseq2/blob/a510a839e007d2b036185b7b4ca76074d287c67e/src/fairseq2/models/transformer/norm_order.py#L12
        self.layer_norm_pre = layer_norm_pre
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size

        self.initializer_range = initializer_range

        super().__init__(**kwargs)


class OmniASRCTCConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniASRForCTC`]. It is used to instantiate a
    OmniASR CTC model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    """

    model_type = "omniasr_ctc"
    sub_configs = {"encoder_config": OmniASREncoderConfig}

    def __init__(
        self,
        vocab_size=10288,
        final_dropout_p=0.0, 
        use_masking=False, 
        temporal_mask_span_len=10, 
        max_temporal_mask_prob=0.0, 
        min_num_temporal_mask_spans=2, 
        spatial_mask_span_len=64, 
        max_spatial_mask_prob=0.0, 
        min_num_spatial_mask_spans=2,
        encoder_config: Union[dict, OmniASREncoderConfig] = None,
        # TODO check token ids, took from Wav2Vec2
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.final_dropout_p = final_dropout_p
        self.use_masking = use_masking
        self.temporal_mask_span_len = temporal_mask_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob
        self.min_num_temporal_mask_spans = min_num_temporal_mask_spans
        self.spatial_mask_span_len = spatial_mask_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob
        self.min_num_spatial_mask_spans = min_num_spatial_mask_spans

        if isinstance(encoder_config, dict):
            self.encoder_config = OmniASREncoderConfig(**encoder_config)
        elif encoder_config is None:
            self.encoder_config = OmniASREncoderConfig()

        self.encoder_config = self.encoder_config
        self.initializer_range = self.encoder_config.initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @classmethod
    def from_encoder_config(cls, encoder_config: OmniASREncoderConfig, **kwargs):
        r"""
        Instantiate a [`OmniASRCTCConfig`] (or a derived class) from omniASR encoder model configuration.

        Returns:
            [`OmniASRCTCConfig`]: An instance of a configuration object
        """

        return cls(encoder_config=encoder_config.to_dict(), **kwargs)


__all__ = ["OmniASRCTCConfig", "OmniASREncoderConfig"]