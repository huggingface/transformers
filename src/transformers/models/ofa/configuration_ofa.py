# coding=utf-8
# Copyright 2022 The OFA-Sys Team. All rights reserved.
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
""" OFA model configuration"""
import warnings

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

OFA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "OFA-Sys/OFA-tiny": "https://huggingface.co/OFA-Sys/OFA-tiny/blob/main/config.json",
    "OFA-Sys/OFA-medium": "https://huggingface.co/OFA-Sys/OFA-medium/blob/main/config.json",
    "OFA-Sys/OFA-base": "https://huggingface.co/OFA-Sys/OFA-base/blob/main/config.json",
    "OFA-Sys/OFA-large": "https://huggingface.co/OFA-Sys/OFA-large/blob/main/config.json",
    # See all OFA models at https://huggingface.co/models?filter=ofa
}


class OFAConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~OFAModel`]. It is used to instantiate an OFA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the OFA [ofa-base](https://huggingface.co/ofa-base)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the OFA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~OFAModel`] or [`~TFOFAModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimension of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop: (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop: (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    model_type = "ofa"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=59457,
        max_position_embeddings=1024,
        encoder_layers=4,
        encoder_ffn_dim=512 * 4,
        encoder_attention_heads=8,
        decoder_layers=4,
        decoder_ffn_dim=512 * 4,
        decoder_attention_heads=8,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=512,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        forced_eos_token_id=2,
        encoder_normalize_before=True,
        decoder_normalize_before=True,
        normformer=True,
        encoder_drop_path_rate=0.0,
        decoder_drop_path_rate=0.0,
        layernorm_embedding=True,
        patch_layernorm_embedding=True,
        resnet_type="resnet101",
        resnet_model_path=None,
        resnet_drop_path_rate=0.0,
        token_bucket_size=256,
        image_bucket_size=42,
        add_type_embedding=True,
        share_decoder_input_output_embed=True,
        attn_scale_factor=2.0,
        code_layernorm_embedding=True,
        code_image_size=128,
        entangle_position_embedding=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.encoder_normalize_before = encoder_normalize_before
        self.decoder_normalize_before = decoder_normalize_before
        self.normformer = normformer
        self.encoder_drop_path_rate = encoder_drop_path_rate
        self.decoder_drop_path_rate = decoder_drop_path_rate
        self.layernorm_embedding = layernorm_embedding
        self.patch_layernorm_embedding = patch_layernorm_embedding
        self.resnet_type = resnet_type
        self.resnet_model_path = resnet_model_path
        self.resnet_drop_path_rate = resnet_drop_path_rate
        self.token_bucket_size = token_bucket_size
        self.image_bucket_size = image_bucket_size
        self.add_type_embedding = add_type_embedding
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.attn_scale_factor = attn_scale_factor
        self.code_layernorm_embedding = code_layernorm_embedding
        self.code_image_size = code_image_size
        self.entangle_position_embedding = entangle_position_embedding

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        # ensure backward compatibility for BART CNN models
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )
