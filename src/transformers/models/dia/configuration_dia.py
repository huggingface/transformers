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
"""Dia model configuration"""

from typing import TYPE_CHECKING

from ...configuration_utils import PretrainedConfig
from ...utils import logging


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)


class DiaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DiaModel`]. It is used to instantiate a
    Dia model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Dia
    [nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 51865):
            Vocabulary size of the Dia model. Defines the number of different tokens that can be represented by the
            `decoder_input_ids` passed when calling [`DiaModel`]
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel features used per input features. Should correspond to the value used in the
            `DiaProcessor` class.
        encoder_layers (`int`, *optional*, defaults to 4):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 4):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_start_token_id (`int`, *optional*, defaults to 50257):
            Corresponds to the "<|startoftranscript|>" token, which is automatically used when no `decoder_input_ids`
            are provided to the `generate` function. It is used to guide the model`s generation process depending on
            the task.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 384):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to False):
            Scale embeddings by diving by sqrt(d_model).
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        max_target_positions (`int`, *optional*, defaults to 448):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        pad_token_id (`int`, *optional*, defaults to 50256):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 50256):
            Begin of stream token id.
        eos_token_id (`int`, *optional*, defaults to 50256):
            End of stream token id.
        suppress_tokens (`List[int]`, *optional*):
            A list containing the non-speech tokens that will be used by the logit processor in the `generate`
            function. NON_SPEECH_TOKENS and NON_SPEECH_TOKENS_MULTI each correspond to the `english-only` and the
            `multilingual` model.
        begin_suppress_tokens (`List[int]`, *optional*, defaults to `[220,50256]`):
            A list containing tokens that will be suppressed at the beginning of the sampling process. Initialized as
            the token for `" "` (`blank_token_id`) and the `eos_token_id`
        use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
            Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
            instance of [`DiaForAudioClassification`].
        classifier_proj_size (`int`, *optional*, defaults to 256):
            Dimensionality of the projection before token mean-pooling for classification. Only relevant when using an
            instance of [`DiaForAudioClassification`].
        apply_spec_augment (`bool`, *optional*, defaults to `False`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
            [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://arxiv.org/abs/1904.08779).
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procedure generates `mask_time_prob*len(time_axis)/mask_time_length` independent masks over the axis. If
            reasoning from the probability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment == True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2),:
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks''
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procedure generates `mask_feature_prob*len(feature_axis)/mask_time_length` independent masks over
            the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0),:
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            `mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks`.
        median_filter_width (`int`, *optional*, defaults to 7):
            Width of the median filter used to smoothen to cross-attention outputs when computing token timestamps.
            Should be an odd number.

    Example:

    ```python
    >>> from transformers import DiaConfig, DiaModel

    >>> # Initializing a Dia tiny style configuration
    >>> configuration = DiaConfig()

    >>> # Initializing a model (with random weights) from the tiny style configuration
    >>> model = DiaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dia"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_key_value_heads": "encoder_attention_heads",
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
    }

    def __init__(
        self,
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=4,
        encoder_attention_heads=6,
        decoder_layers=4,
        decoder_attention_heads=6,
        decoder_ffn_dim=1536,
        encoder_ffn_dim=1536,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        decoder_start_token_id=50257,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=384,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        scale_embedding=False,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,
        suppress_tokens=None,
        begin_suppress_tokens=[220, 50256],
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        apply_spec_augment=False,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        median_filter_width=7,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        # Audio Classification-specific parameters. Feel free to ignore for other classes.
        self.classifier_proj_size = classifier_proj_size
        self.use_weighted_layer_sum = use_weighted_layer_sum

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        self.median_filter_width = median_filter_width

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            **kwargs,
        )


__all__ = ["DiaConfig"]
