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
"""Whisper model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


# fmt: off
NON_SPEECH_TOKENS = [
    1, 2, 7, 8, 9, 10, 14, 25,
    26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
    63, 90, 91, 92, 93, 357, 366, 438, 532, 685,
    705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377,
    1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211,
    4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786,
    11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791,
    17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409,
    34949, 40283, 40493, 40549, 47282, 49146, 50257, 50359, 50360, 50361
]
NON_SPEECH_TOKENS_MULTI = [
    1, 2, 7, 8, 9, 10, 14, 25,
    26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
    63, 90, 91, 92, 93, 359, 503, 522, 542, 873,
    893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627,
    3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647,
    7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793,
    14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675,
    22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865,
    42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362
]
# fmt: on


@auto_docstring(checkpoint="openai/whisper-tiny")
@strict
class WhisperConfig(PreTrainedConfig):
    r"""
    max_source_positions (`int`, *optional*, defaults to 1500):
        The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
    max_target_positions (`int`, *optional*, defaults to 448):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    suppress_tokens (`list[int]`, *optional*):
        A list containing the non-speech tokens that will be used by the logit processor in the `generate`
        function. NON_SPEECH_TOKENS and NON_SPEECH_TOKENS_MULTI each correspond to the `english-only` and the
        `multilingual` model.
    begin_suppress_tokens (`list[int]`, *optional*, defaults to `[220,50256]`):
        A list containing tokens that will be suppressed at the beginning of the sampling process. Initialized as
        the token for `" "` (`blank_token_id`) and the `eos_token_id`
    use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
        Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
        instance of [`WhisperForAudioClassification`].
    classifier_proj_size (`int`, *optional*, defaults to 256):
        Dimensionality of the projection before token mean-pooling for classification. Only relevant when using an
        instance of [`WhisperForAudioClassification`].
    apply_spec_augment (`bool`, *optional*, defaults to `False`):
        Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
        [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
        Recognition](https://huggingface.co/papers/1904.08779).
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
    mask_feature_min_masks (`int`, *optional*, defaults to 0):
        The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
        step, irrespectively of `mask_feature_prob`. Only relevant if
        `mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks`.
    median_filter_width (`int`, *optional*, defaults to 7):
        Width of the median filter used to smoothen to cross-attention outputs when computing token timestamps.
        Should be an odd number.

    Example:

    ```python
    >>> from transformers import WhisperConfig, WhisperModel

    >>> # Initializing a Whisper tiny style configuration
    >>> configuration = WhisperConfig()

    >>> # Initializing a model (with random weights) from the tiny style configuration
    >>> model = WhisperModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "whisper"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_key_value_heads": "encoder_attention_heads",
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 51865
    num_mel_bins: int = 80
    encoder_layers: int = 4
    encoder_attention_heads: int = 6
    decoder_layers: int = 4
    decoder_attention_heads: int = 6
    decoder_ffn_dim: int = 1536
    encoder_ffn_dim: int = 1536
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    decoder_start_token_id: int = 50257
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 384
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    scale_embedding: bool = False
    max_source_positions: int = 1500
    max_target_positions: int = 448
    pad_token_id: int | None = 50256
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    suppress_tokens: list | None = None
    begin_suppress_tokens: list[int] | tuple[int, ...] | None = (220, 50256)
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256
    apply_spec_augment: bool = False
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    median_filter_width: int = 7
    tie_word_embeddings: bool = True


__all__ = ["WhisperConfig"]
