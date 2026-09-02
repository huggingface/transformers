# Copyright Deepmind and The HuggingFace Inc. team. All rights reserved.
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
"""Perceiver model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="deepmind/language-perceiver")
@strict
class PerceiverConfig(PreTrainedConfig):
    r"""
    num_latents (`int`, *optional*, defaults to 256):
        The number of latents.
    d_latents (`int`, *optional*, defaults to 1280):
        Dimension of the latent embeddings.
    num_blocks (`int`, *optional*, defaults to 1):
        Number of blocks in the Transformer encoder.
    num_self_attends_per_block (`int`, *optional*, defaults to 26):
        The number of self-attention layers per block.
    num_self_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each self-attention layer in the Transformer encoder.
    num_cross_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each cross-attention layer in the Transformer encoder.
    qk_channels (`int`, *optional*):
        Dimension to project the queries + keys before applying attention in the cross-attention and self-attention
        layers of the encoder. Will default to preserving the dimension of the queries if not specified.
    v_channels (`int`, *optional*):
        Dimension to project the values before applying attention in the cross-attention and self-attention layers
        of the encoder. Will default to preserving the dimension of the queries if not specified.
    cross_attention_shape_for_attention (`str`, *optional*, defaults to `"kv"`):
        Dimension to use when downsampling the queries and keys in the cross-attention layer of the encoder.
    self_attention_widening_factor (`int`, *optional*, defaults to 1):
        Dimension of the feed-forward layer in the cross-attention layer of the Transformer encoder.
    cross_attention_widening_factor (`int`, *optional*, defaults to 1):
        Dimension of the feed-forward layer in the self-attention layers of the Transformer encoder.
    use_query_residual (`float`, *optional*, defaults to `True`):
        Whether to add a query residual in the cross-attention layer of the encoder.
    image_size (`int`, *optional*, defaults to 56):
        Size of the images after preprocessing, for [`PerceiverForImageClassificationLearned`].
    train_size (`list[int]`, *optional*, defaults to `[368, 496]`):
        Training size of the images for the optical flow model.
    num_frames (`int`, *optional*, defaults to 16):
        Number of video frames used for the multimodal autoencoding model.
    audio_samples_per_frame (`int`, *optional*, defaults to 1920):
        Number of audio samples per frame for the multimodal autoencoding model.
    samples_per_patch (`int`, *optional*, defaults to 16):
        Number of audio samples per patch when preprocessing the audio for the multimodal autoencoding model.
    output_shape (`list[int]`, *optional*, defaults to `[1, 16, 224, 224]`):
        Shape of the output (batch_size, num_frames, height, width) for the video decoder queries of the multimodal
        autoencoding model. This excludes the channel dimension.
    output_num_channels (`int`, *optional*, defaults to 512):
        Number of output channels for each modalitiy decoder.

    Example:

    ```python
    >>> from transformers import PerceiverModel, PerceiverConfig

    >>> # Initializing a Perceiver deepmind/language-perceiver style configuration
    >>> configuration = PerceiverConfig()

    >>> # Initializing a model from the deepmind/language-perceiver style configuration
    >>> model = PerceiverModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "perceiver"

    num_latents: int = 256
    d_latents: int = 1280
    d_model: int = 768
    num_blocks: int = 1
    num_self_attends_per_block: int = 26
    num_self_attention_heads: int = 8
    num_cross_attention_heads: int = 8
    qk_channels: int | None = None
    v_channels: int | None = None
    cross_attention_shape_for_attention: str = "kv"
    self_attention_widening_factor: int = 1
    cross_attention_widening_factor: int = 1
    hidden_act: str = "gelu"
    attention_probs_dropout_prob: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_query_residual: bool = True
    vocab_size: int = 262
    max_position_embeddings: int = 2048
    image_size: int | list[int] | tuple[int, int] = 56
    train_size: list[int] | tuple[int, ...] = (368, 496)
    num_frames: int = 16
    audio_samples_per_frame: int = 1920
    samples_per_patch: int = 16
    output_shape: list[int] | tuple[int, ...] = (1, 16, 224, 224)
    output_num_channels: int = 512
    _label_trainable_num_channels: int = 1024


__all__ = ["PerceiverConfig"]
