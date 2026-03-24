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
"""VideoMAE model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="MCG-NJU/videomae-base")
@strict
class VideoMAEConfig(PreTrainedConfig):
    r"""
    num_frames (`int`, *optional*, defaults to 16):
        The number of frames in each video.
    tubelet_size (`int`, *optional*, defaults to 2):
        The number of tubelets.
    use_mean_pooling (`bool`, *optional*, defaults to `True`):
        Whether to mean pool the final hidden states instead of using the final hidden state of the [CLS] token.
    decoder_num_attention_heads (`int`, *optional*, defaults to 6):
        Number of attention heads for each attention layer in the decoder.
    decoder_hidden_size (`int`, *optional*, defaults to 384):
        Dimensionality of the decoder.
    decoder_num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of hidden layers in the decoder.
    decoder_intermediate_size (`int`, *optional*, defaults to 1536):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
    norm_pix_loss (`bool`, *optional*, defaults to `True`):
        Whether to normalize the target patch pixels.

    Example:

    ```python
    >>> from transformers import VideoMAEConfig, VideoMAEModel

    >>> # Initializing a VideoMAE videomae-base style configuration
    >>> configuration = VideoMAEConfig()

    >>> # Randomly initializing a model from the configuration
    >>> model = VideoMAEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "videomae"

    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    num_frames: int = 16
    tubelet_size: int = 2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True
    use_mean_pooling: bool = True
    decoder_num_attention_heads: int = 6
    decoder_hidden_size: int = 384
    decoder_num_hidden_layers: int = 4
    decoder_intermediate_size: int = 1536
    norm_pix_loss: bool = True


__all__ = ["VideoMAEConfig"]
