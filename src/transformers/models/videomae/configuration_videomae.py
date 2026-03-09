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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="MCG-NJU/videomae-base")
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

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_frames=16,
        tubelet_size=2,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        use_mean_pooling=True,
        decoder_num_attention_heads=6,
        decoder_hidden_size=384,
        decoder_num_hidden_layers=4,
        decoder_intermediate_size=1536,
        norm_pix_loss=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_mean_pooling = use_mean_pooling

        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.norm_pix_loss = norm_pix_loss


__all__ = ["VideoMAEConfig"]
