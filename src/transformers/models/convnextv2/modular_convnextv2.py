# Copyright 2023 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ConvNextV2 model."""

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...backbone_utils import filter_output_hidden_states
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..convnext.configuration_convnext import ConvNextConfig
from ..convnext.modeling_convnext import (
    ConvNextBackbone,
    ConvNextDropPath,
    ConvNextEmbeddings,
    ConvNextEncoder,
    ConvNextForImageClassification,
    ConvNextLayerNorm,
    ConvNextModel,
    ConvNextPreTrainedModel,
    ConvNextStage,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/convnextv2-tiny-1k-224")
@strict
class ConvNextV2Config(ConvNextConfig):
    r"""
    num_stages (`int`, *optional*, defaults to 4):
        The number of stages in the model.

    Example:

    ```python
    >>> from transformers import ConvNextV2Config, ConvNextV2Model

    >>> # Initializing a ConvNeXTV2 convnextv2-tiny-1k-224 style configuration
    >>> configuration = ConvNextV2Config()

    >>> # Initializing a model (with random weights) from the convnextv2-tiny-1k-224 style configuration
    >>> model = ConvNextV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "convnextv2"
    layer_scale_init_value = AttributeError()


class ConvNextV2GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # Compute and normalize global spatial feature maps
        global_features = torch.linalg.vector_norm(hidden_states, ord=2, dim=(1, 2), keepdim=True)
        norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + 1e-6)
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states

        return hidden_states


class ConvNextV2LayerNorm(ConvNextLayerNorm):
    pass


class ConvNextV2DropPath(ConvNextDropPath):
    pass


class ConvNextV2Layer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        # depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = ConvNextV2LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = ACT2FN[config.hidden_act]
        self.grn = ConvNextV2GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = ConvNextV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.dwconv(features)
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        features = features.permute(0, 2, 3, 1)
        features = self.layernorm(features)
        features = self.pwconv1(features)
        features = self.act(features)
        features = self.grn(features)
        features = self.pwconv2(features)
        # (batch_size, height, width, num_channels) -> (batch_size, num_channels, height, width)
        features = features.permute(0, 3, 1, 2)

        features = residual + self.drop_path(features)
        return features


class ConvNextV2Embeddings(ConvNextEmbeddings):
    pass


class ConvNextV2Stage(ConvNextStage):
    pass


@auto_docstring
class ConvNextV2PreTrainedModel(ConvNextPreTrainedModel):
    config: ConvNextV2Config
    base_model_prefix = "convnextv2"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["ConvNextV2Layer"]

    @torch.no_grad()
    def _init_weights(self, module):
        # Skip ConvNextPreTrainedModel layer-scale branch (IJEPA-style)
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, ConvNextV2GRN):
            init.zeros_(module.weight)
            init.zeros_(module.bias)


class ConvNextV2Encoder(ConvNextEncoder):
    pass


class ConvNextV2Model(ConvNextModel):
    pass


@auto_docstring(
    custom_intro="""
    ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """
)
class ConvNextV2ForImageClassification(ConvNextV2PreTrainedModel, ConvNextForImageClassification):
    accepts_loss_kwargs = False

    def __init__(self, config: ConvNextV2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.convnextv2 = ConvNextV2Model(config)
        if config.num_labels > 0:
            self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)
        else:
            self.classifier = nn.Identity()
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor | None = None, labels: torch.LongTensor | None = None, **kwargs
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs: BaseModelOutputWithPoolingAndNoAttention = self.convnextv2(pixel_values, **kwargs)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels=labels, pooled_logits=logits, config=self.config)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    ConvNeXT V2 backbone, to be used with frameworks like DETR and MaskFormer.
    """
)
class ConvNextV2Backbone(ConvNextBackbone):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = ConvNextV2Embeddings(config)
        self.encoder = ConvNextV2Encoder(config)
        self.num_features = [config.hidden_sizes[0]] + list(config.hidden_sizes)

        # Add layer norms to hidden states of out_features
        hidden_states_norms = {}
        for stage, num_channels in zip(self.out_features, self.channels):
            hidden_states_norms[stage] = ConvNextV2LayerNorm(num_channels, data_format="channels_first")
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
        >>> model = AutoBackbone.from_pretrained("facebook/convnextv2-tiny-1k-224")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs: BaseModelOutputWithNoAttention = self.encoder(embedding_output, **kwargs)
        hidden_states = encoder_outputs.hidden_states

        feature_maps = []
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                feature_maps.append(hidden_state)

        return BackboneOutput(feature_maps=tuple(feature_maps), hidden_states=hidden_states)


__all__ = [
    "ConvNextV2Config",
    "ConvNextV2PreTrainedModel",
    "ConvNextV2Model",
    "ConvNextV2ForImageClassification",
    "ConvNextV2Backbone",
]
