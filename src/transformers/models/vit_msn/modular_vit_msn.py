# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViT MSN (masked siamese network) model - modular file inheriting from ViT."""

import torch
from torch import nn

from ... import initialization as init
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..vit.modeling_vit import (
    ViTAttention,
    ViTEmbeddings,
    ViTEncoder,
    ViTLayer,
    ViTMLP,
    ViTPatchEmbeddings,
    ViTPreTrainedModel,
)
from .configuration_vit_msn import ViTMSNConfig


class ViTMSNPatchEmbeddings(ViTPatchEmbeddings):
    pass


class ViTMSNEmbeddings(ViTEmbeddings):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    ViT MSN uses zeros initialization for cls_token and position_embeddings (vs ViT's randn).
    """

    def __init__(self, config: ViTMSNConfig, use_mask_token: bool = False) -> None:
        super().__init__(config, use_mask_token=use_mask_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))


class ViTMSNAttention(ViTAttention):
    pass


class ViTMSNMLP(ViTMLP):
    pass


class ViTMSNLayer(ViTLayer):
    pass


class ViTMSNEncoder(ViTEncoder):
    pass


class ViTMSNPreTrainedModel(ViTPreTrainedModel):
    base_model_prefix = "vit"

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights - ViT MSN uses normal (not trunc_normal) and zeros for embeddings."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, ViTMSNEmbeddings):
            init.zeros_(module.cls_token)
            init.zeros_(module.position_embeddings)
            if module.mask_token is not None:
                init.zeros_(module.mask_token)


@auto_docstring
class ViTMSNModel(ViTMSNPreTrainedModel):
    def __init__(self, config: ViTMSNConfig, use_mask_token: bool = False) -> None:
        r"""
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        """
        super().__init__(config)
        self.config = config
        self.embeddings = ViTMSNEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTMSNEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> ViTMSNPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMSNModel
        >>> import torch
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        >>> model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")
        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values is not None and pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )
        encoder_outputs: BaseModelOutput = self.encoder(embedding_output, attention_mask, **kwargs)

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        return BaseModelOutput(last_hidden_state=sequence_output)


@auto_docstring
class ViTMSNForImageClassification(ViTMSNPreTrainedModel):
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = ViTMSNModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        interpolate_pos_encoding: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMSNForImageClassification
        >>> import torch
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> torch.manual_seed(2)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read())).convert("RGB")

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        >>> model = ViTMSNForImageClassification.from_pretrained("facebook/vit-msn-small")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        tusker
        ```
        """
        outputs: BaseModelOutput = self.vit(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            attention_mask=attention_mask,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["ViTMSNModel", "ViTMSNForImageClassification", "ViTMSNPreTrainedModel"]
