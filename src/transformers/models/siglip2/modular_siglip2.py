# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.siglip.configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    SiglipForImageClassification,
    SiglipModel,
    SiglipMultiheadAttentionPoolingHead,
    SiglipOutput,
    SiglipPreTrainedModel,
    SiglipTextModel,
    SiglipTextModelOutput,
    SiglipVisionModel,
    SiglipVisionModelOutput,
    SiglipVisionTransformer,
)

from ...modeling_attn_mask_utils import _prepare_4d_attention_mask


class Siglip2TextConfig(SiglipTextConfig):
    pass


class Siglip2VisionConfig(SiglipVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`Siglip2VisionModel`]. It is used to instantiate a
    Siglip2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip2
    [google/siglip2-base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        num_patches (`int`, *optional*, defaults to 256):
            The number of patches in the image with the size of (`patch_size`, `patch_size`).
            The image is resized to fill maximum of this number of patches, and to preserve
            the aspect ratio. In case the resulted number of patches is lower, the image is
            padded in "patch" dimension.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import Siglip2VisionConfig, Siglip2VisionModel

    >>> # Initializing a Siglip2VisionConfig with google/siglip2-base-patch16-naflex style configuration
    >>> configuration = Siglip2VisionConfig()

    >>> # Initializing a Siglip2VisionModel (with random weights) from the google/siglip2-base-patch16-naflex style configuration
    >>> model = Siglip2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        del self.image_size


class Siglip2Config(SiglipConfig):
    pass


class Siglip2VisionOutput(SiglipVisionModelOutput):
    pass


class Siglip2TextOutput(SiglipTextModelOutput):
    pass


class Siglip2Output(SiglipOutput):
    pass


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            spatial_shapes (`List[Tuple[int, int]]`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional resized and padded positional embeddings
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class Siglip2VisionTransformer(SiglipVisionTransformer):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    # Update: add `spatial_shapes` and `attention_mask`
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embeddings(pixel_values, spatial_shapes)

        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = attention_mask

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state, attention_mask) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Siglip2PreTrainedModel(SiglipPreTrainedModel):
    pass


class Siglip2TextModel(SiglipTextModel):
    pass


class Siglip2MultiheadAttentionPoolingHead(SiglipMultiheadAttentionPoolingHead):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__(config)
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Siglip2VisionModel(SiglipVisionModel):
    # Update: add `spatial_shapes` and `pixel_attention_mask`
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class Siglip2Model(SiglipModel):
    # Update: add `spatial_shapes` and `pixel_attention_mask`
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Use Siglip2Model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = vision_outputs.pooler_output

        return pooled_output

    # Update: add `spatial_shapes` and `pixel_attention_mask`
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Siglip2Output:
        # Use Siglip2 model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            # Adapted from https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/trainers/proj/image_text/siglip2.py#L287
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        return Siglip2Output(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class Siglip2ForImageClassification(SiglipForImageClassification):
    # Update: add `spatial_shapes` and `pixel_attention_mask`
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> ImageClassifierOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        # average pool the patch tokens
        if pixel_attention_mask is not None:
            pool_mask = pixel_attention_mask[..., None].to(sequence_output.device)
            sequence_output = torch.sum(sequence_output * pool_mask, dim=1) / torch.sum(pool_mask, dim=1)
        else:
            sequence_output = torch.mean(sequence_output, dim=1)

        # apply classifier
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Siglip2Config",
    "Siglip2TextConfig",
    "Siglip2VisionConfig",
    "Siglip2Model",
    "Siglip2PreTrainedModel",
    "Siglip2TextModel",
    "Siglip2VisionModel",
    "Siglip2ForImageClassification",
]
