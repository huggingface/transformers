# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch DINOv2 model."""

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...backbone_utils import BackboneMixin, filter_output_hidden_states
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput, BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, torch_int
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..swin.modeling_swin import SwinDropPath
from ..vit.modeling_vit import (
    ViTAttention,
    ViTForImageClassification,
    ViTPatchEmbeddings,
    ViTPreTrainedModel,
)
from .configuration_dinov2 import Dinov2Config


class Dinov2PatchEmbeddings(ViTPatchEmbeddings):
    pass


class Dinov2Embeddings(nn.Module):
    """Construct the CLS token, mask token, position and patch embeddings."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size)) if config.use_mask_token else None
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None and self.mask_token is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        embeddings = self.dropout(embeddings)
        return embeddings


class Dinov2Attention(ViTAttention):
    pass


class Dinov2LayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class Dinov2MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2SwiGLUFFN(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)


class Dinov2DropPath(SwinDropPath):
    pass


class Dinov2Layer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2Attention(config)
        self.layer_scale1 = Dinov2LayerScale(config)
        self.drop_path = Dinov2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_swiglu_ffn:
            self.mlp = Dinov2SwiGLUFFN(config)
        else:
            self.mlp = Dinov2MLP(config)
        self.layer_scale2 = Dinov2LayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output, _ = self.attention(hidden_states_norm, attention_mask=attention_mask, **kwargs)
        self_attention_output = self.layer_scale1(self_attention_output)
        hidden_states = self.drop_path(self_attention_output) + hidden_states

        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)
        layer_output = self.drop_path(layer_output) + hidden_states
        return layer_output


@auto_docstring
class Dinov2PreTrainedModel(ViTPreTrainedModel):
    config: Dinov2Config
    base_model_prefix = "dinov2"
    _no_split_modules = ["Dinov2Embeddings", "Dinov2Layer"]
    _can_record_outputs = {
        "hidden_states": Dinov2Layer,
        "attentions": Dinov2Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module) -> None:
        super()._init_weights(module)
        if isinstance(module, Dinov2LayerScale):
            init.constant_(module.lambda1, self.config.layerscale_value)


class Dinov2Encoder(Dinov2PreTrainedModel):
    def __init__(self, config: Dinov2Config):
        super().__init__(config)
        self.layer = nn.ModuleList([Dinov2Layer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, **kwargs)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class Dinov2Model(Dinov2PreTrainedModel):
    def __init__(self, config: Dinov2Config):
        super().__init__(config)
        self.config = config
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = None
        self.post_init()

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        bool_masked_pos: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        """
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )
        encoder_outputs: BaseModelOutput = self.encoder(embedding_output, attention_mask=attention_mask, **kwargs)
        sequence_output = self.layernorm(encoder_outputs.last_hidden_state)
        pooled_output = sequence_output[:, 0, :]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)


@auto_docstring(
    custom_intro="""
    Dinov2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    """
)
class Dinov2ForImageClassification(ViTForImageClassification):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dinov2 = Dinov2Model(config)
        self.classifier = (
            nn.Linear(config.hidden_size * 2, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs: BaseModelOutputWithPooling = self.dinov2(pixel_values, attention_mask=attention_mask, **kwargs)

        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0]
        patch_tokens = sequence_output[:, 1:]
        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        logits = self.classifier(linear_input)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Dinov2 backbone, to be used with frameworks like DETR and MaskFormer.
    """
)
class Dinov2Backbone(BackboneMixin, Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
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

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```"""
        embedding_output = self.embeddings(pixel_values)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )
        # The Backbone needs the per-stage hidden_states mid-forward to compute feature_maps,
        # so we iterate self.encoder.layer directly rather than going through self.encoder.forward.
        # @capture_outputs handles injection into the returned BackboneOutput.
        hidden_state = embedding_output
        hidden_states = (hidden_state,)
        for layer in self.encoder.layer:
            hidden_state = layer(hidden_state, attention_mask, **kwargs)
            hidden_states = hidden_states + (hidden_state,)

        feature_maps = []
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(batch_size, height // patch_size, width // patch_size, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps.append(hidden_state)

        return BackboneOutput(feature_maps=tuple(feature_maps))


__all__ = ["Dinov2ForImageClassification", "Dinov2Model", "Dinov2PreTrainedModel", "Dinov2Backbone"]
