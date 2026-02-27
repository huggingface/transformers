# Copyright 2026 Illuin Technology and contributors, and The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ... import initialization as init
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..modernbert.modeling_modernbert import ModernBertPredictionHead
from ..smolvlm.modeling_smolvlm import SmolVLMModel, SmolVLMPreTrainedModel


logger = logging.get_logger(__name__)


class ModernVBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ModernVBert`] model. It is used to
    instantiate a ModernVBert model according to the specified arguments and defines the model architecture.
    e.g. [ModernVBERT/modernvbert](https://huggingface.co/ModernVBERT/modernvbert).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    See the documentation for [`PretrainedConfig`] for more details.

    Args:
        text_config (`AutoConfig`, *optional*): Configuration for the text encoder.
        vision_config (`ModernVBertVisionConfig`, *optional*): Configuration for the vision encoder.
        image_token_id (`int | None`, *optional*, defaults to 50407): The token id reserved for image tokens inserted into the text stream.
        pixel_shuffle_factor (`int | None`, *optional*, defaults to 4): Scale factor used by any pixel-shuffle / upsampling operations in the vision head.
        initializer_range (`float | None`, *optional*, defaults to 0.02): The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_cutoff_factor (`float | None`, *optional*, defaults to 2.0): The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
        classifier_pooling (`Literal["cls", "mean"]`, *optional*, defaults to `"cls"`): The pooling strategy to use for classification tasks.
        classifier_dropout (`float | None`, *optional*, defaults to 0.0): The dropout probability for the classification head.
        classifier_bias (`bool | None`, *optional*, defaults to `False`): Whether to add a bias term to the classification head.

    Example:
    ```python
    >>> from transformers import ModernVBertConfig

    >>> # Initializing configuration
    >>> configuration = ModernVBertConfig()

    >>> # Initializing a model from the configuration (model class is implemented in
    >>> # `modernvbert.modeling_modernvbert`)

    >>> from transformers import ModernVBertModel
    >>> model = ModernVBertModel(configuration)

    >>> # Accessing the model configuration
    >>> cfg = model.config
    ```"""

    model_type = "modernvbert"
    sub_configs: dict[str, Any] = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int | None = 50407,
        pixel_shuffle_factor: int | None = 4,
        initializer_range: float | None = 0.02,
        initializer_cutoff_factor: float | None = 2.0,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout: float | None = 0.0,
        classifier_bias: bool | None = False,
        **kwargs,
    ):
        if classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {classifier_pooling}.'
            )

        if text_config is None:
            text_config = CONFIG_MAPPING["modernbert"]()
        elif isinstance(text_config, dict):
            text_config = CONFIG_MAPPING["modernbert"](**text_config)
        self.text_config = text_config

        if vision_config is None:
            vision_config = CONFIG_MAPPING["siglip_vision_model"]()
        elif isinstance(vision_config, dict):
            vision_config = CONFIG_MAPPING["siglip_vision_model"](**vision_config)
        self.vision_config = vision_config

        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias

        super().__init__(image_token_id=image_token_id, **kwargs)


@dataclass
class ModernVBertBaseModelOutput(BaseModelOutput):
    """
    Base class for ModernVBERT model's outputs.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: tuple[torch.FloatTensor] | None = None


@dataclass
class ModernVBertMaskedLMOutput(MaskedLMOutput):
    """
    Base class for ModernVBERT model's outputs with masked language modeling loss.
    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class ModernVBertConnector(nn.Module):
    """
    Connector module for ModernVBERT. It performs a pixel shuffle operation followed by a linear projection to match the text model's hidden size.
    Based on https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
    """

    def __init__(self, config):
        super().__init__()
        self.pixel_shuffle_factor = config.pixel_shuffle_factor
        self.modality_projection = nn.Linear(
            config.vision_config.hidden_size * (config.pixel_shuffle_factor**2),
            config.text_config.hidden_size,
            bias=False,
        )

    def pixel_shuffle(self, image_hidden_states, pixel_shuffle_factor):
        batch_size, seq_length, embed_dim = image_hidden_states.size()
        height = width = int(seq_length**0.5)
        image_hidden_states = image_hidden_states.view(batch_size, height, width, embed_dim)
        image_hidden_states = image_hidden_states.view(
            batch_size, height, int(width / pixel_shuffle_factor), embed_dim * pixel_shuffle_factor
        )
        image_hidden_states = image_hidden_states.permute(0, 2, 1, 3)
        image_hidden_states = image_hidden_states.reshape(
            batch_size,
            int(width / pixel_shuffle_factor),
            int(height / pixel_shuffle_factor),
            embed_dim * (pixel_shuffle_factor**2),
        )
        image_hidden_states = image_hidden_states.permute(0, 2, 1, 3)
        return image_hidden_states.reshape(
            batch_size, int(seq_length / (pixel_shuffle_factor**2)), embed_dim * (pixel_shuffle_factor**2)
        )

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.pixel_shuffle_factor)
        return self.modality_projection(image_hidden_states)


@auto_docstring
class ModernVBertPreTrainedModel(SmolVLMPreTrainedModel):
    config_class = ModernVBertConfig
    _no_split_modules = []

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)

        def init_weight(module: nn.Module, std: float):
            cutoff_factor = getattr(self.config, "initializer_cutoff_factor", 2.0)
            init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.bias is not None:
                    init.zeros_(module.bias)

        if isinstance(module, ModernVBertConnector):
            out_std = self.config.initializer_range / math.sqrt(2.0 * self.config.text_config.num_hidden_layers)
            init_weight(module.modality_projection, out_std)
        elif isinstance(module, ModernVBertForMaskedLM):
            out_std = self.config.initializer_range / math.sqrt(2.0 * self.config.text_config.num_hidden_layers)
            init_weight(module.lm_head, out_std)
        elif isinstance(
            module,
            (
                ModernVBertForSequenceClassification,
                ModernVBertForTokenClassification,
            ),
        ):
            final_out_std = self.config.initializer_range / math.sqrt(self.config.text_config.hidden_size)
            init_weight(module.classifier, final_out_std)


@auto_docstring(
    custom_intro="""
    ModernVBertModel is a model that combines a vision encoder (SigLIP) and a text encoder (ModernBert).

    ModernVBert is the base model of the visual retriver ColModernVBert, and was introduced in the following paper:
    [*ModernVBERT: Towards Smaller Visual Document Retrievers*](https://arxiv.org/abs/2510.01149).
    """
)
class ModernVBertModel(SmolVLMModel):
    def __init__(self, config: ModernVBertConfig):
        super().__init__(config)

        # init components
        self.connector = ModernVBertConnector(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.vision_model = AutoModel.from_config(config.vision_config)

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2)
            / (config.pixel_shuffle_factor**2)
        )

        # initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ModernVBertBaseModelOutput:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # Images processing
        if pixel_values is not None:
            image_hidden_states = self.get_image_features(
                pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask
            ).pooler_output

        # Merge image and text embeddings
        if image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids, inputs_embeds=inputs_embeds, image_hidden_states=image_hidden_states
            )

        # Language model pass
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        return ModernVBertBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class ModernVBertPredictionHead(ModernBertPredictionHead):
    pass


@auto_docstring
class ModernVBertForMaskedLM(ModernVBertPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.text_model.embeddings.tok_embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)

        self.vocab_size = config.text_config.vocab_size

        self.model = ModernVBertModel(config)
        self.projection_head = ModernVBertPredictionHead(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, self.vocab_size, bias=config.text_config.decoder_bias)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ModernVBertMaskedLMOutput:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            text_config.]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., text_config.]`.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        hidden_states = outputs[0]

        logits = self.lm_head(self.projection_head(hidden_states))

        loss = None
        if labels is not None:
            criterion = CrossEntropyLoss()
            loss = criterion(logits.view(-1, self.vocab_size), labels.view(-1))

        return ModernVBertMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The ModernVBert Model with a sequence classification head on top that performs pooling.
    """
)
class ModernVBertForSequenceClassification(ModernVBertPreTrainedModel):
    def __init__(self, config: ModernVBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernVBertModel(config)
        self.head = ModernVBertPredictionHead(config.text_config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.text_config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SequenceClassifierOutput:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            text_config.]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., text_config.]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ModernVBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.
    """
)
class ModernVBertForTokenClassification(ModernVBertPreTrainedModel):
    def __init__(self, config: ModernVBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = ModernVBertModel(config)
        self.head = ModernVBertPredictionHead(config.text_config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.text_config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="ModernVBERT/modernvbert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TokenClassifierOutput:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            text_config.]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., text_config.]`.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ModernVBertConfig",
    "ModernVBertPreTrainedModel",
    "ModernVBertModel",
    "ModernVBertForMaskedLM",
    "ModernVBertForSequenceClassification",
    "ModernVBertForTokenClassification",
]
