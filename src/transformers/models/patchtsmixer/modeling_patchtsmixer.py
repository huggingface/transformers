# coding=utf-8
# Copyright 2023 IBM and HuggingFace Inc. team. All Rights Reserved.
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
""" PyTorch PatchTSMixer model."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_patchtsmixer import PatchTSMixerConfig
from .layers import (
    ForecastHead,
    InjectRevinStatistics4D,
    LinearHead,
    Patch,
    PatchMasking,
    PatchTSMixer,
    PretrainHead,
    RevIN,
    set_seed,
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSMixerConfig"


PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtsmixer-etth1-pretrain",
    # See all PatchTST models at https://huggingface.co/models?filter=patchtsmixer
]


PATCHTSMIXER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSMixerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PATCHTSMIXER_INPUTS_DOCSTRING = r"""
    Args:
        context_values (`torch.FloatTensor` of shape `(batch_size, seq_length, input_size)`):
            Context values of the time series. For a pretraining task, this denotes the input time series to predict
            the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
            for classification or regression tasks, it denotes the appropriate context values of the time series.

            For univariate time series, `input_size` dimension should be 1. For multivariate time series, it is > 1.

        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, input_size)` for forecasting,
            `(batch_size, n_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target values
            of the time series, that serve as labels for the model. The `target_values` is what the Transformer needs
            during training to learn to output, given the `context_values`. Note that, this is NOT required for a
            pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, input_size)`. Even if we want to forecast
            only specific channels by setting the indices in `forecast_channel_indices` parameter, pass the target data
            with all channels, as channel Filtering for both prediction and target will be manually applied before the
            loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, n_targets)`.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.
"""


class PatchTSMixerPreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = PatchTSMixerConfig
    base_model_prefix = "model"
    main_input_name = "context_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights"""
        # print("Module = ", module)
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PatchTSMixerEncoder)):
            module.gradient_checkpointing = value


@dataclass
class PatchTSMixerEncoderOutputWithNoAttention(ModelOutput):
    """
    Base class for `PatchTSMixerEncoderOutputWithNoAttention`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, num_features)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    """
    Encoder for PatchTSMixer which inputs patched time-series and outputs patched embeddings.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.encoder = PatchTSMixer(
            num_patches=config.num_patches,
            patch_size=config.patch_len,
            in_channels=config.input_size,
            num_features=config.num_features,
            expansion_factor=config.expansion_factor,
            num_layers=config.num_layers,
            dropout=config.dropout,
            mode=config.mode,
            gated_attn=config.gated_attn,
            self_attn=config.self_attn,
            self_attn_heads=config.self_attn_heads,
            norm_mlp=config.norm_mlp,
            use_pe=config.use_pe,
            pe=config.pe,
            learn_pe=config.learn_pe,
        )

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(output_type=PatchTSMixerEncoderOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, context_values: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> PatchTSMixerEncoderOutputWithNoAttention:
        r"""
        Args:
        context_values (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, patch_len)`):
            Patched input context.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        Returns:
        """

        # context_values: [bs  x n_vars x num_patches x patch_len]
        # return: [bs x n_vars x num_patches x num_features]
        last_hidden_state, hidden_states = self.encoder(context_values, output_hidden_states=output_hidden_states)
        return PatchTSMixerEncoderOutputWithNoAttention(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states
        )


@dataclass
class PatchTSMixerModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, num_features)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patched_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_len)`):
            Patched input data to the model.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*):
            Bool Tensor indicating True in masked patches and False otherwise.
        revin_mean: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
        revin_std_dev: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patched_input: torch.FloatTensor = None
    mask: Optional[torch.FloatTensor] = None
    revin_mean: Optional[torch.FloatTensor] = None
    revin_stdev: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The PatchTSMixer Model for time-series forecasting.",
    PATCHTSMIXER_START_DOCSTRING,
)
class PatchTSMixerModel(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig, mask_input: bool = False):
        super().__init__(config)

        set_seed(config.seed_number)

        self.encoder = PatchTSMixerEncoder(config)
        self.patching = Patch(config.seq_len, patch_len=config.patch_len, stride=config.stride)

        if mask_input is True:
            self.masking = PatchMasking(
                mask_type=config.mask_type,
                mask_ratio=config.mask_ratio,
                mask_patches=config.mask_patches,
                mask_patch_ratios=config.mask_patch_ratios,
                channel_consistent_masking=config.channel_consistent_masking,
                d_size=config.d_size,
                cv_channel_indices=None,
                mask_value=config.mask_value,
            )
        else:
            self.masking = None

        if config.revin is True:
            self.revin = RevIN()
        else:
            self.revin = None

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(output_type=PatchTSMixerModelOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, context_values: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> PatchTSMixerModelOutputWithNoAttention:
        r"""
        Args:
        context_values (`torch.FloatTensor` of shape `(batch_size, seq_length, input_size)`):
            Context values of the time series. For a pretraining task, this denotes the input time series to predict
            the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
            for classification or regression tasks, it denotes the appropriate context values of the time series.

            For univariate time series, `input_size` dimension should be 1. For multivariate time series, it is > 1.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.


        Returns:

        """

        revin_mean = None
        revin_stdev = None
        mask = None

        if self.revin is not None:
            context_values = self.revin(context_values, mode="norm")  # x: tensor [bs x seq_len x input_size]
            revin_mean = self.revin.mean
            revin_stdev = self.revin.stdev

        patched_x = self.patching(context_values)  # [bs x input_size x num_patch x patch_len]

        enc_input = patched_x

        if self.masking is not None:
            enc_input, mask = self.masking(patched_x)
            # enc_input: [bs x input_size x num_patch x patch_len]
            # mask: [bs x input_size x num_patch]

        encoder_output = self.encoder(enc_input, output_hidden_states=output_hidden_states)

        return PatchTSMixerModelOutputWithNoAttention(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patched_input=patched_x,
            mask=mask,
            revin_mean=revin_mean,
            revin_stdev=revin_stdev,
        )


class PatchTSMixerMaskedPretrainHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.head = PretrainHead(
            num_patches=config.num_patches,
            num_features=config.num_features,
            input_size=config.input_size,
            patch_size=config.patch_len,
            head_dropout=config.head_dropout,
            mode=config.mode,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Refers the embedding output from the backbone

        Returns: `torch.FloatTensor` of shape `(batch_size, input_size, num_patches, patch_len)`

        """

        # context_values: [bs x n_vars x num_patches x num_features]
        # return: [bs x n_vars x num_patches x patch_len]

        return self.head(hidden_state)


@dataclass
class PatchTSMixerForMaskPreTrainingOutputWithNoAttention(ModelOutput):
    """
    Output type of [`PatchTSMixerForMaskPreTrainingOutputWithNoAttention`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, patch_len)`):
            Prediction output from the pretrain head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForMaskPretraining(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for mask pretraining.

    Args:
        config (`PatchTSMixerConfig`, *mandatory*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.model = PatchTSMixerModel(config, mask_input=True)
        self.head = PatchTSMixerMaskedPretrainHead(config)
        self.masked_loss = config.masked_loss
        if config.masked_loss is True:
            self.loss = torch.nn.MSELoss(reduction="none")
        else:
            self.loss = torch.nn.MSELoss(reduction="mean")

        if config.revin is True:
            self.revin = RevIN()
        else:
            self.revin = None

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    # @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=PatchTSMixerForMaskPreTrainingOutputWithNoAttention, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self, context_values: torch.Tensor, output_hidden_states: Optional[bool] = False, return_loss: bool = True
    ) -> PatchTSMixerForMaskPreTrainingOutputWithNoAttention:
        r"""
        Args:
        context_values (`torch.FloatTensor` of shape `(batch_size, seq_length, input_size)`):
            Context values of the time series. For a pretraining task, this denotes the input time series to predict
            the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
            for classification or regression tasks, it denotes the appropriate context values of the time series.

            For univariate time series, `input_size` dimension should be 1. For multivariate time series, it is > 1.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """

        # context_values: tensor [bs x seq_len x input_size]
        model_output = self.model(
            context_values, output_hidden_states=output_hidden_states
        )  # x.last_hidden_state: [bs x nvars x num_patch x num_features]
        x_hat = self.head(model_output.last_hidden_state)  # tensor [bs x nvars x num_patch x patch_len]

        if return_loss is True:
            loss_val = self.loss(x_hat, model_output.patched_input)
        else:
            loss_val = None

        # calculate masked_loss
        if self.masked_loss is True and loss_val is not None:
            loss_val = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        return PatchTSMixerForMaskPreTrainingOutputWithNoAttention(
            prediction_logits=x_hat,  # tensor [bs x nvars x num_patch x patch_len]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )


class PatchTSMixerForecastHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.head = ForecastHead(
            num_patches=config.num_patches,
            in_channels=config.input_size,
            patch_size=config.patch_len,
            num_features=config.num_features,
            forecast_len=config.forecast_len,
            head_dropout=config.head_dropout,
            mode=config.mode,
            forecast_channel_indices=config.forecast_channel_indices,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Backbone embedding.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, forecast_len, input_size)` or `(batch_size, forecast_len,
            forecast_channels)`:
        Forecast output. If forecast_channel_indices is not None, then only the channel indices to be forecasted will
        be returned.
        """

        # x: [bs x input_size x num_patches x num_features]
        # return: [bs x forecast_len x input_size]

        return self.head(hidden_state, y=None)


@dataclass
class PatchTSMixerForForecastOutputWithNoAttention(ModelOutput):
    """
    Output type of [`PatchTSMixerForForecastOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, forecast_len, input_size)`):
            Prediction output from the forecast head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForForecasting(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for forecasting application.

    Args:
        config (`PatchTSMixerConfig`, *mandatory*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerForecastHead(config)
        self.loss = torch.nn.MSELoss(reduction="mean")
        if config.revin is True:
            self.revin = RevIN(denorm_channels=config.forecast_channel_indices)
        else:
            self.revin = None

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForForecastOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        context_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForForecastOutputWithNoAttention:
        r"""

        Returns:

        """

        # context_values: tensor [bs x seq_len x input_size]
        model_output = self.model(
            context_values,
            output_hidden_states=output_hidden_states,
        )  # model_output: [bs x nvars x num_patch x num_features]

        y_hat = self.head(
            model_output.last_hidden_state,
        )  # tensor [bs x forecast_len x input_size]

        if self.revin is not None:
            self.revin.set_statistics(mean=model_output.revin_mean, stdev=model_output.revin_stdev)
            y_hat = self.revin(y_hat, mode="denorm")

        if target_values is not None and return_loss is True:
            if self.config.forecast_channel_indices is not None:
                loss_val = self.loss(y_hat, target_values[..., self.config.forecast_channel_indices])
            else:
                loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForForecastOutputWithNoAttention(
            prediction_logits=y_hat,  # tensor [bs x forecast_len x input_size]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )


class PatchTSMixerClassificationHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.head = LinearHead(
            num_patches=config.num_patches,
            in_channels=config.input_size,
            num_features=config.num_features,
            head_dropout=config.head_dropout,
            output_dim=config.n_classes,
            output_range=config.output_range,
            head_agg=config.head_agg,
            mode=config.mode,
        )

    def forward(self, hidden_state) -> torch.Tensor:
        """
        Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Refers the embedding output from the backbone.

        Returns: `torch.FloatTensor` of shape `(batch_size, no_classes)`

        """

        # x: [bs x nvars x num_patch x num_features]
        # output: [bs x n_classes]

        return self.head(hidden_state, y=None)


@dataclass
class PatchTSMixerForClassificationOutputWithNoAttention(ModelOutput):
    """
    Output type of [`PatchTSMixerForClassificationOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, n_classes)`):
            Prediction output from the classfication head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForClassification(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for classification application.

    Args:
        config (`PatchTSMixerConfig`, *mandatory*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerClassificationHead(config)
        self.loss = torch.nn.CrossEntropyLoss()

        if config.revin is True:
            if config.mode == "flatten":
                raise Exception("revin is not enabled for classification task when mode == flatten")
            self.inject_revin = InjectRevinStatistics4D(
                num_features=config.num_features, num_patches=config.num_patches
            )

        else:
            self.inject_revin = None

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=PatchTSMixerForClassificationOutputWithNoAttention, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        context_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForClassificationOutputWithNoAttention:
        r"""

        Returns:

        """

        model_output = self.model(
            context_values,
            output_hidden_states=output_hidden_states,
        )  # x: [bs x nvars x num_patch x num_features]

        if self.inject_revin is not None:
            revin_statistics = (
                model_output.revin_mean,
                model_output.revin_stdev,
            )  # revin_mean,revin_stddev: [bs x 1 x n_channels]
            model_output.last_hidden_state = self.inject_revin(
                model_output.last_hidden_state, revin_statistics
            )  # x: [bs x nvars x num_patch x num_features]

        y_hat = self.head(model_output.last_hidden_state)  # tensor [bs x n_labels]

        if target_values is not None and return_loss is True:
            loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForClassificationOutputWithNoAttention(
            prediction_logits=y_hat,  # tensor [bs x n_labels]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )


class PatchTSMixerRegressionHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.head = LinearHead(
            num_patches=config.num_patches,
            in_channels=config.input_size,
            num_features=config.num_features,
            head_dropout=config.head_dropout,
            output_dim=config.n_targets,
            output_range=config.output_range,
            head_agg=config.head_agg,
            mode=config.mode,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Refers the embedding output from the backbone.

        Returns: `torch.FloatTensor` of shape `(batch_size, n_targets)`

        """

        # hidden_state: [bs x input_size x num_patches x num_features]
        # return: [bs x n_targets]
        return self.head(hidden_state, y=None)


@dataclass
class PatchTSMixerForRegressionOutputWithNoAttention(ModelOutput):
    """
    Output type of [`PatchTSMixerForRegressionOutputWithNoAttention`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, n_targets)`):
            Prediction output from the regression head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForRegression(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for regression application.

    Args:
        config (`PatchTSMixerConfig`, *mandatory*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerRegressionHead(config)
        self.loss = torch.nn.MSELoss(reduction="mean")

        if config.revin is True:
            if config.mode == "flatten":
                raise Exception("revin is not enabled for regression task when mode == flatten")
            self.inject_revin = InjectRevinStatistics4D(
                num_features=config.num_features, num_patches=config.num_patches
            )
        else:
            self.inject_revin = None

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=PatchTSMixerForRegressionOutputWithNoAttention, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        context_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForRegressionOutputWithNoAttention:
        r"""

        Returns:

        """

        # context_values: tensor [bs x seq_len x input_size]
        # target_values: tensor [bs x n_targets]

        model_output = self.model(
            context_values,
            output_hidden_states=output_hidden_states,
        )  # model_output: [bs x nvars x num_patch x num_features]

        if self.inject_revin is not None:
            revin_statistics = (
                model_output.revin_mean,
                model_output.revin_stdev,
            )  # revin_mean,revin_stddev: [bs x 1 x n_channels]
            model_output.last_hidden_state = self.inject_revin(
                model_output.last_hidden_state, revin_statistics
            )  # x: [bs x nvars x num_patch x num_features]

        y_hat = self.head(model_output.last_hidden_state)  # tensor [bs x n_targets]

        if target_values is not None and return_loss is True:
            loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForRegressionOutputWithNoAttention(
            prediction_logits=y_hat,  # tensor [bs x n_targets]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )
