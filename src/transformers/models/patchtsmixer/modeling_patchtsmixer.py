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

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
import random
import numpy as np

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ...modeling_outputs import ModelOutput 
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from .configuration_patchtsmixer import PatchTSMixerConfig

from .layers import (InjectRevinStatistics4D, LinearHead, PatchTSMixer, Patch,
                     PatchMasking, ForecastHead, PretrainHead, RevIN, set_seed)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSMixerConfig"


PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtsmixer-base",
    # See all PatchTST models at https://huggingface.co/models?filter=patchtst
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
        context_values (`torch.FloatTensor` of shape `(batch_size, seq_length, in_channels)`):
            Context values of the time series. For a pretraining task, this denotes the input time series
            to predict the masked portion. For a forecasting task, this denotes the history/past time
            series values. Similarly, for classification or regression tasks, it denotes the appropriate
            context values of the time series. 

            For univariate time series, `in_channels` dimension should be 1. For multivariate time 
            series, it is > 1.
            
        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, in_channels)` or 
            `(batch_size, forecast_length, len(forecast_channel_indices))` or
            `(batch_size, n_targets)`, or `(batch_size,)` *optional*):
            Target values of the time series, that serve as labels for the model. The `target_values` is what the
            Transformer needs during training to learn to output, given the `context_values`. Note that, this is 
            NOT required for a pretraining task.

            For a forecasting task, the shape can be `(batch_size, target_len, in_channels)` or 
            `(batch_size, forecast_length, len(forecast_channel_indices))`. Here, `target_len = forcast_len`,
            and output channels can be `in_channels` or `len(forecast_channel_indices)`.

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


class PatchTSMixerEncoderOutputWithNoAttention(ModelOutput):
    """
    Base class for `PatchTSMixerEncoderOutput`, with potential hidden states.
    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, height, width)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None,


class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        
        self.encoder = PatchTSMixer(
            num_patches=config.num_patches,
            patch_size=config.patch_len,
            in_channels=config.in_channels,
            num_features=config.num_features,
            expansion_factor=config.expansion_factor,
            num_layers=config.num_layers,
            dropout=config.dropout,
            mode=config.mode,
            gated_attn=config.gated_attn,
            self_attn=config.self_attn,
            self_attn_heads=config.self_attn_heads,
            norm_mlp=config.norm_mlp,
            use_pe = config.use_pe,
            pe = config.pe,
            learn_pe = config.learn_pe,
        )
        

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    def forward(
            self, 
            context_values: torch.Tensor, 
            output_hidden_states: Optional[bool] = False
        ) -> PatchTSMixerEncoderOutputWithNoAttention:
        """
        context_values: [bs  x n_vars x num_patches x patch_len]
        return: [bs x n_vars x num_patches x num_features]
        """
        last_hidden_state, hidden_states  = self.encoder(context_values, output_hidden_states = output_hidden_states)
        return PatchTSMixerEncoderOutputWithNoAttention(last_hidden_state=last_hidden_state,
                                            hidden_states=hidden_states)



@add_start_docstrings(
    "The PatchTSMixer Model for time-series forecasting.",
    PATCHTSMIXER_START_DOCSTRING,
)

class PatchTSMixerModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.
    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        patched_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_len)`): 
            Patched input data
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*): 
            Bool Tensor indicating True in masked places and False otherwise.
        revin_mean: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*): 
            Bool Tensor indicating True in masked places and False otherwise. Gives the mean of the context window
            per channel.
        revin_std_dev: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*): 
            Bool Tensor indicating True in masked places and False otherwise. Gives the std dev of the context window
            per channel.
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
    patched_input: torch.FloatTensor = None
    mask: Optional[torch.FloatTensor] = None
    revin_mean: Optional[torch.FloatTensor] = None,
    revin_stdev: Optional[torch.FloatTensor] = None,


class PatchTSMixerModel(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig, mask_input: bool = False):
        super().__init__(config)

        set_seed(config.seed_number)

        self.encoder = PatchTSMixerEncoder(config)
        self.patching = Patch(
            config.seq_len, patch_len=config.patch_len, stride=config.stride
        )

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

    def forward(
            self, 
            context_values: torch.Tensor, 
            output_hidden_states: Optional[bool] = False
        ) -> PatchTSMixerModelOutputWithNoAttention:
        """
        context_values: tensor [bs x seq_len x in_channels]
        """
        revin_mean = None
        revin_stdev = None
        mask = None
        
        if self.revin is not None:
            context_values = self.revin(context_values, mode = "norm") # x: tensor [bs x seq_len x in_channels]
            revin_mean = self.revin.mean
            revin_stdev = self.revin.stdev
        
        patched_x = self.patching(context_values)  # [bs x in_channels x num_patch x patch_len]
        
        enc_input = patched_x

        if self.masking is not None:
            
            enc_input, mask = self.masking(patched_x) 
            # enc_input: [bs x in_channels x num_patch x patch_len] 
            # mask: [bs x in_channels x num_patch]
        
        encoder_output = self.encoder(enc_input, output_hidden_states = output_hidden_states)


        return PatchTSMixerModelOutputWithNoAttention(
            last_hidden_state=encoder_output.last_hidden_state, 
            hidden_states = encoder_output.hidden_states,
            patched_input=patched_x,
            mask=mask,
            revin_mean=revin_mean,
            revin_stdev=revin_stdev,
        )


class PatchTSMixerPretrainHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.head = PretrainHead(
            num_patches=config.num_patches,
            num_features=config.num_features,
            in_channels=config.in_channels,
            patch_size=config.patch_len,
            head_dropout=config.head_dropout,
            mode=config.mode,  
        )

    def forward(self, context_values: torch.Tensor) -> torch.Tensor:
        """
        context_values: [bs x n_vars x num_patches x num_features]

        return: [bs x n_vars x num_patches x patch_len]
        """
        return self.head(context_values)


class PatchTSMixerForMaskPreTrainingOutputWithNoAttention(ModelOutput):
    """
    Output type of [`PatchTSMixerForMaskPreTrainingOutputWithNoAttention`].

    Args: 
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, patch_len)`):
            Prediction output from the pretrain head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForMaskPretraining(PatchTSMixerPreTrainedModel):
    # PatchTSTModel + Pretraining Head
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.model = PatchTSMixerModel(config, mask_input=True)
        self.head = PatchTSMixerPretrainHead(config)
        self.masked_loss = config.masked_loss
        if config.masked_loss is True:
            self.loss = torch.nn.MSELoss(reduction="none")
        else:
            self.loss = torch.nn.MSELoss(reduction="mean")
        
        if config.revin == True:
            self.revin = RevIN()
        else:
            self.revin = None
        
        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    def forward(
            self, 
            context_values: torch.Tensor,
            output_hidden_states: Optional[bool] = False,
            return_loss: bool = True
        ) -> PatchTSMixerForMaskPreTrainingOutputWithNoAttention:
        """
        context_values: tensor [bs x seq_len x in_channels]
        """
        model_output = self.model(context_values, output_hidden_states = output_hidden_states) # x.last_hidden_state: [bs x nvars x num_patch x num_features]
        x_hat = self.head(
            model_output.last_hidden_state
        )  # tensor [bs x nvars x num_patch x patch_len]

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
            hidden_states = model_output.hidden_states,
            loss=loss_val,
        )


class PatchTSMixerForecastHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.head = ForecastHead(
            num_patches=config.num_patches,
            in_channels=config.in_channels,
            patch_size=config.patch_len,
            num_features=config.num_features,
            forecast_len=config.forecast_len,
            head_dropout=config.head_dropout,
            mode=config.mode,
            forecast_channel_indices=config.forecast_channel_indices,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        """
        x: [bs x in_channels x num_patches x num_features]
        return: [bs x forecast_len x in_channels]
        """
        return self.head(x, y = y)


class PatchTSMixerForForecastOutputWithNoAttention(ModelOutput):
    """
    Output type of [`PatchTSMixerForForecastOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, forecast_len, in_channels)`):
            Prediction output from the forecast head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForForecasting(PatchTSMixerPreTrainedModel):
    # PatchTSMixModel + Forecasting Head
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerForecastHead(config)
        self.loss = torch.nn.MSELoss(reduction="mean")
        if config.revin is True:
            self.revin = RevIN(denorm_channels = config.forecast_channel_indices)
        else:
            self.revin = None
        
        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    def forward(
        self,
        context_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True
    ) -> PatchTSMixerForForecastOutputWithNoAttention:
        """
        context_values: tensor [bs x seq_len x in_channels]
        """

        model_output = self.model(
            context_values,
            output_hidden_states = output_hidden_states,
        )  # model_output: [bs x nvars x num_patch x num_features]


        y_hat = self.head(
            model_output.last_hidden_state,
        )  # tensor [bs x forecast_len x in_channels]

        if self.revin is not None:
            self.revin.set_statistics(mean = model_output.revin_mean, stdev = model_output.revin_stdev)
            y_hat = self.revin(y_hat, mode = "denorm")


        if target_values is not None and return_loss is True:
            if self.config.forecast_channel_indices is not None:
                loss_val = self.loss(y_hat, target_values[..., self.config.forecast_channel_indices])
            else:    
                loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForForecastOutputWithNoAttention(
            prediction_logits=y_hat,  # tensor [bs x forecast_len x in_channels]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states = model_output.hidden_states,
            loss=loss_val,
        )


class PatchTSMixerClassificationHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.head = LinearHead(
            num_patches=config.num_patches,
            in_channels=config.in_channels,
            num_features=config.num_features,
            head_dropout=config.head_dropout,
            output_dim=config.n_classes,
            output_range=config.output_range,
            head_agg=config.head_agg,
            mode=config.mode,
        )

    def forward(self, context_values, target_values=None) -> torch.Tensor:
        """
        x: [bs x nvars x num_patch x num_features]
        output: [bs x n_classes]
        """
        return self.head(context_values, y=target_values)


class PatchTSMixerForClassificationOutputWithNoAttention(ModelOutput):
    """
    Output type of [`PatchTSMixerForClassificationOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, forecast_len, in_channels)`):
            Prediction output from the forecast head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForClassification(PatchTSMixerPreTrainedModel):
    # PatchTSMixer model + classification head
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerClassificationHead(config)
        self.loss = torch.nn.CrossEntropyLoss()

        if config.revin is True:
            if config.mode == "flatten":
                raise Exception("revin is not enabled for classification task when mode == flatten")
            self.inject_revin = InjectRevinStatistics4D(num_features = config.num_features,
                                num_patches = config.num_patches)
                            
                        
        else:
            self.inject_revin = None
        
        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()


    def forward(
            self, 
            context_values: torch.Tensor, 
            target_values: torch.Tensor = None,
            output_hidden_states: Optional[bool] = False,
            return_loss: bool = True
    ) -> PatchTSMixerForClassificationOutputWithNoAttention:

        """
        context_values: tensor [bs x seq_len x in_channels]
        target_values: tensor [bs x n_classes]
        """

        model_output = self.model(
            context_values,
            output_hidden_states = output_hidden_states,
        )  # x: [bs x nvars x num_patch x num_features]

        if self.inject_revin is not None:
            revin_statistics = (model_output.revin_mean, model_output.revin_stdev) # revin_mean,revin_stddev: [bs x 1 x n_channels]
            model_output.last_hidden_state = self.inject_revin(model_output.last_hidden_state, revin_statistics) # x: [bs x nvars x num_patch x num_features]


        y_hat = self.head(model_output.last_hidden_state)  # tensor [bs x n_labels]

        if target_values is not None and return_loss is True:
            loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForClassificationOutputWithNoAttention(
            prediction_logits=y_hat,  # tensor [bs x n_labels]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states = model_output.hidden_states,
            loss=loss_val,
        )


class PatchTSMixerRegressionHead(nn.Module):
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        
        self.head = LinearHead(
            num_patches=config.num_patches,
            in_channels=config.in_channels,
            num_features=config.num_features,
            head_dropout=config.head_dropout,
            output_dim=config.n_targets,
            output_range=config.output_range,
            head_agg=config.head_agg,
            mode=config.mode,
        )

    def forward(self, context_values: torch.Tensor, target_values: torch.Tensor=None ) -> torch.Tensor:
        """
        context_values: [bs x in_channels x num_patches x num_features]
        return: [bs x n_targets]
        """
        return self.head(context_values, y = target_values)


class PatchTSMixerForRegressionOutputWithNoAttention(ModelOutput):
    """
    Output type of [`PatchTSMixerForRegressionOutputWithNoAttention`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, forecast_len, in_channels)`):
            Prediction output from the forecast head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, num_features)`):
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
    # PatchTSMixerModel + Regression Head
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerRegressionHead(config)
        self.loss = torch.nn.MSELoss(reduction="mean")

        if config.revin == True:
            if config.mode == "flatten":
                raise Exception("revin is not enabled for regression task when mode == flatten")
            self.inject_revin = InjectRevinStatistics4D(num_features = config.num_features,
                                num_patches = config.num_patches)
        else:
            self.inject_revin = None
        
        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    def forward(
        self,
        context_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForRegressionOutputWithNoAttention:
        """
        context_values: tensor [bs x seq_len x in_channels]
        target_values: tensor [bs x n_targets]
        """

        model_output = self.model(
            context_values,
            output_hidden_states = output_hidden_states,
        )  # model_output: [bs x nvars x num_patch x num_features]


        if self.inject_revin is not None:
            revin_statistics = (model_output.revin_mean, model_output.revin_stdev) # revin_mean,revin_stddev: [bs x 1 x n_channels]
            model_output.last_hidden_state = self.inject_revin(model_output.last_hidden_state, revin_statistics) # x: [bs x nvars x num_patch x num_features]
            
        y_hat = self.head(model_output.last_hidden_state)  # tensor [bs x n_targets]

        if target_values is not None and return_loss is True:
            loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForRegressionOutputWithNoAttention(
            prediction_logits=y_hat,  # tensor [bs x n_targets]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states = model_output.hidden_states,
            loss=loss_val,
        )

