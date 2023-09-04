# coding=utf-8
# Copyright (c) 2021 THUML @ Tsinghua University
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
                     PatchMasking, ForecastHead, PretrainHead, RevIN)

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
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Past values of the time series, that serve as context in order to predict the future. These values may
            contain lags, i.e. additional values from the past which are added in order to serve as "extra context".
            The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
            `static_categorical_features`, `static_real_features`, `past_time_features`).

            The sequence length here is equal to `context_length` + `max(config.lags_sequence)`.

            Missing values need to be replaced with zeros.

        past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`, *optional*):
            Optional time features, which the model internally will add to `past_values`. These could be things like
            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
            could also be so-called "age" features, which basically help the model know "at which point in life" a
            time-series is. Age features have small values for distant past time steps and increase monotonically the
            more we approach the current time step.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features.

            The PatchTSMixer only learns additional embeddings for `static_categorical_features`.

        past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in
            `[0, 1]`:

            - 1 for values that are **observed**,
            - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        static_categorical_features (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*):
            Optional static categorical features for which the model will learn an embedding, which it will add to the
            values of the time series.

            Static categorical features are features which have the same value for all time steps (static over time).

            A typical example of a static categorical feature is a time series ID.

        static_real_features (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*):
            Optional static real features which the model will add to the values of the time series.

            Static real features are features which have the same value for all time steps (static over time).

            A typical example of a static real feature is promotion information.

        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length)`):
            Future values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs to learn to output, given the `past_values`.

            See the demo notebook and code snippets for details.

            Missing values need to be replaced with zeros.

        future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`, *optional*):
            Optional time features, which the model internally will add to `future_values`. These could be things like
            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
            could also be so-called "age" features, which basically help the model know "at which point in life" a
            time-series is. Age features have small values for distant past time steps and increase monotonically the
            more we approach the current time step.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional features.

            The PatchTSMixer only learns additional embeddings for `static_categorical_features`.

        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on certain token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
            make sure the model can only look at previous inputs in order to predict the future.

        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def set_seed(x=42):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(x)


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
    Base class for PatchTSMixerEncoderOutput, with potential hidden states.
    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        patched_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_len)`): 
            Patched input data
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
        )
        

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, context_values: torch.Tensor, output_hidden_states: Optional[bool] = False) -> PatchTSMixerEncoderOutputWithNoAttention:
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
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        set_seed(config.seed_number)

        if hasattr(config, "mask_input"):
            mask_input = config.mask_input
        else:
            mask_input = False

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


class PatchTSMixerForPreTrainingOutputWithNoAttention(ModelOutput):
    """
    Output type of [`PatchTSMixerForPreTrainingOutput`].

    Args:
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, patch_len)`):
            Prediction output from the pretrain head.

        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForPretraining(PatchTSMixerPreTrainedModel):
    # PatchTSTModel + Pretraining Head
    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        config.mask_input = True
        self.model = PatchTSMixerModel(config)
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
        self.post_init()

    def forward(
            self, 
            context_values: torch.Tensor,
            output_hidden_states: Optional[bool] = False,
            return_loss: bool = True
        ) -> PatchTSMixerForPreTrainingOutputWithNoAttention:
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
        
        return PatchTSMixerForPreTrainingOutputWithNoAttention(
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
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, forecast_len, in_channels)`):
            Prediction output from the forecast head.

        backbone_embeddings (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
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
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, n_classes)`):
            Prediction output from the classification head.

        backbone_embeddings (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
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
            self.inject_revin = InjectRevinStatistics4D(num_features = config.num_features,
                                num_patches = config.num_patches)
                        
        else:
            self.inject_revin = None
        
        # Initialize weights and apply final processing
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
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, n_targets)`):
            Prediction output from the regression head.

        backbone_embeddings (`torch.FloatTensor` of shape `(batch_size, in_channels, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
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
            self.inject_revin = InjectRevinStatistics4D(num_features = config.num_features,
                                num_patches = config.num_patches)
        else:
            self.inject_revin = None
        
        # Initialize weights and apply final processing
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

