# coding=utf-8
# Copyright 2023 Amazon and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch PatchTST model."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import math
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...modeling_outputs import BaseModelOutputWithNoAttention
from .configuration_patchtst import PatchTSTConfig
from torch.nn.modules.activation import MultiheadAttention
from ...utils import ModelOutput

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSTConfig"


PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtst-base",
    # See all PatchTST models at https://huggingface.co/models?filter=patchtst
]


class PatchTSTAttention(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.self_attn = MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            bias=config.bias,
            add_bias_kv=True,
            add_zero_attn=False,
            batch_first=True,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: Tensor [bs x q_len x d_model]
        """
        src, _ = self.self_attn(src, src, src, need_weights=False)
        return src


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        w_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        w_pos = torch.empty((q_len, 1))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "zeros":
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "normal" or pe == "gauss":
        w_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        w_pos = coord1d_pos_encoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        w_pos = coord1d_pos_encoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        w_pos = coord2d_pos_encoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == "exp2d":
        w_pos = coord2d_pos_encoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == "sincos":
        pos_enc = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc - pos_enc.mean()
        pos_enc = pos_enc / (pos_enc.std() * 10)
        w_pos = pos_enc
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
        )
    return nn.Parameter(w_pos, requires_grad=learn_pe)


def coord2d_pos_encoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = (
            2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x)
            - 1
        )
        # pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def coord1d_pos_encoding(q_len, exponential=False, normalize=True):
    cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1)) - 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


class TSTEncoderLayer(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.pre_norm = config.pre_norm

        assert (
            not config.d_model % config.encoder_attention_heads
        ), f"d_model ({config.d_model}) must be divisible by n_heads ({config.encoder_attention_heads})"

        # Multi-Head attention
        self.self_attn = PatchTSTAttention(config)

        # Add & Norm of the sublayer 1
        self.dropout_path1 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
        if "batch" in config.norm.lower():
            self.norm_sublayer1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
        else:
            self.norm_sublayer1 = nn.LayerNorm(config.d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.encoder_ffn_dim, bias=config.bias),
            get_activation_fn(config.activation_function),
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
            nn.Linear(config.encoder_ffn_dim, config.d_model, bias=config.bias),
        )

        # Add & Norm of sublayer 2
        self.dropout_path2 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
        if "batch" in config.norm.lower():
            self.norm_sublayer2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
        else:
            self.norm_sublayer2 = nn.LayerNorm(config.d_model)

    def forward(self, src: torch.Tensor):
        """
        src: tensor [bs x seq_len x d_model]
        Return:
            Tensor [bs x seq_len x d_model]
        """
        # First sublayer: mixing across time
        if self.pre_norm:
            ## Norm and Multi-Head attention and Add residual connection
            src = src + self.dropout_path1(
                self.self_attn(self.norm_sublayer1(src))
            )  # Add: residual connection with residual dropout
        else:
            ## Multi-Head attention and Add residual connection and Norm - Standard Transformer from BERT
            src = self.norm_sublayer1(src + self.dropout_path1(self.self_attn(src)))

        # Second sublayer: mixing across hidden dimension
        if self.pre_norm:
            ## Norm and Position-wise Feed-Forward and Add residual connection
            src = src + self.dropout_path2(
                self.ff(self.norm_sublayer2(src))
            )  # Add: residual connection with residual dropout
        else:
            ## Position-wise Feed-Forward and Add residual connection and Norm - Standard Transformer from BERT
            src = self.norm_sublayer2(
                src + self.dropout_path2(self.ff(src))
            )  # Add: residual connection with residual dropout

        return src


class TSTEncoder(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(config) for i in range(config.encoder_layers)])

    def forward(
        self, src: torch.Tensor, output_hidden_states: Optional[bool] = False, output_attention: Optional[bool] = False
    ) -> torch.Tensor:
        """
        src: tensor [bs x seq_len x d_model]
        Return:
            Tensor [bs x seq_len x d_model]
        """
        all_hidden_states = []
        for mod in self.layers:
            if output_hidden_states:
                src = mod(src)
                all_hidden_states.append(src)
        if output_hidden_states:
            return src, all_hidden_states
        return src


class PatchTSTPreTrainedModel(PreTrainedModel):
    config_class = PatchTSTConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights"""
        if self.config.use_cls_token:
            torch.nn.init.normal_(self.config.cls_token, std=0.02)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PatchTSTEncoder)):
            module.gradient_checkpointing = value


class PatchTSTEncoder(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        # self.n_vars = c_in
        self.num_patch = (max(config.context_length, config.patch_length) - config.patch_length) // config.stride + 1
        self.d_model = config.d_model
        self.shared_embedding = config.shared_embedding
        self.use_cls_token = config.use_cls_token

        # Added params for patching
        self.patch_last = config.patch_last
        self.mask_ratio = config.mask_ratio

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not self.shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(config.input_size):
                self.W_P.append(nn.Linear(config.patch_length, self.d_model))
        else:
            self.W_P = nn.Linear(config.patch_length, config.d_model)

        # Positional encoding
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            self.W_pos = positional_encoding(
                config.positional_encoding, config.learn_pe, self.num_patch + 1, config.d_model
            )
        else:
            self.W_pos = positional_encoding(
                config.positional_encoding, config.learn_pe, self.num_patch, config.d_model
            )

        # Positional dropout
        self.dropout = nn.Dropout(config.positional_dropout) if config.positional_dropout > 0 else nn.Identity()

        # Encoder
        self.encoder = TSTEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor [bs x nvars x num_patch x patch_len]    #[bs x num_patch x nvars x patch_len]
        return:
            tensor [bs x nvars x num_patch x d_model]
                or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        """

        # bs, num_patch, n_vars, patch_len = x.shape
        bs, n_vars, num_patch, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, i, :, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.W_P(x)  # x: [bs x nvars  x num_patch x d_model]

        # x: [bs x nvars x num_patch x d_model] -> [bs * nvars x num_patch x d_model]
        x = x.view(bs * n_vars, num_patch, self.d_model)  # x: [bs * nvars x num_patch x d_model]

        if self.use_cls_token:
            # print(f'x and W_pos shapes: {x.shape}, {self.W_pos.shape}')
            x = self.dropout(x + self.W_pos[1:, :])  # x: [bs * nvars x num_patch x d_model]
            # append cls token
            cls_token = self.cls_token + self.W_pos[:1, :]  # cls_token: [1 x 1 x d_model]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # get the same copy for all the batch samples
            x = torch.cat((cls_tokens, x), dim=1)  # x: [bs * nvars x (num_patch+1) x d_model]
        else:
            # print(f'x and W_pos shapes: {x.shape}, {self.W_pos.shape}')
            x = self.dropout(x + self.W_pos)  # x: [bs * nvars x num_patch x d_model]

        # Encoder
        x = self.encoder(
            x
        )  # x: [bs * nvars x num_patch x d_model] or [bs * nvars x (num_patch+1) x d_model] if use cls_token
        x = torch.reshape(
            x, (bs, n_vars, -1, self.d_model)
        )  # x: [bs x nvars x num_patch x d_model] or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        return x


PATCHTST_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PATCHTST_INPUTS_DOCSTRING = r"""
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`):
            Past values of the time series, that serve as context in order to predict the future. The sequence size of
            this tensor must be larger than the `context_length` of the model, since the model will use the larger size
            to construct lag features, i.e. additional values from the past which are added in order to serve as "extra
            context".

            The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if no
            `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
            look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length of
            the past.

            The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
            `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

            Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

            For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
            variates in the time series per time step.
        past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`):
            Required time features, which the model internally will add to `past_values`. These could be things like
            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
            could also be so-called "age" features, which basically help the model know "at which point in life" a
            time-series is. Age features have small values for distant past time steps and increase monotonically the
            more we approach the current time step. Holiday features are also a good example of time features.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features. The Time Series Transformer only learns
            additional embeddings for `static_categorical_features`.

            Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
            must but known at prediction time.

            The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
        past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):
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
        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, input_size)`, *optional*):
            Future values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`.

            The sequence length here is equal to `prediction_length`.

            See the demo notebook and code snippets for details.

            Optionally, during training any missing values need to be replaced with zeros and indicated via the
            `future_observed_mask`.

            For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
            variates in the time series per time step.
        future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`):
            Required time features for the prediction window, which the model internally will add to `future_values`.
            These could be things like "month of year", "day of the month", etc. encoded as vectors (for instance as
            Fourier features). These could also be so-called "age" features, which basically help the model know "at
            which point in life" a time-series is. Age features have small values for distant past time steps and
            increase monotonically the more we approach the current time step. Holiday features are also a good example
            of time features.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features. The Time Series Transformer only learns
            additional embeddings for `static_categorical_features`.

            Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
            must but known at prediction time.

            The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
        future_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):
            Boolean mask to indicate which `future_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:

            - 1 for values that are **observed**,
            - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

            This mask is used to filter out missing values for the final loss calculation.
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


class PatchTSTEncoder(PatchTSTPreTrainedModel):
    """
    PatchTST encoder consisting of *config.encoder_layers* self attention layers with distillation layers. Each
    attention layer is an [`PatchTSTEncoderLayer`].

    Args:
        config: PatchTSTConfig
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        # self.n_vars = c_in
        self.num_patch = (max(config.context_length, config.patch_length) - config.patch_length) // config.stride + 1
        self.d_model = config.d_model
        self.shared_embedding = config.shared_embedding
        self.use_cls_token = config.use_cls_token

        # Added params for patching
        self.patch_last = config.patch_last
        self.mask_ratio = config.mask_ratio

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not self.shared_embedding:
            self.w_p = nn.ModuleList()
            for _ in range(config.input_size):
                self.w_p.append(nn.Linear(config.patch_length, self.d_model))
        else:
            self.w_p = nn.Linear(config.patch_length, config.d_model)

        # Positional encoding
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            self.w_pos = positional_encoding(
                config.positional_encoding, config.learn_pe, self.num_patch + 1, config.d_model
            )
        else:
            self.w_pos = positional_encoding(
                config.positional_encoding, config.learn_pe, self.num_patch, config.d_model
            )

        # Positional dropout
        self.dropout = nn.Dropout(config.positional_dropout) if config.positional_dropout > 0 else nn.Identity()

        # Encoder
        self.encoder = TSTEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor [bs x nvars x num_patch x patch_len]    #[bs x num_patch x nvars x patch_len]
        return:
            tensor [bs x nvars x num_patch x d_model]
                or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        """

        # bs, num_patch, n_vars, patch_len = x.shape
        bs, n_vars, num_patch, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.w_p[i](x[:, i, :, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.w_p(x)  # x: [bs x nvars  x num_patch x d_model]

        # x: [bs x nvars x num_patch x d_model] -> [bs * nvars x num_patch x d_model]
        x = x.view(bs * n_vars, num_patch, self.d_model)  # x: [bs * nvars x num_patch x d_model]

        if self.use_cls_token:
            # print(f'x and W_pos shapes: {x.shape}, {self.W_pos.shape}')
            x = self.dropout(x + self.w_pos[1:, :])  # x: [bs * nvars x num_patch x d_model]
            # append cls token
            cls_token = self.cls_token + self.w_pos[:1, :]  # cls_token: [1 x 1 x d_model]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # get the same copy for all the batch samples
            x = torch.cat((cls_tokens, x), dim=1)  # x: [bs * nvars x (num_patch+1) x d_model]
        else:
            # print(f'x and W_pos shapes: {x.shape}, {self.W_pos.shape}')
            x = self.dropout(x + self.w_pos)  # x: [bs * nvars x num_patch x d_model]

        # Encoder
        x = self.encoder(
            x
        )  # x: [bs * nvars x num_patch x d_model] or [bs * nvars x (num_patch+1) x d_model] if use cls_token
        x = torch.reshape(
            x, (bs, n_vars, -1, self.d_model)
        )  # x: [bs x nvars x num_patch x d_model] or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        return x


@add_start_docstrings(
    "The bare PatchTST Model outputting raw hidden-states without any specific head on top.",
    PATCHTST_START_DOCSTRING,
)
# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesTransformerModel with TimeSeriesTransformer->PatchTST,TIME_SERIES_TRANSFORMER->PATCHTST,time-series-transformer->patchtst,TimeSeries->PatchTST
class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.encoder = PatchTSTEncoder(config)

    def forward(self, x: torch.Tensor):
        encoder_output = self.encoder(x)
        return BaseModelOutputWithNoAttention(last_hidden_state=encoder_output, hidden_states=None)


class PretrainHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.d_model, config.patch_length)
        self.use_cls_token = config.use_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor [bs x nvars x num_patch x d_model]
                or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]
        if self.use_cls_token:
            x = x[:, :, 1:, :]  # remove the first cls token
        return x


def cv_random_masking(
    xb: torch.Tensor,
    mask_ratio: float,
    cv_channel_indices: list = None,
    channel_consistent_masking: bool = True,
    d_size="4D",
    mask_value=0,
):
    """cv_random_masking: Mask the input considering the control variables.

    Args:
        xb (Tensor): Input to mask [ bs x nvars x num_patch x patch_len] or [ bs x tsg1 x tag2 x nvars x num_patch x patch_len]
        mask_ratio (float): Mask ratio.
        cv_channel_indices (list, optional): Control Variable channel indices. These channels will not be masked. Defaults to None.
        channel_consistent_masking (bool, optional): When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary across channels. Defaults to True.
        d_size (str, optional): Input data size. Allowed values: 4D, 6D. Defaults to "4D".
        mask_value (int, optional): Value to use for masking. Defaults to 0.

    Returns:
        Tensor: xb_mask, masked input, same shape as input
        Tensor: Mask tensor of shape [bs x c x n] or [bs x tsg1 x tsg2 x c x n]
    """
    if d_size == "4D":
        bs, nvars, L, D = xb.shape

    len_keep = int(L * (1 - mask_ratio))

    if d_size == "4D":
        if channel_consistent_masking:
            noise = torch.rand(bs, 1, L, device=xb.device)  # noise in [0, 1], bs x 1 x  L
            noise = noise.repeat(1, nvars, 1)  # bs x nvars x L
        else:
            noise = torch.rand(bs, nvars, L, device=xb.device)  # noise in [0, 1], bs x nvars x L

        mask = torch.ones(bs, nvars, L, device=xb.device)  # mask: [bs x nvars x num_patch]
        mask[:, :, :len_keep] = 0

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x nvars x L]
    mask = torch.gather(mask, dim=-1, index=ids_restore)

    if d_size == "4D":
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, D)  # mask: [bs x nvars x num_patch x patch_len]
        if cv_channel_indices is not None:
            mask[:, cv_channel_indices, :, :] = 0

    xb_mask = xb.masked_fill(mask.bool(), mask_value)
    return xb_mask, mask[..., 0]


class PatchMasking(nn.Module):
    def __init__(
        self,
        mask_type: str = "random",
        mask_ratio=0.5,
        mask_patches: list = [2, 3],
        mask_patch_ratios: list = [1, 1],
        channel_consistent_masking: bool = True,
        d_size: str = "4D",
        cv_channel_indices: list = None,
        mask_value=0,
    ):
        """PatchMasking: Class to random or forcast masking.

        Args:
            mask_type (str, optional): Masking type. Allowed values are random, forecast. Defaults to random.
            mask_ratio (float, optional): Mask ratio.
            mask_patches (list, optional): List of patch lengths to mask in the end of the data.
            mask_patch_ratios (list, optional): List of weights to use for each patch length. For Ex.
            if patch_lengths is [5,4] and mix_ratio is [1,1], then equal weights to both patch lengths. Defaults to None.
            cv_channel_indices (list, optional): Control Variable channel indices. These channels will not be masked. Defaults to None.
            channel_consistent_masking (bool, optional): When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary across channels. Defaults to True.
            d_size (str, optional): Input data size. Allowed values: 4D, 6D. Defaults to "4D".
            mask_value (int, optional): Value to use for masking. Defaults to 0.
        """

        self.mask_ratio = mask_ratio
        self.channel_consistent_masking = channel_consistent_masking
        self.d_size = d_size
        self.mask_type = mask_type
        self.mask_patches = mask_patches
        self.mask_patch_ratios = mask_patch_ratios
        self.cv_channel_indices = cv_channel_indices
        self.mask_value = mask_value
        if self.cv_channel_indices is not None:
            self.cv_channel_indices.sort()

        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Input:
            x: patched input
                4D: [bs x n_vars x num_patch  x patch_len]

        Output:
            x_mask: Masked patched input
                4D: [bs x n_vars x num_patch  x patch_len]
            mask: bool tensor indicating True on masked points
                4D: [bs x n_vars x num_patch]
        """

        if self.mask_type == "random":
            x_mask, mask = cv_random_masking(
                xb=x,
                mask_ratio=self.mask_ratio,
                cv_channel_indices=self.cv_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                d_size=self.d_size,
                mask_value=self.mask_value,
            )

        else:
            raise Exception("Invalid mask type")

        mask = mask.bool()  # mask: [bs x n_vars x num_patch]

        return x_mask, mask


class Patch(nn.Module):
    """
    A class to patchify the time series sequence into different patches
    """

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        padding: bool = False,  # TODO: use this to set whether we want to pad zeros to the sequence
    ):
        super().__init__()

        assert (
            seq_len > patch_len
        ), f"Sequence length ({seq_len}) has to be greater than the patch length ({patch_len})"

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride

        # get the number of patches
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        tgt_len = patch_len + stride * (self.num_patch - 1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor, required): Input of shape [bs x ... x seq_len x n_vars]
        Returns:
            z: output tensor data [bs x ... x n_vars x num_patch x patch_len]
        """
        seq_len = x.shape[-2]
        assert seq_len == self.seq_len, f"Input sequence length ({seq_len}) doesn't match model ({self.seq_len})."

        # x = x[:, :, self.s_begin:, :]  # xb: [bs x ... x tgt_len x nvars]
        z = x.transpose(0, -2)[self.s_begin :]  # z: [tgt_len x ... x bs x n_vars]
        z = z.transpose(0, -2).contiguous()  # z: [bs x ... x tgt_len x n_vars]  # TODO: need a better solution
        z = z.unfold(
            dimension=-2, size=self.patch_len, step=self.stride
        )  # xb: [bs x ... x num_patch x n_vars x patch_len]
        z = z.transpose(-2, -3).contiguous()  # xb: [bs x ... x n_vars x num_patch x patch_len]
        return z


class PatchTSTForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSTForPretraining(PatchTSTPreTrainedModel):
    # PatchTSTModel + Pretraining Head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.patching = Patch(config.context_length, patch_len=config.patch_length, stride=config.stride)
        self.masking = PatchMasking(
            mask_type=config.mask_type,
            mask_ratio=config.mask_ratio,
            mask_patches=config.mask_patches,
            mask_patch_ratios=config.mask_patch_ratios,
            channel_consistent_masking=config.channel_consistent_masking,
            d_size=config.d_size,
            cv_channel_indices=config.cv_channel_indices,
            mask_value=config.mask_value,
        )
        self.model = PatchTSTModel(config)
        self.head = PretrainHead(config)
        self.loss = torch.nn.MSELoss(reduction="mean")

    def forward(
        self, past_values: torch.Tensor, future_values: Optional[torch.Tensor] = None
    ) -> PatchTSTForPreTrainingOutput:
        """
        past_values (x): tensor [bs x seq_len x n_vars ]
        future_values (y): labels
        """
        patched_x = self.patching(past_values) # patched_x: [bs x n_vars x num_patch x patch_len] for pretrain
        masked_x, masked = self.masking(patched_x)
        model_output = self.model(masked_x)  # x: [bs x nvars x num_patch x d_model]
        #  or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        x_hat = self.head(model_output[0])  # tensor [bs x nvars x num_patch x patch_len]

        loss_val = self.loss(x_hat, patched_x)
        return PatchTSTForPreTrainingOutput(
            loss=loss_val,
            prediction_logits=x_hat,
        )


class PatchTSTForClassification(PatchTSTPreTrainedModel):
    # PatchTST model + classification head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.patching = Patch(config.context_length, patch_len=config.patch_length, stride=config.stride)

        self.model = PatchTSTModel(config)
        self.head = ClassificationHead(config)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, past_values, future_values=None):
        patched_x = self.patching(past_values)
        model_output = self.model(patched_x)
        y_hat = self.head(model_output[0])

        loss_val = None
        if future_values is not None:
            loss_val = self.loss(y_hat, future_values)
        return PatchTSTForClassificationOutput(
            loss=loss_val,
            prediction_logits=y_hat,
        )


class ClassificationHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        self.linear = nn.Linear(config.input_size * config.d_model, config.num_classes)

    def forward(self, x):
        """
        x: [bs x nvars x num_patch x d_model] or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        output: [bs x n_classes]
        """
        if self.use_cls_token:
            x = x[:, :, 0, :]  # use the first output token, x: bs x nvars x d_model
        elif self.pooling == "mean":
            x = x.mean(dim=2)  # x: [bs x nvars x d_model]
        elif self.pooling == "max":
            x = x.max(dim=2)  # x: [bs x nvars x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")

        x = self.flatten(x)  # x: bs x nvars * d_model
        y = self.linear(self.dropout(x))  # y: bs x n_classes
        return y


class PatchTSTForClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSTForClassification`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PredictionHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.individual = config.individual
        self.n_vars = config.input_size
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        head_dimension = config.d_model if config.pooling else config.d_model * config.num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=2))
                self.linears.append(nn.Linear(head_dimension, config.prediction_length))
                self.dropouts.append(nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
                                     )
        else:
            self.flatten = nn.Flatten(start_dim=2)
            self.linear = nn.Linear(head_dimension, config.prediction_length)
            self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        x: [bs x nvars x num_patch x d_model]
            or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        output: [bs x forecast_len x nvars]
        """

        if self.use_cls_token:
            y = x[:, :, 0, :]      # y: [bs x nvars x d_model]
        else:
            if self.pooling == 'mean':
                y = x.mean(dim=2)  # y: [bs x nvars x d_model]
            elif self.pooling == 'max':
                y = x.max(dim=2)  # y: [bs x nvars x d_model]
            else:
                y = x       # y: [bs x nvars x num_patch x d_model]

        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](y[:, i, :])  # y: [bs x (d_model * num_patch)] or [bs x d_model)]
                z = self.linears[i](z)  # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x forecast_len]
        else:
            z = self.flatten(y)         # z: [bs x nvars x (d_model * num_patch)] or [bs x nvars x d_model)]
            z = self.dropout(z)
            x = self.linear(z)  # x: [bs x nvars x forecast_len]

        x = x.transpose(2, 1)  # [bs x forecast_len x nvars]

        return x


class PatchTSTForPredictionOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPredictiontion`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    # PatchTST model + classification head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.patching = Patch(config.context_length, patch_len=config.patch_length, stride=config.stride)

        self.model = PatchTSTModel(config)
        self.head = PredictionHead(config)
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, past_values: torch.Tensor, future_values: Optional[torch.Tensor]):
        patched_x = self.patching(past_values)
        model_output = self.model(patched_x)
        y_hat = self.head(model_output[0])

        loss_val = None
        if future_values is not None:
            loss_val = self.loss(y_hat, future_values)
        return PatchTSTForPredictionOutput(
            loss=loss_val,
            prediction_logits=y_hat,
        )

