# coding=utf-8
# Copyright 2024 Google LLC and HuggingFace Inc. team.
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
"""PyTorch TimesFM model."""


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################


import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_timesfm import TimesFMConfig
from .timesfm_layers import *


@dataclass
class TimesFMOutput(BaseModelOutput):
    mean_predictions: np.ndarray = None
    full_predictions: np.ndarray = None


class TimesFMPreTrainedModel(PreTrainedModel):
    """handles the loading for all models."""

    config_class = TimesFMConfig
    base_model_prefix = "timesfm"
    main_input_name = "inputs"

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, a=-0.1, b=0.1)

        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, RMSNorm):
            nn.init.zeros_(module.weight)

        elif isinstance(module, PositionalEmbedding):
            pass


class PatchedTimeSeriesDecoder(TimesFMPreTrainedModel):
    """Patched time-series decoder."""

    def __init__(self, config: TimesFMConfig):
        super().__init__(config)

        self.config = config
        self.input_ff_layer = ResidualBlock(
            input_dims=2 * config.patch_len,
            output_dims=config.model_dim,
            hidden_dims=config.model_dim,
        )
        self.freq_emb = nn.Embedding(
            num_embeddings=config.freq_size, embedding_dim=config.model_dim
        )
        self.horizon_ff_layer = ResidualBlock(
            input_dims=config.model_dim,
            output_dims=config.horizon_len * (1 + len(config.quantiles)),
            hidden_dims=config.model_dim,
        )
        self.stacked_transformer = StackedDecoder(
            hidden_size=self.config.model_dim,
            intermediate_size=self.config.model_dim,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            num_layers=self.config.num_layers,
            rms_norm_eps=self.config.rms_norm_eps,
        )
        if self.config.use_positional_embedding:
            self.position_emb = PositionalEmbedding(
                embedding_dims=self.config.model_dim,
            )

    def _forward_transform(
        self, inputs: torch.Tensor, patched_pads: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Input is of shape [B, N, P]."""
        mu, sigma = masked_mean_std(inputs, patched_pads)
        sigma = torch.where(
            sigma < self.config.tolerance,
            torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
            sigma,
        )

        # Normalize each patch
        outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
        outputs = torch.where(
            torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(
                self.config.pad_val, dtype=outputs.dtype, device=outputs.device
            ),
            outputs,
        )
        return outputs, (mu, sigma)

    def _reverse_transform(
        self, outputs: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Output is of shape [B, N, P, Q]."""
        mu, sigma = stats
        return outputs * sigma[:, None, None, None] + mu[:, None, None, None]

    def _preprocess_input(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor,
    ]:
        """Preprocess input for stacked transformer."""

        # Reshape into patches (using view for efficiency)
        bsize = input_ts.shape[0]
        patched_inputs = input_ts.view(bsize, -1, self.config.patch_len)
        patched_pads = input_padding.view(bsize, -1, self.config.patch_len)

        patched_inputs = torch.where(
            torch.abs(patched_pads - 1.0) < self.config.tolerance,
            torch.tensor(0.0, dtype=patched_inputs.dtype, device=patched_inputs.device),
            patched_inputs,
        )
        patched_pads = torch.where(
            torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
            patched_pads,
        )
        patched_inputs, stats = self._forward_transform(patched_inputs, patched_pads)

        # B x N x D
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        print(">>> PatchedDecoder patched_inputs", patched_inputs.shape)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        print(">>> PatchedDecoder concat_inputs", concat_inputs.shape)
        model_input = self.input_ff_layer(concat_inputs)

        # A patch should not be padded even if there is at least one zero.
        patched_padding = torch.min(patched_pads, dim=-1)[
            0
        ]  # Get the values from the min result
        if self.config.use_positional_embedding:
            pos_emb = self.position_emb(model_input.shape[1]).to(model_input.device)
            pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
            pos_emb = shift_padded_seq(patched_padding, pos_emb)
            model_input += pos_emb

        return model_input, patched_padding, stats, patched_inputs

    def _postprocess_output(
        self,
        model_output: torch.Tensor,
        num_outputs: int,
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Postprocess output of stacked transformer."""

        # B x N x (H.Q)
        output_ts = self.horizon_ff_layer(model_output)

        # Reshape using view
        b, n, _ = output_ts.shape
        output_ts = output_ts.view(b, n, self.config.horizon_len, num_outputs)

        return self._reverse_transform(output_ts, stats)

    def forward(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        print(">>> PatchedDecoder input_ts", input_ts.shape)
        num_outputs = len(self.config.quantiles) + 1
        model_input, patched_padding, stats, _ = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )
        f_emb = self.freq_emb(freq)  # B x 1 x D
        model_input += f_emb

        print(">>> PatchedDecoder model_input", model_input.shape)
        model_output, all_attentions, all_hidden_states = self.stacked_transformer(model_input, patched_padding, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            all_hidden_states = [model_input] + all_hidden_states

        output_ts = self._postprocess_output(model_output, num_outputs, stats)
        return output_ts, all_attentions, all_hidden_states

    def decode(
        self,
        input_ts: torch.Tensor,
        paddings: torch.Tensor,
        freq: torch.LongTensor,
        horizon_len: int,
        output_patch_len: int | None = None,
        max_len: int = 512,
        return_forecast_on_context: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Auto-regressive decoding without caching.

        Args:
          input_ts: input time-series and paddings. Time-series shape B x C.
          paddings: padding shape B x (C + H) where H is the prediction length.
          freq: frequency shape B x 1
          horizon_len: prediction length.
          output_patch_len: output length to be fetched from one step of
            auto-regressive decoding.
          max_len: maximum training context length.
          return_forecast_on_context: whether to return the model forecast on the
            context except the first input patch.

        Returns:
          Tuple of two forecasting results:
          - Point (mean) output predictions as a tensor with shape B x H'.
          - Full predictions (mean and quantiles) as a tensor with shape
            B x H' x (1 + # quantiles).
          In particular, if return_forecast_on_context is True, H' is H plus
          the forecastable context length, i.e. context_len - (first) patch_len.
        """
        final_out = input_ts
        context_len = final_out.shape[1]
        full_outputs = []
        if paddings.shape[1] != final_out.shape[1] + horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}"
            )
        if output_patch_len is None:
            output_patch_len = self.config.horizon_len
        num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len
        for step_index in range(num_decode_patches):
            current_padding = paddings[:, 0 : final_out.shape[1]]
            input_ts = final_out[:, -max_len:]
            input_padding = current_padding[:, -max_len:]
            fprop_outputs, all_attentions, all_hidden_states = self.forward(input_ts, input_padding, freq, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            if return_forecast_on_context and step_index == 0:
                # For the first decodings step, collect the model forecast on the
                # context except the unavailable first input batch forecast.
                new_full_ts = fprop_outputs[:, :-1, : self.config.patch_len, :]
                new_full_ts = fprop_outputs.view(
                    new_full_ts.size(0), -1, new_full_ts.size(3)
                )

                full_outputs.append(new_full_ts)

            # (full batch, last patch, output_patch_len, index of mean forecast = 0)
            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
            # (full batch, last patch, output_patch_len, all output indices)
            full_outputs.append(new_full_ts)
            final_out = torch.concatenate([final_out, new_ts], axis=-1)

        if return_forecast_on_context:
            # `full_outputs` indexing starts at after the first input patch.
            full_outputs = torch.concatenate(full_outputs, axis=1)[
                :, : (context_len - self.config.patch_len + horizon_len), :
            ]
        else:
            # `full_outputs` indexing starts at the forecast horizon.
            full_outputs = torch.concatenate(full_outputs, axis=1)[:, 0:horizon_len, :]

        return full_outputs[:, :, 0], full_outputs, all_attentions, all_hidden_states


class TimesFMModel(TimesFMPreTrainedModel):
    def __init__(self, config: TimesFMConfig):
        super().__init__(config)

        self.config = config

        self.decoder = PatchedTimeSeriesDecoder(config)

        self.context_len = config.context_len
        self.horizon_len = config.horizon_len
        self.input_patch_len = config.patch_len
        self.output_patch_len = config.horizon_len
        self.num_layers = config.num_layers
        self.model_dims = config.model_dim
        self.quantiles = config.quantiles
        self.num_heads = config.num_heads

        self.num_cores = 1
        self.per_core_batch_size = config.per_core_batch_size
        self.global_batch_size = config.per_core_batch_size * self.num_cores
        self._horizon_start = self.context_len - self.input_patch_len

    def _preprocess(
        self, inputs: Sequence[np.array], freq: Sequence[int]
    ) -> tuple[np.array, np.array, int]:
        """Formats and pads raw inputs to feed into the model.

        This function both pads each time series to match the context length, and
        pads the inputs to meet the SPMD shape requirement.

        Args:
          inputs: A list of 1d JTensors. Each JTensor is the context time series of
            a single forecast task.
          freq: list of frequencies

        Returns:
        A tuple of:
        - the padded input time series to meet the model required context.
        - the padding indicator.
        - the number of padded examples for SPMD so that each core has the same
            number (a multiple of `batch_size`) of examples.
        """
        print(">>> TimesFMModel _preprocess", len(inputs), inputs[0].shape)
        input_ts, input_padding, inp_freq = [], [], []

        pmap_pad = (
            (len(inputs) - 1) // self.global_batch_size + 1
        ) * self.global_batch_size - len(inputs)

        for i, ts in enumerate(inputs):
            input_len = ts.shape[0]
            padding = np.zeros(shape=(input_len + self.horizon_len,), dtype=float)
            if input_len < self.context_len:
                num_front_pad = self.context_len - input_len
                ts = np.concatenate(
                    [np.zeros(shape=(num_front_pad,), dtype=float), ts], axis=0
                )
                padding = np.concatenate(
                    [np.ones(shape=(num_front_pad,), dtype=float), padding], axis=0
                )
            elif input_len > self.context_len:
                ts = ts[-self.context_len :]
                padding = padding[-(self.context_len + self.horizon_len) :]

            input_ts.append(ts)
            input_padding.append(padding)
            inp_freq.append(freq[i])

        # Padding the remainder batch.
        for _ in range(pmap_pad):
            input_ts.append(input_ts[-1])
            input_padding.append(input_padding[-1])
            inp_freq.append(inp_freq[-1])

        print(">>> TimesFMModel input_ts", len(input_ts), input_ts[0].shape)

        return (
            np.stack(input_ts, axis=0),
            np.stack(input_padding, axis=0),
            np.array(inp_freq).astype(np.int32).reshape(-1, 1),
            pmap_pad,
        )

    def forward(
        self,
        inputs: Sequence[Any],
        freq: Sequence[int] | None = None,
        window_size: int | None = None,
        forecast_context_len: int | None = None,
        return_forecast_on_context: bool = False,
        truncate_negative: bool = False,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecasts on a list of time series.

        Args:
          inputs: list of time series forecast contexts. Each context time series
            should be in a format convertible to JTensor by `jnp.array`.
          freq: frequency of each context time series. 0 for high frequency
            (default), 1 for medium, and 2 for low. Notice this is different from
            the `freq` required by `forecast_on_df`.
          window_size: window size of trend + residual decomposition. If None then
            we do not do decomposition.
          forecast_context_len: optional max context length.
          return_forecast_on_context: True to return the forecast on the context
            when available, i.e. after the first input patch.
          truncate_negative: truncate to only non-negative values if all the contexts
            have non-negative values.

        Returns:
        A tuple for JTensors:
        - the mean forecast of size (# inputs, # forecast horizon),
        - the full forecast (mean + quantiles) of size
            (# inputs,  # forecast horizon, 1 + # quantiles).

        Raises:
        ValueError: If the checkpoint is not properly loaded.
        """
        if return_dict is None:
            return_dict = self.config.use_return_dict

        if forecast_context_len is None:
            fcontext_len = self.context_len
        else:
            fcontext_len = forecast_context_len
        inputs = [np.array(ts)[-fcontext_len:] for ts in inputs]
        print(">>> TimesFMModel forward", len(inputs), inputs[0].shape)
        inp_min = np.min([np.min(ts) for ts in inputs])

        if window_size is not None:
            new_inputs = []
            for ts in inputs:
                new_inputs.extend(moving_average(ts, window_size))
            inputs = new_inputs

        if freq is None:
            logging.info("No frequency provided via `freq`. Default to high (0).")
            freq = [0] * len(inputs)

        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        input_ts, input_padding, inp_freq, pmap_pad = self._preprocess(inputs, freq)
        print(">>> TimesFMModel input_ts", input_ts.shape)
        with torch.no_grad():
            mean_outputs = []
            full_outputs = []
            all_attentions = []
            all_hidden_states = []
            assert input_ts.shape[0] % self.global_batch_size == 0
            for i in range(input_ts.shape[0] // self.global_batch_size):
                input_ts_in = torch.from_numpy(
                    np.array(
                        input_ts[
                            i
                            * self.global_batch_size : (i + 1)
                            * self.global_batch_size
                        ],
                        dtype=np.float32,
                    )
                )
                input_padding_in = torch.from_numpy(
                    np.array(
                        input_padding[
                            i
                            * self.global_batch_size : (i + 1)
                            * self.global_batch_size
                        ],
                        dtype=np.float32,
                    )
                )
                inp_freq_in = torch.from_numpy(
                    np.array(
                        inp_freq[
                            i
                            * self.global_batch_size : (i + 1)
                            * self.global_batch_size,
                            :,
                        ],
                        dtype=np.int32,
                    )
                ).long()
                mean_output, full_output, attentions, hidden_states = self.decoder.decode(
                    input_ts=input_ts_in,
                    paddings=input_padding_in,
                    freq=inp_freq_in,
                    horizon_len=self.horizon_len,
                    return_forecast_on_context=return_forecast_on_context,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                mean_output = mean_output.detach().cpu().numpy()
                full_output = full_output.detach().cpu().numpy()
                mean_output = np.array(mean_output)
                full_output = np.array(full_output)
                mean_outputs.append(mean_output)
                full_outputs.append(full_output)

                if output_attentions:
                    if not all_attentions:
                        all_attentions = [[] for _ in range(len(attentions))]
                    for j in range(len(attentions)):
                        attentions[j] = attentions[j].detach().cpu().numpy()
                        attentions[j] = np.array(attentions[j])
                        all_attentions[j].append(attentions[j])
                if output_hidden_states:
                    if not all_hidden_states:
                        all_hidden_states = [[] for _ in range(len(hidden_states))]
                    for j in range(len(hidden_states)):
                        hidden_states[j] = hidden_states[j].detach().cpu().numpy()
                        hidden_states[j] = np.array(hidden_states[j])
                        all_hidden_states[j].append(hidden_states[j])

        mean_outputs = np.concatenate(mean_outputs, axis=0)
        full_outputs = np.concatenate(full_outputs, axis=0)

        if output_attentions:
            for j in range(len(all_attentions)):
                all_attentions[j] = np.concatenate(all_attentions[j], axis=0)
        if output_hidden_states:
            for j in range(len(all_hidden_states)):
                all_hidden_states[j] = np.concatenate(all_hidden_states[j], axis=0)

        if output_attentions:
            print(">> TimesFMModel attentions", len(attentions), attentions[0].shape)
        if output_hidden_states:
            print(">> TimesFMModel hidden_states", len(hidden_states), hidden_states[0].shape)

        if pmap_pad > 0:
            mean_outputs = mean_outputs[:-pmap_pad, ...]
            full_outputs = full_outputs[:-pmap_pad, ...]

        if window_size is not None:
            mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
            full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]
        if inp_min >= 0 and truncate_negative:
            mean_outputs = np.maximum(mean_outputs, 0.0)
            full_outputs = np.maximum(full_outputs, 0.0)

        if return_dict:
            result = TimesFMOutput()
            result.mean_predictions = mean_outputs
            result.full_predictions = full_outputs
            if output_attentions:
                result.attentions = all_attentions
            if output_hidden_states:
                result.hidden_states = all_hidden_states

            return result
        else:
            return_tuple = [mean_outputs, full_outputs]
            if output_attentions:
                return_tuple.append(all_attentions)
            if output_hidden_states:
                return_tuple.append(all_hidden_states)
            return tuple(return_tuple)
