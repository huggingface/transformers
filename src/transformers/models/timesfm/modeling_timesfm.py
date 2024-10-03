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
import multiprocessing
from typing import Any, Sequence
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from ...modeling_utils import PreTrainedModel
from .configuration_timesfm import TimesFMConfig
from .timesfm_layers import *

# TODO: shall remove this dependency after API design is finalized.
from utilsforecast.processing import make_future_dataframe


class TimesFMPreTrainedModel(PreTrainedModel):
    """handles the loading for all models."""

    config_class = TimesFMConfig
    base_model_prefix = "timesfm"

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
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
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
    ) -> torch.Tensor:
        num_outputs = len(self.config.quantiles) + 1
        model_input, patched_padding, stats, _ = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )
        f_emb = self.freq_emb(freq)  # B x 1 x D
        model_input += f_emb
        model_output = self.stacked_transformer(model_input, patched_padding)

        output_ts = self._postprocess_output(model_output, num_outputs, stats)
        return output_ts

    def decode(
        self,
        input_ts: torch.Tensor,
        paddings: torch.Tensor,
        freq: torch.LongTensor,
        horizon_len: int,
        output_patch_len: int | None = None,
        max_len: int = 512,
        return_forecast_on_context: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            fprop_outputs = self(input_ts, input_padding, freq)
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

        return (full_outputs[:, :, 0], full_outputs)


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
        self.backend = config.backend
        self.quantiles = config.quantiles
        self.num_heads = config.num_heads

        self.num_cores = 1
        self.per_core_batch_size = config.per_core_batch_size
        self.global_batch_size = config.per_core_batch_size * self.num_cores
        self._horizon_start = self.context_len - self.input_patch_len
        self._device = config.backend

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

        return (
            np.stack(input_ts, axis=0),
            np.stack(input_padding, axis=0),
            np.array(inp_freq).astype(np.int32).reshape(-1, 1),
            pmap_pad,
        )

    def forecast(
        self,
        inputs: Sequence[Any],
        freq: Sequence[int] | None = None,
        window_size: int | None = None,
        forecast_context_len: int | None = None,
        return_forecast_on_context: bool = False,
        truncate_negative: bool = False,
    ) -> tuple[np.array, np.array]:
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
        raise NotImplementedError("`forecast` is not implemented.")

    def forecast_on_df(
        self,
        inputs: pd.DataFrame,
        freq: str,
        forecast_context_len: int = 0,
        value_name: str = "values",
        model_name: str = "timesfm",
        window_size: int | None = None,
        num_jobs: int = 1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Forecasts on a list of time series.

        Args:
          inputs: A pd.DataFrame of all time series. The dataframe should have a
            `unique_id` column for identifying the time series, a `ds` column for
            timestamps and a value column for the time series values.
          freq: string valued `freq` of data. Notice this is different from the
            `freq` required by `forecast`. See `freq_map` for allowed values.
          forecast_context_len: If provided none zero, we take the last
            `forecast_context_len` time-points from each series as the forecast
            context instead of the `context_len` set by the model.
          value_name: The name of the value column.
          model_name: name of the model to be written into future df.
          window_size: window size of trend + residual decomposition. If None then
            we do not do decomposition.
          num_jobs: number of parallel processes to use for dataframe processing.
          verbose: output model states in terminal.

        Returns:
          Future forecasts dataframe.
        """
        if not (
            "unique_id" in inputs.columns
            and "ds" in inputs.columns
            and value_name in inputs.columns
        ):
            raise ValueError(
                f"DataFrame must have unique_id, ds and {value_name} columns."
            )
        if not forecast_context_len:
            forecast_context_len = self.context_len
        logging.info("Preprocessing dataframe.")
        df_sorted = inputs.sort_values(by=["unique_id", "ds"])
        new_inputs = []
        uids = []
        if num_jobs == 1:
            if verbose:
                print("Processing dataframe with single process.")
            for key, group in df_sorted.groupby("unique_id"):
                inp, uid = process_group(
                    key,
                    group,
                    value_name,
                    forecast_context_len,
                )
                new_inputs.append(inp)
                uids.append(uid)
        else:
            if num_jobs == -1:
                num_jobs = multiprocessing.cpu_count()
            if verbose:
                print("Processing dataframe with multiple processes.")
            with multiprocessing.Pool(processes=num_jobs) as pool:
                results = pool.starmap(
                    process_group,
                    [
                        (key, group, value_name, forecast_context_len)
                        for key, group in df_sorted.groupby("unique_id")
                    ],
                )
            new_inputs, uids = zip(*results)
        if verbose:
            print("Finished preprocessing dataframe.")
        freq_inps = [freq_map(freq)] * len(new_inputs)
        _, full_forecast = self.forecast(
            new_inputs, freq=freq_inps, window_size=window_size
        )
        if verbose:
            print("Finished forecasting.")
        fcst_df = make_future_dataframe(
            uids=uids,
            last_times=df_sorted.groupby("unique_id")["ds"].tail(1),
            h=self.horizon_len,
            freq=freq,
        )
        fcst_df[model_name] = full_forecast[:, 0 : self.horizon_len, 0].reshape(-1, 1)

        for i, q in enumerate(self.quantiles):
            q_col = f"{model_name}-q-{q}"
            fcst_df[q_col] = full_forecast[:, 0 : self.horizon_len, 1 + i].reshape(
                -1, 1
            )
            if q == 0.5:
                fcst_df[model_name] = fcst_df[q_col]
        logging.info("Finished creating output dataframe.")
        return fcst_df

    def forward(self, x, **kwargs):
        if isinstance(x, pd.DataFrame):
            assert "freq" in kwargs, "Frequency must be provided for DataFrame input."
            return self.forecast_on_df(x, **kwargs)
        else:
            return self.forecast(x, **kwargs)
