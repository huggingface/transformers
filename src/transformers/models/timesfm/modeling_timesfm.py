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


import dataclasses
import logging
import multiprocessing
from typing import Any, Sequence
from os import path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from ...modeling_utils import PreTrainedModel

import patched_decoder as ppd
from utilsforecast.processing import make_future_dataframe
from configuration_timesfm import TimesFMConfig


def process_group(key, group, value_name, forecast_context_len):
    group = group.tail(forecast_context_len)
    return np.array(group[value_name], dtype=np.float32), key


def moving_average(arr, window_size):
    """Calculates the moving average using NumPy's convolution function."""
    # Pad with zeros to handle initial window positions
    arr_padded = np.pad(arr, (window_size - 1, 0), "constant")
    smoothed_arr = np.convolve(arr_padded, np.ones(window_size), "valid") / window_size
    return [smoothed_arr, arr - smoothed_arr]


def freq_map(freq: str):
    """Returns the frequency map for the given frequency string."""
    freq = str.upper(freq)
    if (
        freq.endswith("H")
        or freq.endswith("T")
        or freq.endswith("MIN")
        or freq.endswith("D")
        or freq.endswith("B")
        or freq.endswith("U")
    ):
        return 0
    elif freq.endswith(("W", "M", "MS")):
        return 1
    elif freq.endswith("Y") or freq.endswith("Q"):
        return 2
    else:
        raise ValueError(f"Invalid frequency: {freq}")


@dataclasses.dataclass(kw_only=True)
class TimesFmCheckpoint:
    """Checkpoint used to initialize a TimesFM model for inference.

    Attributes:
      version: Version of the checkpoint, e.g. "jax", "torch", "tensorflow", etc.
        The factory will create the corresponding TimesFm inference class based on
        this version.
      path: Path to the checkpoint.
      type: If provided, type of the checkpoint used by the specific checkpoint
        loader per version.
      step: If provided, step of the checkpoint.
    """

    version: str = "torch"
    path: str | None = None
    huggingface_repo_id: str | None = None
    type: Any = None
    step: int | None = None


class TimesFmBase:
    """Base TimesFM forecast API for inference.

    This class is the scaffolding for calling TimesFM forecast. To properly use:
      1. Create an instance with the correct hyperparameters of a TimesFM model.
      2. Call `load_from_checkpoint` to load a compatible checkpoint.
      3. Call `forecast` for inference.
    """

    def _logging(self, s):
        print(s)

    def __init__(self, hparams: TimesFMConfig) -> None:
        """Initializes the TimesFM forecast API.

        Args:
          hparams: Hyperparameters of the model.
          checkpoint: Checkpoint to load. Notice `checkpoint.version` will decide
            which TimesFM version to use.
        """
        self.hparams = hparams

        # Expand hparams for conciseness within the model code.
        self.context_len = hparams.context_len
        self.horizon_len = hparams.horizon_len
        self.input_patch_len = hparams.patch_len
        self.output_patch_len = hparams.horizon_len
        self.num_layers = hparams.num_layers
        self.model_dims = hparams.model_dim
        self.backend = hparams.backend
        self.quantiles = hparams.quantiles
        self.num_heads = hparams.num_heads

        # Rewrite these values in subclasses for SPMD.
        self.num_cores = 1
        self.per_core_batch_size = hparams.per_core_batch_size
        self.global_batch_size = hparams.per_core_batch_size
        self._horizon_start = self.context_len - self.input_patch_len

    def load_from_checkpoint(self, checkpoint: TimesFmCheckpoint) -> None:
        """Loads a checkpoint and compiles the decoder."""
        raise NotImplementedError("`load_from_checkpoint` is not implemented.")

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


class TimesFMModel(TimesFmBase, nn.Module):
    """Body of the TimesFM model, excluding the head."""

    def __post_init__(self):
        self._model_config = TimesFMConfig(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.model_dims,
            intermediate_size=self.model_dims,
            patch_len=self.input_patch_len,
            horizon_len=self.output_patch_len,
            head_dim=self.model_dims // self.num_heads,
            quantiles=self.quantiles,
        )

        self.num_cores = 1
        self.global_batch_size = self.per_core_batch_size
        self._device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and self.backend == "gpu") else "cpu"
        )
        self._model = ppd.PatchedTimeSeriesDecoder(self._model_config)
        self._model.to(self._device)
        self._model.eval()

    def forecast(
        self,
        inputs: Sequence[Any],
        freq: Sequence[int] | None = None,
        window_size: int | None = None,
        forecast_context_len: int | None = None,
        return_forecast_on_context: bool = False,
        truncate_negative: bool = False,
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
        if forecast_context_len is None:
            fcontext_len = self.context_len
        else:
            fcontext_len = forecast_context_len
        inputs = [np.array(ts)[-fcontext_len:] for ts in inputs]
        inp_min = np.min([np.min(ts) for ts in inputs])

        if window_size is not None:
            new_inputs = []
            for ts in inputs:
                new_inputs.extend(moving_average(ts, window_size))
            inputs = new_inputs

        if freq is None:
            logging.info("No frequency provided via `freq`. Default to high (0).")
            freq = [0] * len(inputs)

        input_ts, input_padding, inp_freq, pmap_pad = self._preprocess(inputs, freq)
        with torch.no_grad():
            mean_outputs = []
            full_outputs = []
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
                ).to(self._device)
                input_padding_in = torch.from_numpy(
                    np.array(
                        input_padding[
                            i
                            * self.global_batch_size : (i + 1)
                            * self.global_batch_size
                        ],
                        dtype=np.float32,
                    )
                ).to(self._device)
                inp_freq_in = (
                    torch.from_numpy(
                        np.array(
                            inp_freq[
                                i
                                * self.global_batch_size : (i + 1)
                                * self.global_batch_size,
                                :,
                            ],
                            dtype=np.int32,
                        )
                    )
                    .long()
                    .to(self._device)
                )
                mean_output, full_output = self._model.decode(
                    input_ts=input_ts_in,
                    paddings=input_padding_in,
                    freq=inp_freq_in,
                    horizon_len=self.horizon_len,
                    return_forecast_on_context=return_forecast_on_context,
                )
                mean_output = mean_output.detach().cpu().numpy()
                full_output = full_output.detach().cpu().numpy()
                mean_output = np.array(mean_output)
                full_output = np.array(full_output)
                mean_outputs.append(mean_output)
                full_outputs.append(full_output)

        mean_outputs = np.concatenate(mean_outputs, axis=0)
        full_outputs = np.concatenate(full_outputs, axis=0)

        if pmap_pad > 0:
            mean_outputs = mean_outputs[:-pmap_pad, ...]
            full_outputs = full_outputs[:-pmap_pad, ...]

        if window_size is not None:
            mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
            full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]
        if inp_min >= 0 and truncate_negative:
            mean_outputs = np.maximum(mean_outputs, 0.0)
            full_outputs = np.maximum(full_outputs, 0.0)
        return mean_outputs, full_outputs


class TimesFMModel(TimesFmBase, nn.Module):
    """TimesFM forecast API for inference."""

    def __init__(self, hparams: TimesFMConfig) -> None:
        super.__init__(hparams)
        self._model_config = hparams
        self._model = ppd.PatchedTimeSeriesDecoder(self._model_config)
        self.num_cores = 1
        self.global_batch_size = self.per_core_batch_size
        self._device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and self.backend == "gpu") else "cpu"
        )

    def forecast(
        self,
        inputs: Sequence[Any],
        freq: Sequence[int] | None = None,
        window_size: int | None = None,
        forecast_context_len: int | None = None,
        return_forecast_on_context: bool = False,
        truncate_negative: bool = False,
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
        if not self._model:
            raise ValueError(
                "Checkpoint not loaded. Call `load_from_checkpoint` before"
                " `forecast`."
            )
        if forecast_context_len is None:
            fcontext_len = self.context_len
        else:
            fcontext_len = forecast_context_len
        inputs = [np.array(ts)[-fcontext_len:] for ts in inputs]
        inp_min = np.min([np.min(ts) for ts in inputs])

        if window_size is not None:
            new_inputs = []
            for ts in inputs:
                new_inputs.extend(moving_average(ts, window_size))
            inputs = new_inputs

        if freq is None:
            logging.info("No frequency provided via `freq`. Default to high (0).")
            freq = [0] * len(inputs)

        input_ts, input_padding, inp_freq, pmap_pad = self._preprocess(inputs, freq)
        with torch.no_grad():
            mean_outputs = []
            full_outputs = []
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
                ).to(self._device)
                input_padding_in = torch.from_numpy(
                    np.array(
                        input_padding[
                            i
                            * self.global_batch_size : (i + 1)
                            * self.global_batch_size
                        ],
                        dtype=np.float32,
                    )
                ).to(self._device)
                inp_freq_in = (
                    torch.from_numpy(
                        np.array(
                            inp_freq[
                                i
                                * self.global_batch_size : (i + 1)
                                * self.global_batch_size,
                                :,
                            ],
                            dtype=np.int32,
                        )
                    )
                    .long()
                    .to(self._device)
                )
                mean_output, full_output = self._model.decode(
                    input_ts=input_ts_in,
                    paddings=input_padding_in,
                    freq=inp_freq_in,
                    horizon_len=self.horizon_len,
                    return_forecast_on_context=return_forecast_on_context,
                )
                mean_output = mean_output.detach().cpu().numpy()
                full_output = full_output.detach().cpu().numpy()
                mean_output = np.array(mean_output)
                full_output = np.array(full_output)
                mean_outputs.append(mean_output)
                full_outputs.append(full_output)

        mean_outputs = np.concatenate(mean_outputs, axis=0)
        full_outputs = np.concatenate(full_outputs, axis=0)

        if pmap_pad > 0:
            mean_outputs = mean_outputs[:-pmap_pad, ...]
            full_outputs = full_outputs[:-pmap_pad, ...]

        if window_size is not None:
            mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
            full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]
        if inp_min >= 0 and truncate_negative:
            mean_outputs = np.maximum(mean_outputs, 0.0)
            full_outputs = np.maximum(full_outputs, 0.0)
        return mean_outputs, full_outputs

    def forward(self, x, **kwargs):
        if isinstance(x, pd.DataFrame):
            assert "freq" in kwargs, "Frequency must be provided for DataFrame input."
            return self.forecast_on_df(x, **kwargs)
        else:
            return self.forecast(x, **kwargs)


## TODO: Define the PreTrainedTimesFMModel class
