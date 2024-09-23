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
from os import path
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
import timesfm_base
import patched_decoder as ppd
from ...modeling_utils import PreTrainedModel


_TOL = 1e-6


class TimesFmTorch(PreTrainedModel, timesfm_base.TimesFmBase):
    """TimesFM forecast API for inference."""

    def __post_init__(self):
        self._model_config = ppd.TimesFMConfig(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.model_dims,
            intermediate_size=self.model_dims,
            patch_len=self.input_patch_len,
            horizon_len=self.output_patch_len,
            head_dim=self.model_dims // self.num_heads,
            quantiles=self.quantiles,
        )
        self._model = None
        self.num_cores = 1
        self.global_batch_size = self.per_core_batch_size
        self._device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and self.backend == "gpu") else "cpu"
        )

    def load_from_checkpoint(
        self,
        checkpoint: timesfm_base.TimesFmCheckpoint,
    ) -> None:
        """Loads a checkpoint and compiles the decoder."""
        checkpoint_path = checkpoint.path
        repo_id = checkpoint.huggingface_repo_id
        if checkpoint_path is None:
            checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
        self._model = ppd.PatchedTimeSeriesDecoder(self._model_config)
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
        logging.info("Loading checkpoint from %s", checkpoint_path)
        self._model.load_state_dict(loaded_checkpoint)
        logging.info("Sending checkpoint to device %s", f"{self._device}")
        self._model.to(self._device)
        self._model.eval()
        # TODO: add compilation.

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
                new_inputs.extend(timesfm_base.moving_average(ts, window_size))
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
