# coding=utf-8
# Copyright 2024 Microsoft and The HuggingFace Inc. team. All rights reserved.
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
"""DPM-Solver++ scheduler for VibeVoice."""

from typing import Optional, Union

import numpy as np

from ...utils import is_diffusers_available, logging

if is_diffusers_available():
    from diffusers import DPMSolverMultistepScheduler
else:
    # If diffusers is not available, we define a dummy class to avoid import errors
    # until the user installs diffusers (checked at runtime or model init)
    class DPMSolverMultistepScheduler:
        def __init__(
            self,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = "linear",
            trained_betas: Optional[Union[np.ndarray, list[float]]] = None,
            solver_order: int = 2,
            prediction_type: str = "epsilon",
            thresholding: bool = False,
            dynamic_thresholding_ratio: float = 0.995,
            sample_max_value: float = 1.0,
            algorithm_type: str = "dpmsolver++",
            solver_type: str = "midpoint",
            lower_order_final: bool = True,
            use_karras_sigmas: bool = False,
            use_lu_lambdas: bool = False,
            lambda_min_clipped: float = -float("inf"),
            variance_type: Optional[str] = None,
        ):
            raise ImportError(
                "VibeVoice requires the `diffusers` library. Please install it with `pip install diffusers`."
            )


logger = logging.get_logger(__name__)

# The VibeVoice implementation relies heavily on the DPMSolverMultistepScheduler from diffusers.
# We modify it slightly or use it as is.
# Based on the inspected code, it seems to be a copy-paste of an older version or modified version.
# For integration into transformers, we prefer to rely on the external library if possible.
# If strict compatibility is required, we would duplicate the full scheduler code here.
# Given the size, we will wrap the diffusers scheduler.


class VibeVoiceDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    """
    Wrapper around diffusers.DPMSolverMultistepScheduler to ensure compatibility.
    """

    pass
