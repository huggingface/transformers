# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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
"""Video processor for MiniCPM-V 4.7.

MiniCPM-V treats video as a sequence of images: frames are extracted and
optionally sub-second frames are stacked into composite images.
"""

import math

from ...utils import auto_docstring, logging
from ..minicpmv4_6.video_processing_minicpmv4_6 import MiniCPMV4_6VideoProcessor


logger = logging.get_logger(__name__)


@auto_docstring
class MiniCPMV4_7VideoProcessor(MiniCPMV4_6VideoProcessor):
    def get_sliced_grid(
        self,
        video_size: tuple[int, int],
        max_slice_nums: int,
        scale_resolution: int,
    ) -> list[int] | None:
        original_height, original_width = video_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1:
            return None

        best_grid = [1, 1]
        min_error = float("inf")
        for num_slices in [multiple - 1, multiple, multiple + 1]:
            if num_slices == 1 or num_slices > max_slice_nums:
                continue
            for num_rows in range(1, num_slices + 1):
                if num_slices % num_rows == 0:
                    num_cols = num_slices // num_rows
                    error = abs(log_ratio - math.log(num_cols / num_rows))
                    if error < min_error:
                        best_grid = [num_rows, num_cols]
                        min_error = error
                    elif error == min_error and num_rows > best_grid[0]:
                        best_grid = [num_rows, num_cols]
        return best_grid


__all__ = ["MiniCPMV4_7VideoProcessor"]
