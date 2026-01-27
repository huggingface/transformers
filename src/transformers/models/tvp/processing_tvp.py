# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for TVP.
"""

from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class TvpProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "truncation": True,
            "padding": "max_length",
            "pad_to_max_length": True,
            "return_token_type_ids": False,
        },
    }


@auto_docstring
class TvpProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)
        self.video_processor = image_processor

    def post_process_video_grounding(self, logits, video_durations):
        """
        Compute the time of the video.

        Args:
            logits (`torch.Tensor`):
                The logits output of TvpForVideoGrounding.
            video_durations (`float`):
                The video's duration.

        Returns:
            start (`float`):
                The start time of the video.
            end (`float`):
                The end time of the video.
        """
        start, end = (
            round(logits.tolist()[0][0] * video_durations, 1),
            round(logits.tolist()[0][1] * video_durations, 1),
        )

        return start, end


__all__ = ["TvpProcessor"]
