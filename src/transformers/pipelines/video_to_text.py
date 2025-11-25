# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import warnings
from io import BytesIO
from typing import Any, Union, overload

import httpx

from ..generation import GenerationConfig
from ..utils import (
    add_end_docstrings,
    is_av_available,
    is_torch_available,
    logging,
    requires_backends,
)
from .base import Pipeline, build_pipeline_init_args


if is_av_available():
    import av
    import numpy as np


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_image_processor=True))
class VideoToTextPipeline(Pipeline):
    """
    Video To Text pipeline using a `AutoModelForImageTextToText`. This pipeline predicts a caption for a given video.

    Unless the model you're using explicitly sets these generation parameters in its configuration files
    (`generation_config.json`), the following default values will be used:
    - max_new_tokens: 256

    Example:

    ```python
    >>> from transformers import pipeline

    >>> captioner = pipeline("video-to-text", model="ydshieh/vit-gpt2-coco-en")
    >>> captioner("path/to/video.mp4")
    [{'generated_text': 'a person is setting a table'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This video to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "video-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=video-to-text).
    """

    _pipeline_calls_generate = True
    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = True
    # Make sure the docstring is updated when the default generation config is changed
    _default_generation_config = GenerationConfig(
        max_new_tokens=256,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "av")
        self.check_model_type(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)

    def _sanitize_parameters(self, max_new_tokens=None, generate_kwargs=None, num_frames=None, frame_sampling_rate=None, timeout=None):
        forward_params = {}
        preprocess_params = {}

        if timeout is not None:
            preprocess_params["timeout"] = timeout
        if num_frames is not None:
            preprocess_params["num_frames"] = num_frames
        if frame_sampling_rate is not None:
            preprocess_params["frame_sampling_rate"] = frame_sampling_rate

        if max_new_tokens is not None:
            forward_params["max_new_tokens"] = max_new_tokens
        if generate_kwargs is not None:
            if max_new_tokens is not None and "max_new_tokens" in generate_kwargs:
                raise ValueError(
                    "`max_new_tokens` is defined both as an argument and inside `generate_kwargs` argument, please use"
                    " only 1 version"
                )
            forward_params.update(generate_kwargs)

        if self.assistant_model is not None:
            forward_params["assistant_model"] = self.assistant_model
        if self.assistant_tokenizer is not None:
            forward_params["tokenizer"] = self.tokenizer
            forward_params["assistant_tokenizer"] = self.assistant_tokenizer

        return preprocess_params, forward_params, {}

    @overload
    def __call__(self, inputs: str, **kwargs: Any) -> list[dict[str, Any]]: ...

    @overload
    def __call__(self, inputs: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]: ...

    def __call__(self, inputs: str | list[str] | None = None, **kwargs):
        """
        Generate text captions for the video(s) passed as inputs.

        Args:
            inputs (`str`, `list[str]`):
                The pipeline handles two types of videos:

                - A string containing a http link pointing to a video
                - A string containing a local path to a video

                The pipeline accepts either a single video or a batch of videos, which must then be passed as a string.
                Videos in a batch must all be in the same format: all as http links or all as local paths.
            max_new_tokens (`int`, *optional*):
                The amount of maximum tokens to generate. By default it will use `generate` default.
            num_frames (`int`, *optional*, defaults to `self.model.config.num_frames`):
                The number of frames sampled from the video to run the generation on. If not provided, will default
                to the number of frames specified in the model configuration.
            frame_sampling_rate (`int`, *optional*, defaults to 1):
                The sampling rate used to select frames from the video. If not provided, will default to 1, i.e. every
                frame will be used.
            generate_kwargs (`Dict`, *optional*):
                Pass it to send all of these arguments directly to `generate` allowing full control of this function.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching videos from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key:

            - **generated_text** (`str`) -- The generated text.
        """
        if "videos" in kwargs:
            warnings.warn(
                "The `videos` argument has been renamed to `inputs`. In version 5 of Transformers, `videos` will no longer be accepted",
                FutureWarning,
            )
            inputs = kwargs.pop("videos")
        if inputs is None:
            raise ValueError("Cannot call the video-to-text pipeline without an inputs argument!")
        return super().__call__(inputs, **kwargs)

    def preprocess(self, video, num_frames=None, frame_sampling_rate=1, timeout=None):
        if num_frames is None:
            # Try to get from model config, otherwise use a default
            if hasattr(self.model.config, "num_frames"):
                num_frames = self.model.config.num_frames
            else:
                num_frames = 8  # Default fallback

        if video.startswith("http://") or video.startswith("https://"):
            video = BytesIO(httpx.get(video, follow_redirects=True, timeout=timeout).content)

        container = av.open(video)

        start_idx = 0
        end_idx = num_frames * frame_sampling_rate - 1
        indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=np.int64)

        video_frames = read_video_pyav(container, indices)
        video_frames = list(video_frames)

        # Process video frames through image processor
        # For models that expect a single image, we'll use the first frame or average frames
        # For models that support multiple frames, we'll pass all frames
        model_inputs = self.image_processor(video_frames, return_tensors="pt")
        model_inputs = model_inputs.to(self.dtype)

        # Some models like GIT need input_ids set to None
        if self.model.config.model_type == "git":
            model_inputs["input_ids"] = None

        return model_inputs

    def _forward(self, model_inputs, **generate_kwargs):
        # Git model sets `model_inputs["input_ids"] = None` in `preprocess`. In batch model, the
        # pipeline will group them into a list of `None`, which fail `_forward`. Avoid this by checking it first.
        if (
            "input_ids" in model_inputs
            and isinstance(model_inputs["input_ids"], list)
            and all(x is None for x in model_inputs["input_ids"])
        ):
            model_inputs["input_ids"] = None

        # User-defined `generation_config` passed to the pipeline call take precedence
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config

        inputs = model_inputs.pop(self.model.main_input_name)
        model_outputs = self.model.generate(inputs, **model_inputs, **generate_kwargs)
        return model_outputs

    def postprocess(self, model_outputs):
        records = []
        for output_ids in model_outputs:
            record = {
                "generated_text": self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
            }
            records.append(record)
        return records


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

