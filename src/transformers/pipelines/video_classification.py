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
from typing import Any, Optional, Union, overload

import httpx

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
    from ..models.auto.modeling_auto import MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class VideoClassificationPipeline(Pipeline):
    """
    Video classification pipeline using any `AutoModelForVideoClassification`. This pipeline predicts the class of a
    video.

    This video classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"video-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=video-classification).
    """

    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "av")
        self.check_model_type(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES)

    def _sanitize_parameters(self, top_k=None, num_frames=None, frame_sampling_rate=None, function_to_apply=None):
        preprocess_params = {}
        if frame_sampling_rate is not None:
            preprocess_params["frame_sampling_rate"] = frame_sampling_rate
        if num_frames is not None:
            preprocess_params["num_frames"] = num_frames

        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        if function_to_apply is not None:
            if function_to_apply not in ["softmax", "sigmoid", "none"]:
                raise ValueError(
                    f"Invalid value for `function_to_apply`: {function_to_apply}. "
                    "Valid options are ['softmax', 'sigmoid', 'none']"
                )
            postprocess_params["function_to_apply"] = function_to_apply
        else:
            postprocess_params["function_to_apply"] = "softmax"
        return preprocess_params, {}, postprocess_params

    @overload
    def __call__(self, inputs: str, **kwargs: Any) -> list[dict[str, Any]]: ...

    @overload
    def __call__(self, inputs: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]: ...

    def __call__(self, inputs: Optional[Union[str, list[str]]] = None, **kwargs):
        """
        Assign labels to the video(s) passed as inputs.

        Args:
            inputs (`str`, `list[str]`):
                The pipeline handles three types of videos:

                - A string containing a http link pointing to a video
                - A string containing a local path to a video

                The pipeline accepts either a single video or a batch of videos, which must then be passed as a string.
                Videos in a batch must all be in the same format: all as http links or all as local paths.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
            num_frames (`int`, *optional*, defaults to `self.model.config.num_frames`):
                The number of frames sampled from the video to run the classification on. If not provided, will default
                to the number of frames specified in the model configuration.
            frame_sampling_rate (`int`, *optional*, defaults to 1):
                The sampling rate used to select frames from the video. If not provided, will default to 1, i.e. every
                frame will be used.
            function_to_apply(`str`, *optional*, defaults to "softmax"):
                The function to apply to the model output. By default, the pipeline will apply the softmax function to
                the output of the model. Valid options: ["softmax", "sigmoid", "none"]. Note that passing Python's
                built-in `None` will default to "softmax", so you need to pass the string "none" to disable any
                post-processing.

        Return:
            A list of dictionaries or a list of list of dictionaries containing result. If the input is a single video,
            will return a list of `top_k` dictionaries, if the input is a list of several videos, will return a list of list of
            `top_k` dictionaries corresponding to the videos.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        # After deprecation of this is completed, remove the default `None` value for `images`
        if "videos" in kwargs:
            warnings.warn(
                "The `videos` argument has been renamed to `inputs`. In version 5 of Transformers, `videos` will no longer be accepted",
                FutureWarning,
            )
            inputs = kwargs.pop("videos")
        if inputs is None:
            raise ValueError("Cannot call the video-classification pipeline without an inputs argument!")
        return super().__call__(inputs, **kwargs)

    def preprocess(self, video, num_frames=None, frame_sampling_rate=1):
        if num_frames is None:
            num_frames = self.model.config.num_frames

        if video.startswith("http://") or video.startswith("https://"):
            video = BytesIO(httpx.get(video, follow_redirects=True).content)

        container = av.open(video)

        start_idx = 0
        end_idx = num_frames * frame_sampling_rate - 1
        indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=np.int64)

        video = read_video_pyav(container, indices)
        video = list(video)

        model_inputs = self.image_processor(video, return_tensors="pt")
        model_inputs = model_inputs.to(self.dtype)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=5, function_to_apply="softmax"):
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        if function_to_apply == "softmax":
            probs = model_outputs.logits[0].softmax(-1)
        elif function_to_apply == "sigmoid":
            probs = model_outputs.logits[0].sigmoid()
        else:
            probs = model_outputs.logits[0]
        scores, ids = probs.topk(top_k)

        scores = scores.tolist()
        ids = ids.tolist()
        return [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]


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
