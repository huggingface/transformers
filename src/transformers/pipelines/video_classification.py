from io import BytesIO
from typing import List, Union

import requests

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "av")
        self.check_model_type(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES)

    def _sanitize_parameters(self, top_k=None, num_frames=None, frame_sampling_rate=None):
        preprocess_params = {}
        if frame_sampling_rate is not None:
            preprocess_params["frame_sampling_rate"] = frame_sampling_rate
        if num_frames is not None:
            preprocess_params["num_frames"] = num_frames

        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return preprocess_params, {}, postprocess_params

    def __call__(self, videos: Union[str, List[str]], **kwargs):
        """
        Assign labels to the video(s) passed as inputs.

        Args:
            videos (`str`, `List[str]`):
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

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single video, will return a
            dictionary, if the input is a list of several videos, will return a list of dictionaries corresponding to
            the videos.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        return super().__call__(videos, **kwargs)

    def preprocess(self, video, num_frames=None, frame_sampling_rate=1):
        if num_frames is None:
            num_frames = self.model.config.num_frames

        if video.startswith("http://") or video.startswith("https://"):
            video = BytesIO(requests.get(video).content)

        container = av.open(video)

        start_idx = 0
        end_idx = num_frames * frame_sampling_rate - 1
        indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=np.int64)

        video = read_video_pyav(container, indices)
        video = list(video)

        model_inputs = self.image_processor(video, return_tensors=self.framework)
        if self.framework == "pt":
            model_inputs = model_inputs.to(self.torch_dtype)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=5):
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        if self.framework == "pt":
            probs = model_outputs.logits.softmax(-1)[0]
            scores, ids = probs.topk(top_k)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

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
