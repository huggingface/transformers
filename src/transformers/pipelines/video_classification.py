from io import BytesIO
from typing import List, Union

import requests

from ..utils import add_end_docstrings, is_decord_available, is_torch_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_decord_available():
    import numpy as np

    from decord import VideoReader, cpu


if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
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
        requires_backends(self, "decord")
        self.check_model_type(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING)

        self.frame_sampling_rate = kwargs.pop("frame_sample_rate", 4)
        self.num_frames = self.model.config.num_frames

    def _sanitize_parameters(self, top_k=None):
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return {}, {}, postprocess_params

    def __call__(self, videos: Union[str, List[str]], **kwargs):
        """
        Assign labels to the video(s) passed as inputs.

        Args:
            videos (`str`, `List[str]`):
                The pipeline handles three types of videos:

                - A string containing a http link pointing to a video
                - A string containing a local path to an video

                The pipeline accepts either a single video or a batch of videos, which must then be passed as a string.
                Videos in a batch must all be in the same format: all as http links or all as local paths.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single video, will return a
            dictionary, if the input is a list of several videos, will return a list of dictionaries corresponding to
            the videos.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        return super().__call__(videos, **kwargs)

    def preprocess(self, video):

        if video.startswith("http://") or video.startswith("https://"):
            video = BytesIO(requests.get(video).content)

        videoreader = VideoReader(video, num_threads=1, ctx=cpu(0))
        videoreader.seek(0)

        start_idx = 0
        end_idx = self.num_frames * self.frame_sampling_rate - 1
        indices = np.linspace(start_idx, end_idx, num=self.num_frames)
        indices = np.clip(indices, start_idx, end_idx).astype(np.int64)

        video = videoreader.get_batch(indices).asnumpy()
        video = list(video)

        model_inputs = self.feature_extractor(video, return_tensors=self.framework)
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
