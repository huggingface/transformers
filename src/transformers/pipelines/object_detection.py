import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from torch.functional import Tensor

import requests

from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import (
    add_end_docstrings,
    is_timm_available,
    is_torch_available,
    is_vision_available,
    requires_backends,
)
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_OBJECT_DETECTION_MAPPING

logger = logging.get_logger(__name__)


Prediction = Dict[str, Any]
Predictions = List[Prediction]


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ObjectDetectionPipeline(Pipeline):
    """
    Object detection pipeline using any :obj:`AutoModelForObjectDetection`. This pipeline predicts bounding boxes of
    objects and their classes.

    This object detection pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"object-detection"`.

    See the list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=object-detection>`__.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        feature_extractor: PreTrainedFeatureExtractor,
        framework: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, feature_extractor=feature_extractor, framework=framework, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "timm")
        requires_backends(self, "vision")

        self.check_model_type(MODEL_FOR_OBJECT_DETECTION_MAPPING)

        self.feature_extractor = feature_extractor

    @staticmethod
    def load_image(image: Union[str, "Image.Image"]):
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                return Image.open(requests.get(image, stream=True).raw)
            elif os.path.isfile(image):
                return Image.open(image)
        elif isinstance(image, Image.Image):
            return image

        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )

    def __call__(
        self,
        images: Union[str, List[str], "Image", List["Image"]],
        threshold: Optional[float] = 0.9,
    ) -> Union[Predictions, List[Prediction]]:
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            images (:obj:`str`, :obj:`List[str]`, :obj:`PIL.Image` or :obj:`List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            threshold (:obj:`float`, `optional`, defaults to 0.9):
                The probability necessary to make a prediction.

        Return:
            A list of dictionaries or a list of list of dictionaries containing the result. If the input is a single
            image, will return a list of dictionaries, if the input is a list of several images, will return a list of
            list of dictionaries corresponding to each image.

            The dictionaries contain the following keys:

            - **label** (:obj:`str`) -- The label identified by the model.
            - **score** (:obj:`float`) -- The score attributed by the model for that label.
            - **box** (:obj:`List[Dict[str, int]]`) -- The bounding box of detected object in image's original size.
        """
        is_batched = isinstance(images, list)

        if not is_batched:
            images = [images]

        images = [self.load_image(image) for image in images]

        with torch.no_grad():
            inputs = self.feature_extractor(images=images, return_tensors="pt")
            outputs = self.model(**inputs)

            target_sizes = torch.IntTensor([[im.height, im.width] for im in images])
            annotations = self.feature_extractor.post_process(outputs, target_sizes, threshold)

            for annotation in annotations:
                for detected_obj in annotation:
                    detected_obj["score"] = detected_obj["score"].item()
                    detected_obj["label"] = self.model.config.id2label[detected_obj["label"].item()]
                    detected_obj["box"] = self._get_bounding_box(detected_obj["box"])

        if not is_batched:
            return annotations[0]

        return annotations

    def _get_bounding_box(self, box: Tensor) -> Dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (tensor): Tensor containing the coordinates in corners format.

        Returns:
            bbox (Dict[str, int]): Dict containing the coordinates in corners format.
        """
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox
