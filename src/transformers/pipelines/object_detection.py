import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import requests

from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available, requires_backends
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

        requires_backends(self, "vision")

        self.check_model_type(MODEL_FOR_OBJECT_DETECTION_MAPPING)

        self.feature_extractor = feature_extractor

    @staticmethod
    def load_image(image: Union[str, "Image.Image"]):
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                image = Image.open(requests.get(image, stream=True).raw)
            elif os.path.isfile(image):
                image = Image.open(image)
            else:
                raise ValueError(
                    f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
                )
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError(
                "Incorrect format used for image. Should be a URL linking to an image, a local path, or a PIL image."
            )
        image = image.convert("RGB")
        return image

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

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            threshold (:obj:`float`, `optional`, defaults to 0.9):
                The probability necessary to make a prediction.

        Return:
            A list of dictionaries or a list of list of dictionaries containing the result. If the input is a single
            image, will return a list of dictionaries, if the input is a list of several images, will return a list of
            list of dictionaries corresponding to each image.

            The dictionaries contain the following keys:

            - **label** (:obj:`str`) -- The class label identified by the model.
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

            if self.framework == "pt":
                target_sizes = torch.IntTensor([[im.height, im.width] for im in images])
            else:
                raise ValueError("The ObjectDetectionPipeline is only available in PyTorch.")

            raw_annotations = self.feature_extractor.post_process(outputs, target_sizes)
            annotations = []
            for annotation in raw_annotations:
                keep = annotation["scores"] > threshold
                scores = annotation["scores"][keep]
                labels = annotation["labels"][keep]
                boxes = annotation["boxes"][keep]

                annotation["scores"] = scores.tolist()
                annotation["labels"] = [self.model.config.id2label[label.item()] for label in labels]
                annotation["boxes"] = [self._get_bounding_box(box) for box in boxes]

                # {"scores": [...], ...} --> [{"score":x, ...}, ...]
                keys = ["score", "label", "box"]
                annotation = [
                    dict(zip(keys, vals))
                    for vals in zip(annotation["scores"], annotation["labels"], annotation["boxes"])
                ]

                annotations.append(annotation)

        if not is_batched:
            return annotations[0]

        return annotations

    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (torch.Tensor): Tensor containing the coordinates in corners format.

        Returns:
            bbox (Dict[str, int]): Dict containing the coordinates in corners format.
        """
        if self.framework != "pt":
            raise ValueError("The ObjectDetectionPipeline is only available in PyTorch.")
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox
