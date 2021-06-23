import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import requests

from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available, requires_backends
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel

if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


Prediction = Dict[str, Any]
Predictions = List[Prediction]


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any :obj:`AutoModelForImageSegmentation`. This pipeline predicts the class of an
    image.

    This image segmentation pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"image-segmentation"`.

    See the list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=image-segmentation>`__.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        feature_extractor: PreTrainedFeatureExtractor,
        framework: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, feature_extractor=feature_extractor, framework=framework, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")

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
        mask_threshold: Optional[float] = 0.9,
    ) -> Union[Predictions, List[Prediction]]:
        """
        Assign labels to the image(s) passed as inputs.

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
            mask_threshold (:obj:`float`, `optional`, defaults to 0.9):
                The probability necessary to keep the pixel within the mask of said prediction.

        Return:
            A list of dictionaries or a list of of list of dictionaries containing result. If the input is a single
            image, will return a list of dictionaries, if the input is a list of several images, will return a list of
            list of dictionaries corresponding to each image.

            The dictionaries contain the following keys:

            - **label** (:obj:`str`) -- The label identified by the model. There could be duplicates within the list
              (corresponding to instances within a panoptic segmentation.
            - **score** (:obj:`int`) -- The score attributed by the model for that label as a probability.
            - **mask** (:obj:`np.array`) -- The bitmask of shape (H, W) of the input image, 0 means the class is not
              affected to the label, 1 means it is.
        """
        is_batched = isinstance(images, list)

        if not is_batched:
            images = [images]

        images = [self.load_image(image) for image in images]

        with torch.no_grad():
            inputs = self.feature_extractor(images=images, return_tensors="pt")
            outputs = self.model(**inputs)

            processed_sizes = [(im.height, im.width) for im in images]

            annotations = []
            for logits, masks, target_size in zip(outputs.logits, outputs.pred_masks, processed_sizes):
                scores, labels = logits.softmax(-1).max(-1)
                keep = labels.ne(outputs.logits.shape[-1] - 1) & (scores > threshold)
                scores = scores[keep]
                labels = labels[keep]
                masks = masks[keep]
                masks = torch.nn.functional.interpolate(masks[:, None], target_size, mode="bilinear").squeeze(1)
                masks = masks.sigmoid() > mask_threshold

                annotation = []
                for mask, score, label in zip(masks, scores, labels):
                    annotation.append(
                        {
                            "mask": mask.cpu().numpy(),
                            "score": score.item(),
                            "label": self.model.config.id2label[label.item()],
                        }
                    )

                annotations.append(annotation)

        if not is_batched:
            return annotations[0]

        return annotations
