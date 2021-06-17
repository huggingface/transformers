import os
from typing import TYPE_CHECKING, List, Optional, Union

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

    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any :obj:`AutoModelForImageClassification`. This pipeline predicts the class of
    an image.

    This image classification pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"image-classification"`.

    See the list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=image-classification>`__.
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

        self.check_model_type(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING)

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

    def __call__(self, images: Union[str, List[str], "Image", List["Image"]], top_k=5):
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
            top_k (:obj:`int`, `optional`, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **label** (:obj:`str`) -- The label identified by the model.
            - **score** (:obj:`int`) -- The score attributed by the model for that label.
        """
        is_batched = isinstance(images, list)

        if not is_batched:
            images = [images]

        images = [self.load_image(image) for image in images]

        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        with torch.no_grad():
            inputs = self.feature_extractor(images=images, return_tensors="pt")
            outputs = self.model(**inputs)

            probs = outputs.logits.softmax(-1)
            scores, ids = probs.topk(top_k)

            scores = scores.tolist()
            ids = ids.tolist()

        if not is_batched:
            scores, ids = scores[0], ids[0]
            labels = [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
        else:
            labels = []
            for scores, ids in zip(scores, ids):
                labels.append(
                    [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
                )

        return labels
