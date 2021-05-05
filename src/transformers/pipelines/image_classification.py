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
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any :obj:`ModelForImageClassification`. This pipeline predicts the class of an
    image.

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
        super().__init__(model, framework=framework, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")

        self.check_model_type(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING)

        self.feature_extractor = feature_extractor

    def save_pretrained(self, save_directory: str):
        """
        Save the pipeline's model and feature processor.

        Args:
            save_directory (:obj:`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory)
        self.feature_extractor.save_pretrained(save_directory)
        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

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
            top_k (:obj:`int`):
                The number of top labels that will be returned by the pipeline.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **label** (:obj:`str`) -- The label identified by the model.
            - **score** (:obj:`int`) -- The score attributed by the model for that label.
        """
        is_batched = type(images) == list

        if not is_batched:
            images = [images]

        if type(images[0]) == str and images[0].startswith("http"):
            images = [Image.open(requests.get(image, stream=True).raw) for image in images]
        elif type(images[0]) == str and os.path.isfile(images[0]):
            images = [Image.open(image) for image in images]

        inputs = self.feature_extractor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)

        probs = outputs.logits.softmax(-1)
        scores, ids = probs.topk(top_k)

        scores = scores.detach().tolist()
        ids = ids.detach().tolist()

        labels = []

        if not is_batched:
            scores, ids = scores[0], ids[0]
            labels = [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
        else:
            for scores, ids in zip(scores, ids):
                labels.append(
                    [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
                )

        return labels
