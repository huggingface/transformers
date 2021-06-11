import os
from typing import TYPE_CHECKING, List, Optional, Union

import requests

from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available, requires_backends
from ..tokenization_utils import PreTrainedTokenizer
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


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any :obj:`AutoModelForZeroShotImageClassification`. This pipeline predicts the
    class of an image.

    This image classification pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"image-classification"`.

    See the list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=image-classification>`__.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        feature_extractor: PreTrainedFeatureExtractor,
        tokenizer: PreTrainedTokenizer,
        framework: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model, feature_extractor=feature_extractor, tokenizer=tokenizer, framework=framework, **kwargs
        )

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")

        # self.check_model_type(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING)

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

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
        candidate_labels: List[str],
        hypothesis_template: str = "a photo of {}",
    ):
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
            candidate_labels (:obj:`List[str]`):
                The candidate labels for this image
            hypothesis_template (:obj:`str`, `optional`, defaults to :obj:`"This is a photo of a {}"`):
                The sentence used in cunjunction with `candidate_labels` to attempt the image classification by
                replacing the placeholder with the candidate_labels. Then likelihood is estimated by using
                likelihood_per_image

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

        with torch.no_grad():
            images = self.feature_extractor(images=images, return_tensors="pt")
            inputs = self.tokenizer(candidate_labels, return_tensors="pt")
            inputs["pixel_values"] = images.pixel_values
            outputs = self.model(**inputs)

            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            scores = probs.tolist()

        if not is_batched:
            scores = scores[0]
            labels = [
                {"score": score, "label": candidate_label}
                for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
            ]
        else:
            labels = []
            all_scores = scores
            for scores in all_scores:
                element_labels = [
                    {"score": score, "label": candidate_label}
                    for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
                ]
                labels.append(element_labels)
        return labels
