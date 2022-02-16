from typing import List, Union

from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available, requires_backends
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, ChunkPipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotImageClassificationPipeline(ChunkPipeline):
    """
    Image classification pipeline using any `AutoModelForZeroShotImageClassification`. This pipeline predicts the class
    of an image.

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-classification).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        # No specific FOR_XXX available yet
        # self.check_model_type(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING)

    def __call__(self, images: Union[str, List[str], "Image", List["Image"]], **kwargs):
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            candidate_labels (`List[str]`):
                The candidate labels for this image
            hypothesis_template (`str`, *optional*, defaults to `"This is a photo of a {}"`):
                The sentence used in cunjunction with *candidate_labels* to attempt the image classification by
                replacing the placeholder with the candidate_labels. Then likelihood is estimated by using
                likelihood_per_image

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        return super().__call__(images, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        postprocess_params = {}
        if "multi_label" in kwargs:
            postprocess_params["multi_label"] = kwargs["multi_label"]
        return preprocess_params, {}, postprocess_params

    def preprocess(self, image, candidate_labels=None, hypothesis_template="This is a photo of {}."):
        n = len(candidate_labels)
        for i, candidate_label in enumerate(candidate_labels):
            image = load_image(image)
            images = self.feature_extractor(images=[image], return_tensors="pt")
            sequence = hypothesis_template.format(candidate_label)
            inputs = self.tokenizer(sequence, return_tensors="pt")
            inputs["pixel_values"] = images.pixel_values
            yield {"is_last": i == n - 1, "candidate_label": candidate_label, **inputs}

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        candidate_label = model_inputs.pop("candidate_label")
        outputs = self.model(**model_inputs)

        # Clip does crossproduct scoring by default, so we're only
        # interested in the results where image and text and in the same
        # batch position.
        logits_per_image = torch.diagonal(outputs.logits_per_image)

        model_outputs = {
            "is_last": is_last,
            "candidate_label": candidate_label,
            "logits_per_image": logits_per_image,
        }
        return model_outputs

    def postprocess(self, model_outputs, multi_label=False):
        candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
        logits = torch.cat([output["logits_per_image"] for output in model_outputs])
        print("Logits", logits)
        probs = logits.softmax(dim=0)
        scores = probs.tolist()

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        return result
