from typing import List, Union

from ..utils import (
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import PIPELINE_INIT_ARGS, ChunkPipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf

    from ..tf_utils import stable_softmax

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotImageClassificationPipeline(ChunkPipeline):
    """
    Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you
    provide an image and a set of `candidate_labels`.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="openai/clip-vit-large-patch14")
    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["animals", "humans", "landscape"],
    ... )
    [{'score': 0.965, 'label': 'animals'}, {'score': 0.03, 'label': 'humans'}, {'score': 0.005, 'label': 'landscape'}]

    >>> classifier(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["black and white", "photorealist", "painting"],
    ... )
    [{'score': 0.996, 'label': 'black and white'}, {'score': 0.003, 'label': 'photorealist'}, {'score': 0.0, 'label': 'painting'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-image-classification).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

            candidate_labels (`List[str]`):
                The candidate labels for this image

            hypothesis_template (`str`, *optional*, defaults to `"This is a photo of {}"`):
                The sentence used in cunjunction with *candidate_labels* to attempt the image classification by
                replacing the placeholder with the candidate_labels. Then likelihood is estimated by using
                logits_per_image

        Return:
            A list of dictionaries containing result, one dictionary per proposed label. The dictionaries contain the
            following keys:

            - **label** (`str`) -- The label identified by the model. It is one of the suggested `candidate_label`.
            - **score** (`float`) -- The score attributed by the model for that label (between 0 and 1).
        """
        return super().__call__(images, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        return preprocess_params, {}, {}

    def preprocess(self, image, candidate_labels=None, hypothesis_template="This is a photo of {}."):
        n = len(candidate_labels)
        for i, candidate_label in enumerate(candidate_labels):
            image = load_image(image)
            images = self.feature_extractor(images=[image], return_tensors=self.framework)
            sequence = hypothesis_template.format(candidate_label)
            inputs = self.tokenizer(sequence, return_tensors=self.framework)
            inputs["pixel_values"] = images.pixel_values
            yield {"is_last": i == n - 1, "candidate_label": candidate_label, **inputs}

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        candidate_label = model_inputs.pop("candidate_label")
        outputs = self.model(**model_inputs)

        # Clip does crossproduct scoring by default, so we're only
        # interested in the results where image and text and in the same
        # batch position.
        diag = torch.diagonal if self.framework == "pt" else tf.linalg.diag_part
        logits_per_image = diag(outputs.logits_per_image)

        model_outputs = {
            "is_last": is_last,
            "candidate_label": candidate_label,
            "logits_per_image": logits_per_image,
        }
        return model_outputs

    def postprocess(self, model_outputs):
        candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
        if self.framework == "pt":
            logits = torch.cat([output["logits_per_image"] for output in model_outputs])
            probs = logits.softmax(dim=0)
            scores = probs.tolist()
        else:
            logits = tf.concat([output["logits_per_image"] for output in model_outputs], axis=0)
            probs = stable_softmax(logits, axis=0)
            scores = probs.numpy().tolist()

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        return result
