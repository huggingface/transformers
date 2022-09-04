from typing import Dict, List, Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotObjectDetectionPipeline(Pipeline):
    """
    Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of
    objects when you provide an image and a set of `candidate_labels`.

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-object-detection"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        requires_backends(self, "vision")

    def __call__(self, images: Union[str, List[str], "Image", List["Image"]], **kwargs):
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

            candidate_labels (`List[str]`):
                The candidate labels for this image

            hypothesis_template (`str`, *optional*, defaults to `"This is a photo of {}"`):
                The sentence used in cunjunction with *candidate_labels* for assigning labels to predicted bounding
                boxes by replacing the placeholder with the candidate_labels. Then likelihood is estimated by using
                logits

        Return:
            A list of dictionaries containing result, one dictionary per proposed label. The dictionaries contain the
            following keys:

            - **label** (`str`) -- The label identified by the model. It is one of the suggested `candidate_label`.
            - **score** (`float`) -- The score attributed by the model for that label (between 0 and 1).
            - **box** (`List[Dict[str,int]]`) -- The bounding box of detected object in image's original size.
        """
        return super().__call__(images, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        postprocess_params = {}
        if "threshold" in kwargs:
            postprocess_params["threshold"] = kwargs["threshold"]
        return preprocess_params, {}, postprocess_params

    def preprocess(self, image, candidate_labels=[None], hypothesis_template="This is a photo of {}."):
        image = load_image(image)
        target_size = torch.IntTensor([[image.height, image.width]])
        images = self.feature_extractor(images=[image], return_tensors=self.framework)
        inputs = self.tokenizer(
            [hypothesis_template.format(label) for label in candidate_labels],
            return_tensors=self.framework,
            padding=True,
        )
        inputs["pixel_values"] = images.pixel_values
        inputs["target_size"] = target_size
        return {"candidate_labels": candidate_labels, "target_size": target_size, **inputs}

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        candidate_labels = model_inputs.pop("candidate_labels")
        outputs = self.model(**model_inputs)

        model_outputs = outputs.__class__(
            {"target_size": target_size, "candidate_labels": candidate_labels, **outputs}
        )
        return model_outputs

    def postprocess(self, model_outputs, threshold=0.1):
        text = model_outputs["candidate_labels"]

        results = self.feature_extractor.post_process(outputs=model_outputs, target_sizes=model_outputs["target_size"])
        keep = results[0]["scores"] >= threshold
        boxes = results[0]["boxes"][keep]
        scores = results[0]["scores"][keep].tolist()
        labels = results[0]["labels"][keep].tolist()

        return [
            {"score": score, "label": text[label], "box": self._get_bounding_box(box)}
            for score, label, box in zip(scores, labels, boxes)
        ]

    def _get_bounding_box(self, box: "torch.Tensor") -> Dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`Dict[str, int]`): Dict containing the coordinates in corners format.
        """
        if self.framework != "pt":
            raise ValueError("The ZeroShotObjectDetectionPipeline is only available in PyTorch.")
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox
