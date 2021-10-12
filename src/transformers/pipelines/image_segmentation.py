import base64
import io
import os
from typing import Any, Dict, List, Union

import numpy as np

import requests

from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available, requires_backends
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_SEGMENTATION_MAPPING

logger = logging.get_logger(__name__)


Prediction = Dict[str, Any]
Predictions = List[Prediction]


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any :obj:`AutoModelForImageSegmentation`. This pipeline predicts masks of objects
    and their classes.

    This image segmntation pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"image-segmentation"`.

    See the list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=image-segmentation>`__.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_IMAGE_SEGMENTATION_MAPPING)

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

    def _sanitize_parameters(self, **kwargs):
        postprocess_kwargs = {}
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "mask_threshold" in kwargs:
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]
        return {}, {}, postprocess_kwargs

    def __call__(self, *args, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

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
            mask_threshold (:obj:`float`, `optional`, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.

        Return:
            A dictionary or a list of dictionaries containing the result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            each image.

            The dictionaries contain the following keys:

            - **label** (:obj:`str`) -- The class label identified by the model.
            - **score** (:obj:`float`) -- The score attributed by the model for that label.
            - **mask** (:obj:`str`) -- base64 string of a grayscale (single-channel) PNG image that contain masks
              information. The PNG image has size (heigth, width) of the original image. Pixel values in the image are
              either 0 or 255 (i.e. mask is absent VS mask is present).
        """

        return super().__call__(*args, **kwargs)

    def preprocess(self, image):
        image = self.load_image(image)
        target_size = torch.IntTensor([[image.height, image.width]])
        inputs = self.feature_extractor(images=[image], return_tensors="pt")
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        outputs = self.model(**model_inputs)
        model_outputs = {"outputs": outputs, "target_size": target_size}
        return model_outputs

    def postprocess(self, model_outputs, threshold=0.9, mask_threshold=0.5):
        raw_annotations = self.feature_extractor.post_process_segmentation(
            model_outputs["outputs"], model_outputs["target_size"], threshold=threshold, mask_threshold=0.5
        )
        raw_annotation = raw_annotations[0]

        raw_annotation["masks"] *= 255  # [0,1] -> [0,255] black and white pixels

        raw_annotation["scores"] = raw_annotation["scores"].tolist()
        raw_annotation["labels"] = [self.model.config.id2label[label.item()] for label in raw_annotation["labels"]]
        raw_annotation["masks"] = [self._get_mask_str(mask) for mask in raw_annotation["masks"].cpu().numpy()]

        # {"scores": [...], ...} --> [{"score":x, ...}, ...]
        keys = ["score", "label", "mask"]
        annotation = [
            dict(zip(keys, vals))
            for vals in zip(raw_annotation["scores"], raw_annotation["labels"], raw_annotation["masks"])
        ]

        return annotation

    def _get_mask_str(self, mask: np.array) -> str:
        """
        Turns mask numpy array into mask base64 str.

        Args:
            mask (np.array): Numpy array (with shape (heigth, width) of the original image) containing masks information. Values in the array are either 0 or 255 (i.e. mask is absent VS mask is present).

        Returns:
            A base64 string of a single-channel PNG image that contain masks information.
        """
        img = Image.fromarray(mask.astype(np.int8), mode="L")
        with io.BytesIO() as out:
            img.save(out, format="PNG")
            png_string = out.getvalue()
            return base64.b64encode(png_string).decode("utf-8")
