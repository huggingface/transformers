import base64
import io
from typing import Any, Dict, List, Union

import numpy as np

from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available, requires_backends
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    import torchvision
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
        MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
    )


logger = logging.get_logger(__name__)


Prediction = Dict[str, Any]
Predictions = List[Prediction]


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any `AutoModelForImageSegmentation`. This pipeline predicts masks of objects and
    their classes.

    This image segmntation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-segmentation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-segmentation).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        self.check_model_type(
            dict(MODEL_FOR_IMAGE_SEGMENTATION_MAPPING.items() + MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING.items())
        )

    def _sanitize_parameters(self, **kwargs):
        postprocess_kwargs = {}
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "mask_threshold" in kwargs:
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]
        if "raw_image" in kwargs:
            postprocess_kwargs["raw_image"] = kwargs["raw_image"]
        return {}, {}, postprocess_kwargs

    def __call__(self, *args, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            raw_image (`bool`, *optional*, defaults to False):
                If this is set to True, the `mask` description of the objects will be real `PIL.Image` (grey level). If
                set to False, then, `mask` will be a base64 encoded of a PNG of this image. Which is easier to
                send/save.
            threshold (`float`, *optional*, defaults to 0.9):
                The probability necessary to make a prediction.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.

        Return:
            A dictionary or a list of dictionaries containing the result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            each image.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **score** (`float`) -- The score attributed by the model for that label.
            - **mask** (`str` or `PIL.Image`) -- base64 string of a grayscale (single-channel) PNG image that contain
              masks information. The PNG image has size (heigth, width) of the original image. Pixel values in the
              image are either 0 or 255 (i.e. mask is absent VS mask is present). if `raw_image` is set to `True`, then
              the `mask` is the raw boolean `PIL.Image`.
        """

        return super().__call__(*args, **kwargs)

    def preprocess(self, image):
        image = load_image(image)
        target_size = torch.IntTensor([[image.height, image.width]])
        inputs = self.feature_extractor(images=[image], return_tensors="pt")
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        model_outputs = self.model(**model_inputs)
        model_outputs["target_size"] = target_size
        return model_outputs

    def postprocess(self, model_outputs, raw_image=False, threshold=0.9, mask_threshold=0.5):
        if hasattr(self.feature_extractor, "post_process_segmentation"):
            # Panoptic
            raw_annotations = self.feature_extractor.post_process_segmentation(
                model_outputs, model_outputs["target_size"], threshold=threshold, mask_threshold=0.5
            )
            raw_annotation = raw_annotations[0]
            raw_annotation["masks"] *= 255  # [0,1] -> [0,255] black and white pixels
            raw_annotation["scores"] = raw_annotation["scores"].tolist()
            raw_annotation["labels"] = [self.model.config.id2label[label.item()] for label in raw_annotation["labels"]]
            raw_annotation["masks"] = [
                Image.fromarray(mask.numpy().astype(np.int8), mode="L") for mask in raw_annotation["masks"]
            ]
            if not raw_image:
                raw_annotation["masks"] = [self._get_mask_str(mask) for mask in raw_annotation["masks"]]

            # {"scores": [...], ...} --> [{"score":x, ...}, ...]
            keys = ["score", "label", "mask"]
            annotation = [
                dict(zip(keys, vals))
                for vals in zip(raw_annotation["scores"], raw_annotation["labels"], raw_annotation["masks"])
            ]
        else:
            # Default logits
            logits = model_outputs.logits
            if len(logits.shape) != 4:
                raise ValueError(f"Logits don't have expected dimensions, expected [1, N, H, W], got {logits.shape}")
            # Softmax
            logits = logits.log_softmax(dim=1)
            batch_size, num_labels, height, width = logits.shape
            expected_num_labels = len(self.model.config.id2label)
            if num_labels != expected_num_labels:
                raise ValueError(
                    f"Logits don't have expected dimensions, expected [1, {num_labels}, H, W], got {logits.shape}"
                )
            size = model_outputs["target_size"].tolist()[0]
            logits_reshaped = torchvision.transforms.Resize(size)(logits)
            classes = logits_reshaped.argmax(dim=1)[0]
            annotation = []
            for label_id in range(num_labels):
                label = self.model.config.id2label[label_id]
                mask = classes == label_id
                score = ((mask * logits_reshaped[0, label_id]).sum() / mask.sum()).exp().item()
                mask = Image.fromarray(mask.numpy().astype(np.int8), mode="L")
                if not raw_image:
                    mask = self._get_mask_str(mask)
                annotation.append({"score": score, "label": label, "mask": mask})

        return annotation

    def _get_mask_str(self, img: "Image") -> str:
        """
        Turns mask numpy array into mask base64 str.

        Args:
            image (`PIL.Image`): PIL.Image (with shape (heigth, width) of the original image) containing masks
                information. Values in the array are either 0 or 255 (i.e. mask is absent VS mask is present).

        Returns:
            A base64 string of a single-channel PNG image that contain masks information.
        """
        with io.BytesIO() as out:
            img.save(out, format="PNG")
            png_string = out.getvalue()
            return base64.b64encode(png_string).decode("utf-8")
