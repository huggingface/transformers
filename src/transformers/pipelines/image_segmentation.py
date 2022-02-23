from typing import Any, Dict, List, Union

import numpy as np

from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available, requires_backends
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch
    from torch import nn

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
            threshold (`float`, *optional*, defaults to 0.9):
                The probability necessary to make a prediction.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.

        Return:
            A dictionary or a list of dictionaries containing the result. If the input is a single image, will return a
            list of dictionaries, if the input is a list of several images, will return a list of list of dictionaries
            corresponding to each image.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **mask** (`PIL.Image`) -- Pil Image with size (heigth, width) of the original image. Pixel values in the
              image are in the range 0-255. 0 means the pixel is *not* part of the *label*, 255 means it definitely is.
            - **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of the
              "object" described by the label and the mask.
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
                Image.fromarray(mask.numpy().astype(np.uint8), mode="L") for mask in raw_annotation["masks"]
            ]
            # {"scores": [...], ...} --> [{"score":x, ...}, ...]
            keys = ["score", "label", "mask"]
            annotation = [
                dict(zip(keys, vals))
                for vals in zip(raw_annotation["scores"], raw_annotation["labels"], raw_annotation["masks"])
            ]
        else:
            # Default logits
            logits = model_outputs.logits
            logits = logits.softmax(dim=1)
            if len(logits.shape) != 4:
                raise ValueError(f"Logits don't have expected dimensions, expected [1, N, H, W], got {logits.shape}")
            batch_size, num_labels, height, width = logits.shape
            expected_num_labels = len(self.model.config.id2label)
            if num_labels != expected_num_labels:
                raise ValueError(
                    f"Logits don't have expected dimensions, expected [1, {num_labels}, H, W], got {logits.shape}"
                )
            size = model_outputs["target_size"].squeeze(0).tolist()
            logits_reshaped = nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)
            classes = logits_reshaped.argmax(dim=1)[0]
            annotation = []

            for label_id in range(num_labels):
                label = self.model.config.id2label[label_id]
                mask = classes == label_id
                mask_sum = mask.sum()

                # Remove empty masks.
                if mask_sum == 0:
                    continue
                mask = Image.fromarray((mask * 255).numpy().astype(np.uint8), mode="L")
                # Semantic segmentation does not output a global score for the mask
                # so we don't attempt to compute one.
                # XXX: We could send a mask with values between 0 and 255 instead
                # of a pure mask to enable users to get the probabilities that
                # are really outputted by the logits.
                annotation.append({"score": None, "label": label, "mask": mask})
        return annotation
