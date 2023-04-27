from typing import Any, Dict, List, Union

import numpy as np

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    from ..models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
        MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
        MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
        MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
    )


logger = logging.get_logger(__name__)


Prediction = Dict[str, Any]
Predictions = List[Prediction]


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
    their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
    >>> segments = segmenter("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    >>> len(segments)
    2

    >>> segments[0]["label"]
    'bird'

    >>> segments[1]["label"]
    'bird'

    >>> type(segments[0]["mask"])  # This is a black and white mask showing where is the bird on the original image.
    <class 'PIL.Image.Image'>

    >>> segments[0]["mask"].size
    (768, 512)
    ```


    This image segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
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
            dict(
                MODEL_FOR_IMAGE_SEGMENTATION_MAPPING.items()
                + MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING.items()
                + MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING.items()
                + MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING.items()
            )
        )

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        if "subtask" in kwargs:
            postprocess_kwargs["subtask"] = kwargs["subtask"]
            preprocess_kwargs["subtask"] = kwargs["subtask"]
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "mask_threshold" in kwargs:
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]
        if "overlap_mask_area_threshold" in kwargs:
            postprocess_kwargs["overlap_mask_area_threshold"] = kwargs["overlap_mask_area_threshold"]

        return preprocess_kwargs, {}, postprocess_kwargs

    def __call__(self, images, **kwargs) -> Union[Predictions, List[Prediction]]:
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
            subtask (`str`, *optional*):
                Segmentation task to be performed, choose [`semantic`, `instance` and `panoptic`] depending on model
                capabilities. If not set, the pipeline will attempt tp resolve in the following order:
                  `panoptic`, `instance`, `semantic`.
            threshold (`float`, *optional*, defaults to 0.9):
                Probability threshold to filter out predicted masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.5):
                Mask overlap threshold to eliminate small, disconnected segments.

        Return:
            A dictionary or a list of dictionaries containing the result. If the input is a single image, will return a
            list of dictionaries, if the input is a list of several images, will return a list of list of dictionaries
            corresponding to each image.

            The dictionaries contain the mask, label and score (where applicable) of each detected object and contains
            the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **mask** (`PIL.Image`) -- A binary mask of the detected object as a Pil Image of shape (width, height) of
              the original image. Returns a mask filled with zeros if no object is found.
            - **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of the
              "object" described by the label and the mask.
        """
        return super().__call__(images, **kwargs)

    def preprocess(self, image, subtask=None):
        image = load_image(image)
        target_size = [(image.height, image.width)]
        if self.model.config.__class__.__name__ == "OneFormerConfig":
            if subtask is None:
                kwargs = {}
            else:
                kwargs = {"task_inputs": [subtask]}
            inputs = self.image_processor(images=[image], return_tensors="pt", **kwargs)
            inputs["task_inputs"] = self.tokenizer(
                inputs["task_inputs"],
                padding="max_length",
                max_length=self.model.config.task_seq_len,
                return_tensors=self.framework,
            )["input_ids"]
        else:
            inputs = self.image_processor(images=[image], return_tensors="pt")
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        model_outputs = self.model(**model_inputs)
        model_outputs["target_size"] = target_size
        return model_outputs

    def postprocess(
        self, model_outputs, subtask=None, threshold=0.9, mask_threshold=0.5, overlap_mask_area_threshold=0.5
    ):
        fn = None
        if subtask in {"panoptic", None} and hasattr(self.image_processor, "post_process_panoptic_segmentation"):
            fn = self.image_processor.post_process_panoptic_segmentation
        elif subtask in {"instance", None} and hasattr(self.image_processor, "post_process_instance_segmentation"):
            fn = self.image_processor.post_process_instance_segmentation

        if fn is not None:
            outputs = fn(
                model_outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                target_sizes=model_outputs["target_size"],
            )[0]

            annotation = []
            segmentation = outputs["segmentation"]

            for segment in outputs["segments_info"]:
                mask = (segmentation == segment["id"]) * 255
                mask = Image.fromarray(mask.numpy().astype(np.uint8), mode="L")
                label = self.model.config.id2label[segment["label_id"]]
                score = segment["score"]
                annotation.append({"score": score, "label": label, "mask": mask})

        elif subtask in {"semantic", None} and hasattr(self.image_processor, "post_process_semantic_segmentation"):
            outputs = self.image_processor.post_process_semantic_segmentation(
                model_outputs, target_sizes=model_outputs["target_size"]
            )[0]

            annotation = []
            segmentation = outputs.numpy()
            labels = np.unique(segmentation)

            for label in labels:
                mask = (segmentation == label) * 255
                mask = Image.fromarray(mask.astype(np.uint8), mode="L")
                label = self.model.config.id2label[label]
                annotation.append({"score": None, "label": label, "mask": mask})
        else:
            raise ValueError(f"Subtask {subtask} is not supported for model {type(self.model)}")
        return annotation
