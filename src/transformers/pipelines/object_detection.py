from typing import TYPE_CHECKING, Any, Literal, Union, overload

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args


if is_vision_available():
    from ..image_utils import load_image


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import (
        MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    )

if TYPE_CHECKING:
    from PIL import Image

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ObjectDetectionPipeline(Pipeline):
    """
    Object detection pipeline using any `AutoModelForObjectDetection`. This pipeline predicts bounding boxes of objects
    and their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> detector = pipeline(model="facebook/detr-resnet-50")
    >>> detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}, {'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}]

    >>> # Results are sorted by score descending. x, y are expressed relative to the top left hand corner.
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"object-detection"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=object-detection).
    """

    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        requires_backends(self, "vision")
        mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)
        self.check_model_type(mapping)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]

        postprocess_kwargs = {}
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "top_k" in kwargs:
            postprocess_kwargs["top_k"] = kwargs["top_k"]
        if "labels" in kwargs:
            postprocess_kwargs["labels"] = kwargs["labels"]
        if "box_format" in kwargs:
            postprocess_kwargs["box_format"] = kwargs["box_format"]

        return preprocess_params, {}, postprocess_kwargs

    @overload
    def __call__(self, image: Union[str, "Image.Image"], *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...

    @overload
    def __call__(
        self, image: list[str] | list["Image.Image"], *args: Any, **kwargs: Any
    ) -> list[list[dict[str, Any]]]: ...

    def __call__(self, *args, **kwargs) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
        """
        Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

        Args:
            inputs (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            threshold (`float`, *optional*, defaults to 0.5):
                The probability necessary to make a prediction.
            top_k (`int`, *optional*, defaults to `None`):
                The number of top detections to return, sorted by descending confidence score. If `None` or higher
                than the total number of detections above `threshold`, all qualifying detections are returned.
            labels (`list[str]`, *optional*, defaults to `None`):
                A list of class-label strings to keep. Only detections whose label appears in this list are
                returned. If `None`, all detected classes are returned.
            box_format (`str`, *optional*, defaults to `"xyxy"`):
                The coordinate format for returned bounding boxes. Accepted values:

                - `"xyxy"`: Returns `{"xmin": int, "ymin": int, "xmax": int, "ymax": int}` in pixel coordinates
                  (default, fully backward-compatible).
                - `"xywh"`: Returns `{"x_center": int, "y_center": int, "width": int, "height": int}` in pixels.
                - `"normalized"`: Returns `{"xmin": float, "ymin": float, "xmax": float, "ymax": float}` as
                  values in `[0, 1]` relative to the image dimensions.
            timeout (`float`, *optional*, defaults to `None`):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A list of dictionaries or a list of list of dictionaries containing the result. If the input is a single
            image, will return a list of dictionaries, if the input is a list of several images, will return a list of
            list of dictionaries corresponding to each image.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **score** (`float`) -- The score attributed by the model for that label.
            - **box** (`dict`) -- The bounding box of detected object. Format depends on the `box_format` argument.
        """
        # After deprecation of this is completed, remove the default `None` value for `images`
        if "images" in kwargs and "inputs" not in kwargs:
            kwargs["inputs"] = kwargs.pop("images")
        return super().__call__(*args, **kwargs)

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout=timeout)
        target_size = torch.IntTensor([[image.height, image.width]])
        inputs = self.image_processor(images=[image], return_tensors="pt")
        inputs = inputs.to(self.dtype)
        if self.tokenizer is not None:
            inputs = self.tokenizer(text=inputs["words"], boxes=inputs["boxes"], return_tensors="pt")
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        outputs = self.model(**model_inputs)
        model_outputs = outputs.__class__({"target_size": target_size, **outputs})
        if self.tokenizer is not None:
            model_outputs["bbox"] = model_inputs["bbox"]
        return model_outputs

    def postprocess(
        self,
        model_outputs,
        threshold: float = 0.5,
        top_k: int | None = None,
        labels: list[str] | None = None,
        box_format: Literal["xyxy", "xywh", "normalized"] = "xyxy",
    ):
        target_size = model_outputs["target_size"]
        if self.tokenizer is not None:
            # This is a LayoutLMForTokenClassification variant.
            # The OCR got the boxes and the model classified the words.
            height, width = target_size[0].tolist()

            def unnormalize(bbox):
                return self._get_bounding_box(
                    torch.Tensor(
                        [
                            (width * bbox[0] / 1000),
                            (height * bbox[1] / 1000),
                            (width * bbox[2] / 1000),
                            (height * bbox[3] / 1000),
                        ]
                    ),
                    box_format=box_format,
                    image_size=(height, width),
                )

            scores, classes = model_outputs["logits"].squeeze(0).softmax(dim=-1).max(dim=-1)
            label_names = [self.model.config.id2label[prediction] for prediction in classes.tolist()]
            boxes = [unnormalize(bbox) for bbox in model_outputs["bbox"].squeeze(0)]
            keys = ["score", "label", "box"]
            annotation = [
                dict(zip(keys, vals))
                for vals in zip(scores.tolist(), label_names, boxes)
                if vals[0] > threshold
            ]
        else:
            # This is a regular ForObjectDetectionModel
            height, width = target_size[0].tolist()
            raw_annotations = self.image_processor.post_process_object_detection(model_outputs, threshold, target_size)
            raw_annotation = raw_annotations[0]

            raw_annotation["scores"] = raw_annotation["scores"].tolist()
            raw_annotation["labels"] = [
                self.model.config.id2label[label.item()] for label in raw_annotation["labels"]
            ]
            raw_annotation["boxes"] = [
                self._get_bounding_box(box, box_format=box_format, image_size=(height, width))
                for box in raw_annotation["boxes"]
            ]

            # {"scores": [...], ...} --> [{"score": x, ...}, ...]
            keys = ["score", "label", "box"]
            annotation = [
                dict(zip(keys, vals))
                for vals in zip(
                    raw_annotation["scores"],
                    raw_annotation["labels"],
                    raw_annotation["boxes"],
                )
            ]

        # Sort by score descending (consistent with ZeroShotObjectDetectionPipeline
        # and ImageClassificationPipeline)
        annotation = sorted(annotation, key=lambda x: x["score"], reverse=True)

        # Filter to label allowlist if provided
        if labels is not None:
            annotation = [ann for ann in annotation if ann["label"] in labels]

        # Truncate to top_k highest-confidence detections
        if top_k is not None:
            annotation = annotation[:top_k]

        return annotation

    def _get_bounding_box(
        self,
        box: "torch.Tensor",
        box_format: Literal["xyxy", "xywh", "normalized"] = "xyxy",
        image_size: tuple[int, int] | None = None,
    ) -> dict:
        """
        Converts a bounding-box tensor into a dictionary using the requested coordinate format.

        Args:
            box (`torch.Tensor`):
                Tensor of shape `(4,)` with coordinates in `[xmin, ymin, xmax, ymax]` pixel format.
            box_format (`str`, *optional*, defaults to `"xyxy"`):
                Output format. One of `"xyxy"`, `"xywh"`, or `"normalized"`.
            image_size (`tuple[int, int]`, *optional*):
                `(height, width)` of the original image. Required when `box_format="normalized"`.

        Returns:
            `dict`: Bounding box in the requested format.
        """
        xmin, ymin, xmax, ymax = box.int().tolist()

        if box_format == "xyxy":
            return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

        elif box_format == "xywh":
            return {
                "x_center": (xmin + xmax) // 2,
                "y_center": (ymin + ymax) // 2,
                "width": xmax - xmin,
                "height": ymax - ymin,
            }

        elif box_format == "normalized":
            if image_size is None:
                raise ValueError("`image_size` must be provided when `box_format='normalized'`.")
            height, width = image_size
            return {
                "xmin": xmin / width,
                "ymin": ymin / height,
                "xmax": xmax / width,
                "ymax": ymax / height,
            }

        else:
            raise ValueError(
                f"Invalid `box_format` '{box_format}'. Choose one of 'xyxy', 'xywh', or 'normalized'."
            )
