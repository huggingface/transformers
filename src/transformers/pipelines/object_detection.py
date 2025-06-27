from typing import TYPE_CHECKING, Any, Union, overload

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
    [{'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]

    >>> # x, y  are expressed relative to the top left hand corner.
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

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

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
        return preprocess_params, {}, postprocess_kwargs

    @overload
    def __call__(self, image: Union[str, "Image.Image"], *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...

    @overload
    def __call__(
        self, image: Union[list[str], list["Image.Image"]], *args: Any, **kwargs: Any
    ) -> list[list[dict[str, Any]]]: ...

    def __call__(self, *args, **kwargs) -> Union[list[dict[str, Any]], list[list[dict[str, Any]]]]:
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
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A list of dictionaries or a list of list of dictionaries containing the result. If the input is a single
            image, will return a list of dictionaries, if the input is a list of several images, will return a list of
            list of dictionaries corresponding to each image.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **score** (`float`) -- The score attributed by the model for that label.
            - **box** (`list[dict[str, int]]`) -- The bounding box of detected object in image's original size.
        """
        # After deprecation of this is completed, remove the default `None` value for `images`
        if "images" in kwargs and "inputs" not in kwargs:
            kwargs["inputs"] = kwargs.pop("images")
        return super().__call__(*args, **kwargs)

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout=timeout)
        target_size = torch.IntTensor([[image.height, image.width]])
        inputs = self.image_processor(images=[image], return_tensors="pt")
        if self.framework == "pt":
            inputs = inputs.to(self.torch_dtype)
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

    def postprocess(self, model_outputs, threshold=0.5):
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
                    )
                )

            scores, classes = model_outputs["logits"].squeeze(0).softmax(dim=-1).max(dim=-1)
            labels = [self.model.config.id2label[prediction] for prediction in classes.tolist()]
            boxes = [unnormalize(bbox) for bbox in model_outputs["bbox"].squeeze(0)]
            keys = ["score", "label", "box"]
            annotation = [dict(zip(keys, vals)) for vals in zip(scores.tolist(), labels, boxes) if vals[0] > threshold]
        else:
            # This is a regular ForObjectDetectionModel
            raw_annotations = self.image_processor.post_process_object_detection(model_outputs, threshold, target_size)
            raw_annotation = raw_annotations[0]
            scores = raw_annotation["scores"]
            labels = raw_annotation["labels"]
            boxes = raw_annotation["boxes"]

            raw_annotation["scores"] = scores.tolist()
            raw_annotation["labels"] = [self.model.config.id2label[label.item()] for label in labels]
            raw_annotation["boxes"] = [self._get_bounding_box(box) for box in boxes]

            # {"scores": [...], ...} --> [{"score":x, ...}, ...]
            keys = ["score", "label", "box"]
            annotation = [
                dict(zip(keys, vals))
                for vals in zip(raw_annotation["scores"], raw_annotation["labels"], raw_annotation["boxes"])
            ]

        return annotation

    def _get_bounding_box(self, box: "torch.Tensor") -> dict[str, int]:
        """
        Turns list [xmin, xmax, ymin, ymax] into dict { "xmin": xmin, ... }

        Args:
            box (`torch.Tensor`): Tensor containing the coordinates in corners format.

        Returns:
            bbox (`dict[str, int]`): Dict containing the coordinates in corners format.
        """
        if self.framework != "pt":
            raise ValueError("The ObjectDetectionPipeline is only available in PyTorch.")
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox
