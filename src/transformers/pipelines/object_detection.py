from typing import TYPE_CHECKING, Any, Optional, Union, overload

from ..models.auto import AutoImageProcessor
from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args


if is_vision_available():
    from ..image_utils import load_image


if is_torch_available():
    import torch
    from torch.export import Dim

    from ..models.auto.modeling_auto import (
        MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    )
    from .exportable import (
        ExportableModule,
        export_pipeline_to_onnx,
        export_pipeline_to_torch,
        export_pipeline_to_torchscript,
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
        xmin, ymin, xmax, ymax = box.int().tolist()
        bbox = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
        return bbox

    def get_exportable_module(
        self,
        include_preprocessing: bool = True,
        include_postprocessing: bool = True,
    ) -> "ExportableModule":
        """
        Get an exportable version of this pipeline that can be exported to ONNX or torch.export.

        The exportable module bundles preprocessing and postprocessing into the model's forward pass,
        allowing deployment without Python dependencies.

        Args:
            include_preprocessing (`bool`, *optional*, defaults to `True`):
                Whether to include preprocessing in the exported model. If False, the exported model
                will expect preprocessed `pixel_values` as input.
            include_postprocessing (`bool`, *optional*, defaults to `True`):
                Whether to include postprocessing in the exported model. If False, the exported model
                will return raw model outputs instead of formatted detections.

        Returns:
            `ExportableModule`: A torch.nn.Module that wraps the pipeline for export.

        Example:
            ```python
            >>> from transformers import pipeline
            >>> import torch

            >>> pipe = pipeline("object-detection", model="facebook/detr-resnet-50")
            >>> exportable = pipe.get_exportable_module()

            >>> # Create example inputs
            >>> from PIL import Image
            >>> import requests
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> images = exportable.get_tensors_inputs(image)
            >>> example_inputs = {
            ...     "images": images,
            ...     "post_process_kwargs": {"target_sizes": torch.tensor([[image.height, image.width]])}
            ... }

            >>> # Export with torch.export
            >>> exported_program = torch.export.export(exportable, args=(), kwargs=example_inputs, strict=False)
            ```
        """
        # If the current processor is slow, reload as fast
        # TODO @yoni: remove this once we load the fast processor by default
        processor = self.image_processor

        # Check if processor supports the needed features for two-stage preprocessing
        if not processor.is_fast:
            logger.warning(
                "Current image processor does not support two-stage preprocessing (slow processor). "
                "Reloading with use_fast=True for export compatibility."
            )
            processor = AutoImageProcessor.from_pretrained(self.model.config._name_or_path, use_fast=True)

        return ExportableModule(
            model=self.model,
            processor=processor,
            post_process_function_name="post_process_object_detection",
            include_preprocessing=include_preprocessing,
            include_postprocessing=include_postprocessing,
        )

    def export(
        self,
        example_image: Union[str, "Image.Image"],
        format: str = "torch",
        save_path: Optional[str] = None,
        dynamic_shapes: Union[dict, bool] = True,
        include_preprocessing: bool = True,
        include_postprocessing: bool = True,
        optimize: bool = True,
        threshold: float = 0.5,
        **export_kwargs,
    ) -> Union["torch.export.ExportedProgram", "torch.jit.ScriptModule", str]:
        """
        Export the pipeline to a specified format (torch.export, ONNX, or TorchScript).

        This method creates an exportable module and exports it in the specified format,
        bundling preprocessing and postprocessing into a single artifact for deployment.

        Args:
            example_image (`str` or `PIL.Image`):
                An example image used for tracing the model. Can be a path, URL, or PIL Image.
            format (`str`, *optional*, defaults to `"torch"`):
                Export format. One of:
                - `"torch"`: Export using torch.export (recommended for PyTorch 2.0+)
                - `"onnx"`: Export to ONNX format
                - `"torchscript"`: Export to TorchScript
            save_path (`str`, *optional*):
                Path to save the exported model. Required for ONNX export.
            dynamic_shapes (`dict` or `bool`, *optional*, defaults to `True`):
                Whether to use dynamic shapes for export. If `True`, uses default dynamic shapes
                allowing variable image sizes. If `dict`, uses the provided configuration.
            include_preprocessing (`bool`, *optional*, defaults to `True`):
                Whether to include preprocessing in the exported model.
            include_postprocessing (`bool`, *optional*, defaults to `True`):
                Whether to include postprocessing in the exported model.
            optimize (`bool`, *optional*, defaults to `True`):
                Whether to optimize the exported model (ONNX only).
            threshold (`float`, *optional*, defaults to `0.5`):
                The score threshold used for filtering detections in postprocessing.
                This value will be used during export and should match the threshold used during inference.
            **export_kwargs:
                Additional arguments passed to the export function.

        Returns:
            Depending on the format:
            - `"torch"`: Returns `torch.export.ExportedProgram`
            - `"onnx"`: Returns path to the saved ONNX file (str)
            - `"torchscript"`: Returns `torch.jit.ScriptModule`

        Note:
            For torch.export, the exported model has a **fixed input signature**. The exported model
            will include `post_process_kwargs` with both `target_sizes` and `threshold` parameters.
            When using the exported model, you must provide the same structure:

            ```python
            outputs = exported_model(
                images=images,
                post_process_kwargs={"target_sizes": sizes, "threshold": 0.5}
            )
            ```

        Example:
            ```python
            >>> from transformers import pipeline

            >>> pipe = pipeline("object-detection", model="facebook/detr-resnet-50")

            >>> # Export to torch.export with dynamic shapes
            >>> exported_program = pipe.export(
            ...     example_image="http://images.cocodataset.org/val2017/000000039769.jpg",
            ...     format="torch",
            ...     dynamic_shapes=True,
            ... )

            >>> # Export to ONNX
            >>> onnx_path = pipe.export(
            ...     example_image="http://images.cocodataset.org/val2017/000000039769.jpg",
            ...     format="onnx",
            ...     save_path="model.onnx",
            ...     dynamic_shapes=True,
            ...     optimize=True,
            ... )
            ```
        """
        example_image = load_image(example_image)

        # Create exportable module
        exportable_module = self.get_exportable_module(
            include_preprocessing=include_preprocessing,
            include_postprocessing=include_postprocessing,
        )

        # Prepare example inputs
        images = exportable_module.get_tensors_inputs(example_image, device=self.device)
        example_inputs = {
            "images": images.to(self.device),
            "post_process_kwargs": {
                "target_sizes": torch.tensor([[example_image.height, example_image.width]], device=self.device),
                "threshold": threshold,  # Use provided threshold parameter
            },
        }

        # Create dynamic shapes configuration if requested
        if dynamic_shapes is True:
            # Default dynamic shapes for object detection
            height_dim = Dim("height", min=32, max=4096)
            width_dim = Dim("width", min=32, max=4096)
            dynamic_shapes = {
                "images": {2: height_dim, 3: width_dim},
                "post_process_kwargs": {
                    "target_sizes": None,  # Dynamic: can change per image
                    "threshold": None,  # Static: scalar float, marked as None (no constraints)
                },
            }

        if format == "torch":
            return export_pipeline_to_torch(
                exportable_module,
                example_inputs,
                save_path=save_path,
                dynamic_shapes=dynamic_shapes,
                **export_kwargs,
            )
        elif format == "onnx":
            return export_pipeline_to_onnx(
                exportable_module,
                example_inputs,
                save_path=save_path,
                dynamic_shapes=dynamic_shapes,
                optimize=optimize,
                **export_kwargs,
            )
        elif format == "torchscript":
            return export_pipeline_to_torchscript(
                exportable_module, example_inputs, save_path=save_path, **export_kwargs
            )
        else:
            raise ValueError(f"Unsupported export format: {format}. Choose from: torch, onnx, torchscript")
