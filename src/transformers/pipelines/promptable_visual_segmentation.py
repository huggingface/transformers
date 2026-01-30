# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Union, overload

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image, valid_images

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_PROMPTABLE_VISUAL_SEGMENTATION_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_processor=True))
class PromptableVisualSegmentationPipeline(Pipeline):
    """
    Promptable Visual Segmentation pipeline using SAM-family models. This pipeline predicts segmentation masks
    for objects when you provide an image and visual prompts. Visual prompts can be points (with positive/negative
    labels) or bounding boxes.

    This task is supported by models: Sam3TrackerModel, Sam2Model, SamModel, and EdgeTamModel.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> segmenter = pipeline(model="facebook/sam2.1-hiera-large", task="promptable-visual-segmentation")
    >>> # Single point prompt
    >>> segmenter(
    ...     "http://images.cocodataset.org/val2017/000000077595.jpg",
    ...     input_points=[[[[450, 600]]]],
    ...     input_labels=[[[1]]],
    ... )
    [[{'score': 0.87, 'mask': tensor([...])}]]

    >>> # Box prompt
    >>> segmenter(
    ...     "http://images.cocodataset.org/val2017/000000136466.jpg",
    ...     input_boxes=[[[59, 144, 76, 163]]],
    ... )
    [[{'score': 0.92, 'mask': tensor([...])}]]

    >>> # Multiple points for refinement (positive and negative)
    >>> segmenter(
    ...     "http://images.cocodataset.org/val2017/000000136466.jpg",
    ...     input_points=[[[[450, 600], [500, 620]]]],
    ...     input_labels=[[[1, 0]]],  # 1=positive (include), 0=negative (exclude)
    ... )
    [[{'score': 0.85, 'mask': tensor([...])}]]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This promptable visual segmentation pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"promptable-visual-segmentation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=promptable-visual-segmentation).
    """

    _load_processor = True
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_PROMPTABLE_VISUAL_SEGMENTATION_MAPPING_NAMES)

        # Handle processor compatibility: Sam3VideoProcessor → Sam3TrackerProcessor
        # facebook/sam3 checkpoint loads Sam3VideoProcessor by default, but this pipeline needs Sam3TrackerProcessor
        if self.processor is not None and self.processor.__class__.__name__ == "Sam3VideoProcessor":
            from ..models.sam3_tracker import Sam3TrackerProcessor

            # Get checkpoint name from model (empty string if instantiated from config, so use 'or' for fallback)
            model_name = getattr(self.model, "name_or_path", "") or "facebook/sam3"
            self.processor = Sam3TrackerProcessor.from_pretrained(model_name)

        # Determine if using SamProcessor (needs reshaped_input_sizes in post_process_masks)
        self._needs_reshaped_sizes = self.processor.__class__.__name__ == "SamProcessor"

    @overload
    def __call__(
        self,
        image: Union[str, "Image.Image"],
        input_points: list[list[list[list[float]]]] | None = None,
        input_labels: list[list[list[int]]] | None = None,
        input_boxes: list[list[list[float]]] | None = None,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]: ...

    @overload
    def __call__(self, image: list[dict[str, Any]], **kwargs: Any) -> list[list[dict[str, Any]]]: ...

    def __call__(
        self,
        image: Union[str, "Image.Image", list[dict[str, Any]]],
        input_points: list[list[list[list[float]]]] | None = None,
        input_labels: list[list[list[int]]] | None = None,
        input_boxes: list[list[list[float]]] | None = None,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """
        Segment objects in the image(s) based on visual prompts.

        Args:
            image (`str`, `PIL.Image`, or `list[dict[str, Any]]`):
                The pipeline handles three types of images:

                - A string containing an http url pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                You can use this parameter to send directly a list of images, or a dataset or a generator like so:

                ```python
                >>> from transformers import pipeline

                >>> segmenter = pipeline(model="facebook/sam2.1-hiera-large", task="promptable-visual-segmentation")
                >>> segmenter(
                ...     [
                ...         {
                ...             "image": "http://images.cocodataset.org/val2017/000000077595.jpg",
                ...             "input_points": [[[[450, 600]]]],
                ...             "input_labels": [[[1]]],
                ...         },
                ...         {
                ...             "image": "http://images.cocodataset.org/val2017/000000136466.jpg",
                ...             "input_boxes": [[[59, 144, 76, 163]]],
                ...         },
                ...     ]
                ... )
                [[{'score': 0.87, 'mask': ...}], [{'score': 0.92, 'mask': ...}]]
                ```

            input_points (`list[list[list[list[float]]]]`, *optional*):
                Point prompts in (x, y) format.
                Structure: [batch, objects, num_points, 2].
                Each point specifies a location on the image to guide segmentation.

            input_labels (`list[list[list[int]]]`, *optional*):
                Labels for the point prompts.
                Structure: [batch, objects, num_points].
                Values: 1 = positive (include in mask), 0 = negative (exclude from mask).
                Must match the structure of `input_points`.

            input_boxes (`list[list[list[float]]]`, *optional*):
                Bounding box prompts in xyxy format [x1, y1, x2, y2] in pixel coordinates.
                Structure: [batch, num_boxes, 4].

            multimask_output (`bool`, *optional*, defaults to False):
                Whether to output multiple mask candidates per prompt. When True, returns 3 masks per object
                ranked by IoU score. When False, returns only the best mask per object.

            mask_threshold (`float`, *optional*, defaults to 0.0):
                Threshold for binarizing the predicted masks.

            top_k (`int`, *optional*, defaults to None):
                The number of top predictions that will be returned by the pipeline. If the provided number is `None`
                or higher than the number of predictions available, it will default to the number of predictions.

            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A list of lists containing prediction results, one list per input image. Each list contains dictionaries
            with the following keys:

            - **score** (`float`) -- IoU confidence score for the predicted mask.
            - **mask** (`torch.Tensor`) -- Binary segmentation mask for the object, shape (height, width).
        """
        # Handle different input formats
        if isinstance(image, (str, Image.Image)):
            inputs = {
                "image": image,
                "input_points": input_points,
                "input_labels": input_labels,
                "input_boxes": input_boxes,
            }
        elif isinstance(image, (list, tuple)) and valid_images(image):
            # Batch of images - create individual inputs for each image
            batch_inputs = self._prepare_batch_inputs(image, input_points, input_labels, input_boxes)
            return list(super().__call__(batch_inputs, **kwargs))
        else:
            """
            Supports the following format
            - {"image": image, "input_points": points, "input_labels": labels}
            - [{"image": image, "input_points": points, "input_labels": labels}]
            - Generator and datasets
            """
            inputs = image

        results = super().__call__(inputs, **kwargs)
        return results

    def _prepare_batch_inputs(self, images, input_points, input_labels, input_boxes):
        """Helper method to prepare batch inputs from separate parameters."""
        # Expand single values to match batch size
        num_images = len(images)
        points_list = input_points if input_points is not None else [None] * num_images
        labels_list = input_labels if input_labels is not None else [None] * num_images
        boxes_list = input_boxes if input_boxes is not None else [None] * num_images

        # Create input dict for each image
        return (
            {
                "image": img,
                "input_points": points,
                "input_labels": labels,
                "input_boxes": boxes,
            }
            for img, points, labels, boxes in zip(images, points_list, labels_list, boxes_list)
        )

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]

        forward_params = {}
        if "multimask_output" in kwargs:
            forward_params["multimask_output"] = kwargs["multimask_output"]

        postprocess_params = {}
        if "mask_threshold" in kwargs:
            postprocess_params["mask_threshold"] = kwargs["mask_threshold"]
        if "top_k" in kwargs:
            postprocess_params["top_k"] = kwargs["top_k"]

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs, timeout=None):
        """
        Preprocess inputs for the model.

        Args:
            inputs: Dictionary containing 'image' and optionally 'input_points', 'input_labels', 'input_boxes'
            timeout: Timeout for image loading

        Returns:
            Dictionary with preprocessed model inputs
        """
        image = load_image(inputs["image"], timeout=timeout)
        input_points = inputs.get("input_points")
        input_labels = inputs.get("input_labels")
        input_boxes = inputs.get("input_boxes")

        # Validate that at least one prompt type is provided
        if input_points is None and input_boxes is None:
            raise ValueError(
                "You must provide at least one prompt type: either 'input_points' (with 'input_labels') or 'input_boxes'. "
                "For example: input_points=[[[[450, 600]]]], input_labels=[[[1]]] or input_boxes=[[[100, 150, 200, 250]]]"
            )

        # Validate that if input_points is provided, input_labels must also be provided
        if input_points is not None and input_labels is None:
            raise ValueError("When providing 'input_points', you must also provide 'input_labels'.")

        # Process inputs - pass all prompts as explicit parameters
        processor_kwargs = {
            "images": image,
            "return_tensors": "pt",
        }

        if input_points is not None:
            processor_kwargs["input_points"] = input_points
            processor_kwargs["input_labels"] = input_labels

        if input_boxes is not None:
            processor_kwargs["input_boxes"] = input_boxes

        model_inputs = self.processor(**processor_kwargs)
        model_inputs = model_inputs.to(self.dtype)

        # Store original size for post-processing
        target_size = torch.tensor([[image.height, image.width]], dtype=torch.int32)
        model_inputs["original_sizes"] = target_size

        # For SamProcessor, we also need to store reshaped_input_sizes
        if self._needs_reshaped_sizes and "reshaped_input_sizes" in model_inputs:
            model_inputs["_reshaped_input_sizes"] = model_inputs["reshaped_input_sizes"]

        return model_inputs

    def _forward(self, model_inputs, multimask_output=False):
        """
        Forward pass through the model.

        Args:
            model_inputs: Preprocessed model inputs
            multimask_output: Whether to output multiple masks per prompt

        Returns:
            Model outputs with additional metadata
        """
        original_sizes = model_inputs.pop("original_sizes")
        reshaped_input_sizes = model_inputs.pop("_reshaped_input_sizes", None)

        outputs = self.model(**model_inputs, multimask_output=multimask_output)

        return {
            "outputs": outputs,
            "original_sizes": original_sizes,
            "reshaped_input_sizes": reshaped_input_sizes,
        }

    def postprocess(self, model_outputs, mask_threshold=0.0, top_k=None):
        """
        Post-process model outputs into final predictions.

        Args:
            model_outputs: Raw model outputs
            mask_threshold: Threshold for binarizing masks
            top_k: Maximum number of predictions to return per image

        Returns:
            List of lists of dictionaries with 'score' and 'mask' keys
        """
        outputs = model_outputs["outputs"]
        original_sizes = model_outputs["original_sizes"]
        reshaped_input_sizes = model_outputs["reshaped_input_sizes"]

        # Get masks and IoU scores from outputs
        pred_masks = outputs.pred_masks  # (batch, objects, num_masks, H, W)
        iou_scores = outputs.iou_scores  # (batch, objects, num_masks)

        # Post-process masks to original image size
        post_process_kwargs = {
            "masks": pred_masks.cpu(),
            "original_sizes": original_sizes.tolist(),
            "mask_threshold": mask_threshold,
            "binarize": True,
        }

        # For SamProcessor, we need to pass reshaped_input_sizes
        if self._needs_reshaped_sizes and reshaped_input_sizes is not None:
            post_process_kwargs["reshaped_input_sizes"] = reshaped_input_sizes.tolist()

        masks = self.processor.post_process_masks(**post_process_kwargs)

        # Format output as per-image list of dictionaries
        final_results = []
        batch_size = pred_masks.shape[0]

        for batch_idx in range(batch_size):
            image_results = []
            num_objects = pred_masks.shape[1]
            num_masks_per_object = pred_masks.shape[2]

            for obj_idx in range(num_objects):
                for mask_idx in range(num_masks_per_object):
                    score = iou_scores[batch_idx, obj_idx, mask_idx].item()
                    mask_tensor = masks[batch_idx][obj_idx, mask_idx]

                    result = {
                        "score": score,
                        "mask": mask_tensor,
                    }
                    image_results.append(result)

            # Sort results by score in descending order
            image_results = sorted(image_results, key=lambda x: x["score"], reverse=True)

            # Apply top_k filtering
            if top_k is not None and len(image_results) > top_k:
                image_results = image_results[:top_k]

            final_results.append(image_results)

        # If single image, return as list with one element (for consistency)
        return final_results if batch_size > 1 or isinstance(pred_masks, (list, tuple)) else final_results
