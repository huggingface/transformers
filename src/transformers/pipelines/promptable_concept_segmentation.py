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

    from ..models.auto.modeling_auto import MODEL_FOR_PROMPTABLE_CONCEPT_SEGMENTATION_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_processor=True))
class PromptableConceptSegmentationPipeline(Pipeline):
    """
    Promptable Concept Segmentation pipeline using `Sam3Model`. This pipeline predicts instance segmentation masks
    and bounding boxes for objects when you provide an image and prompts. Prompts can be text descriptions
    (e.g., "yellow school bus"), visual box exemplars (positive/negative), or combinations of both.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> segmenter = pipeline(model="facebook/sam3", task="promptable-concept-segmentation")
    >>> segmenter(
    ...     "http://images.cocodataset.org/val2017/000000077595.jpg",
    ...     text="ear",
    ... )
    [{'score': 0.87, 'box': {'xmin': 120, 'ymin': 45, 'xmax': 210, 'ymax': 130}, 'mask': tensor([...])}, ...]

    >>> # Using box prompts
    >>> segmenter(
    ...     "http://images.cocodataset.org/val2017/000000136466.jpg",
    ...     input_boxes=[[[59, 144, 76, 163], [87, 148, 104, 159]]],
    ...     input_boxes_labels=[[1, 1]],
    ... )
    [{'score': 0.92, 'box': {'xmin': 59, 'ymin': 144, 'xmax': 76, 'ymax': 163}, 'mask': tensor([...])}, ...]

    >>> # Combined text and negative box
    >>> segmenter(
    ...     "http://images.cocodataset.org/val2017/000000136466.jpg",
    ...     text="handle",
    ...     input_boxes=[[[40, 183, 318, 204]]],
    ...     input_boxes_labels=[[0]],  # 0 = negative (exclude this region)
    ... )
    [{'score': 0.85, 'box': {'xmin': 250, 'ymin': 100, 'xmax': 280, 'ymax': 150}, 'mask': tensor([...])}, ...]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This promptable concept segmentation pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"promptable-concept-segmentation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=promptable-concept-segmentation).
    """

    _load_processor = True
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_PROMPTABLE_CONCEPT_SEGMENTATION_MAPPING_NAMES)

        # Ensure we have Sam3Processor (not Sam3VideoProcessor) for text and box prompt support
        # facebook/sam3 checkpoint loads Sam3VideoProcessor by default, but this pipeline needs Sam3Processor
        if self.processor is not None and self.processor.__class__.__name__ == "Sam3VideoProcessor":
            from ..models.sam3 import Sam3Processor

            # Try to get the model checkpoint name
            model_name = getattr(self.model, "name_or_path", None)
            if not model_name and hasattr(self.model, "config"):
                model_name = getattr(self.model.config, "_name_or_path", None)

            # Default to facebook/sam3 if we can't determine the model name
            # (facebook/sam3 is the canonical checkpoint for this task)
            if not model_name:
                model_name = "facebook/sam3"

            logger.info(
                "Detected Sam3VideoProcessor but promptable-concept-segmentation requires Sam3Processor. "
                f"Loading Sam3Processor from {model_name}."
            )
            self.processor = Sam3Processor.from_pretrained(model_name)

    @overload
    def __call__(
        self,
        image: Union[str, "Image.Image"],
        text: str | None = None,
        input_boxes: list[list[list[float]]] | None = None,
        input_boxes_labels: list[list[int]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]: ...

    @overload
    def __call__(self, image: list[dict[str, Any]], **kwargs: Any) -> list[list[dict[str, Any]]]: ...

    def __call__(
        self,
        image: Union[str, "Image.Image", list[dict[str, Any]]],
        text: str | list[str] | None = None,
        input_boxes: list[list[list[float]]] | None = None,
        input_boxes_labels: list[list[int]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
        """
        Segment objects in the image(s) based on the provided prompts.

        Args:
            image (`str`, `PIL.Image`, or `list[dict[str, Any]]`):
                The pipeline handles three types of images:

                - A string containing an http url pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                You can use this parameter to send directly a list of images, or a dataset or a generator like so:

                ```python
                >>> from transformers import pipeline

                >>> segmenter = pipeline(model="facebook/sam3", task="promptable-concept-segmentation")
                >>> segmenter(
                ...     [
                ...         {
                ...             "image": "http://images.cocodataset.org/val2017/000000077595.jpg",
                ...             "text": "ear",
                ...         },
                ...         {
                ...             "image": "http://images.cocodataset.org/val2017/000000136466.jpg",
                ...             "text": "dial",
                ...         },
                ...     ]
                ... )
                [[{'score': 0.87, 'box': {...}, 'mask': ...}], [{'score': 0.92, 'box': {...}, 'mask': ...}]]
                ```

            text (`str` or `list[str]`, *optional*):
                Text prompt(s) describing the concept to segment (e.g., "yellow school bus", "ear", "handle").
                Can be a single string or a list of strings for batched inference.

            input_boxes (`list[list[list[float]]]`, *optional*):
                Visual box prompts in xyxy format [x1, y1, x2, y2] in pixel coordinates.
                Structure: [batch, num_boxes, 4]. Used to provide visual exemplars of the concept.

            input_boxes_labels (`list[list[int]]`, *optional*):
                Labels for the box prompts. 1 = positive (include), 0 = negative (exclude).
                Structure: [batch, num_boxes]. Must match the structure of `input_boxes`.

            threshold (`float`, *optional*, defaults to 0.3):
                The probability necessary to make a prediction.

            mask_threshold (`float`, *optional*, defaults to 0.5):
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

            - **score** (`float`) -- Confidence score for the detected instance.
            - **box** (`dict[str, int]`) -- Bounding box of the detected object in image's original size with keys
              `xmin`, `ymin`, `xmax`, `ymax`.
            - **mask** (`torch.Tensor`) -- Binary segmentation mask for the instance, shape (height, width).
        """
        # Handle different input formats
        if isinstance(image, str | Image.Image):
            inputs = {
                "image": image,
                "text": text,
                "input_boxes": input_boxes,
                "input_boxes_labels": input_boxes_labels,
            }
        elif isinstance(image, list | tuple) and valid_images(image):
            # Batch of images - create individual inputs for each image
            batch_inputs = self._prepare_batch_inputs(image, text, input_boxes, input_boxes_labels)
            return list(super().__call__(batch_inputs, **kwargs))
        else:
            """
            Supports the following format
            - {"image": image, "text": text}
            - [{"image": image, "text": text}]
            - Generator and datasets
            """
            inputs = image

        results = super().__call__(inputs, **kwargs)
        return results

    def _prepare_batch_inputs(self, images, text, input_boxes, input_boxes_labels):
        """Helper method to prepare batch inputs from separate parameters."""
        # Expand single values to match batch size
        num_images = len(images)
        text_list = text if isinstance(text, list) else [text] * num_images
        boxes_list = input_boxes if input_boxes is not None else [None] * num_images
        labels_list = input_boxes_labels if input_boxes_labels is not None else [None] * num_images

        # Create input dict for each image
        return (
            {
                "image": img,
                "text": txt,
                "input_boxes": boxes,
                "input_boxes_labels": box_labels,
            }
            for img, txt, boxes, box_labels in zip(images, text_list, boxes_list, labels_list)
        )

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]

        postprocess_params = {}
        if "threshold" in kwargs:
            postprocess_params["threshold"] = kwargs["threshold"]
        if "mask_threshold" in kwargs:
            postprocess_params["mask_threshold"] = kwargs["mask_threshold"]
        if "top_k" in kwargs:
            postprocess_params["top_k"] = kwargs["top_k"]

        return preprocess_params, {}, postprocess_params

    def _normalize_boxes_format(self, input_boxes):
        """Ensure input_boxes is in the correct format: [batch, num_boxes, 4]."""
        if input_boxes is None:
            return None
        if not isinstance(input_boxes, list):
            return [[input_boxes]]
        if len(input_boxes) > 0 and not isinstance(input_boxes[0], list):
            return [input_boxes]
        return input_boxes

    def _normalize_labels_format(self, input_boxes_labels):
        """Ensure input_boxes_labels is in the correct format: [batch, num_boxes]."""
        if input_boxes_labels is None:
            return None
        if not isinstance(input_boxes_labels, list):
            return [[input_boxes_labels]]
        if len(input_boxes_labels) > 0 and not isinstance(input_boxes_labels[0], list):
            return [input_boxes_labels]
        return input_boxes_labels

    def preprocess(self, inputs, timeout=None):
        """
        Preprocess inputs for the model.

        Args:
            inputs: Dictionary containing 'image' and optionally 'text', 'input_boxes', 'input_boxes_labels'
            timeout: Timeout for image loading

        Returns:
            Dictionary with preprocessed model inputs
        """
        image = load_image(inputs["image"], timeout=timeout)
        text = inputs.get("text")
        input_boxes = inputs.get("input_boxes")
        input_boxes_labels = inputs.get("input_boxes_labels")

        # Validate that at least one prompt type is provided
        if text is None and input_boxes is None:
            raise ValueError(
                "You must provide at least one prompt type: either 'text' or 'input_boxes'. "
                "For example: text='cat' or input_boxes=[[[100, 150, 200, 250]]]"
            )

        # Normalize box formats
        input_boxes = self._normalize_boxes_format(input_boxes)
        input_boxes_labels = self._normalize_labels_format(input_boxes_labels)

        # Process inputs - pass text, input_boxes, input_boxes_labels as explicit parameters
        model_inputs = self.processor(
            images=image,
            text=text,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(self.dtype)

        # Store original size for post-processing
        target_size = torch.tensor([[image.height, image.width]], dtype=torch.int32)
        model_inputs["target_size"] = target_size

        # Store the text prompt for output labeling
        model_inputs["prompt_text"] = text

        return model_inputs

    def _forward(self, model_inputs):
        """
        Forward pass through the model.

        Args:
            model_inputs: Preprocessed model inputs

        Returns:
            Model outputs with additional metadata
        """
        target_size = model_inputs.pop("target_size")
        prompt_text = model_inputs.pop("prompt_text")

        outputs = self.model(**model_inputs)

        return {
            "outputs": outputs,
            "target_size": target_size,
            "prompt_text": prompt_text,
        }

    def postprocess(self, model_outputs, threshold=0.3, mask_threshold=0.5, top_k=None):
        """
        Post-process model outputs into final predictions.

        Args:
            model_outputs: Raw model outputs
            threshold: Score threshold for filtering predictions
            mask_threshold: Threshold for binarizing masks
            top_k: Maximum number of predictions to return

        Returns:
            List of dictionaries with 'score', 'box', and 'mask' keys
        """
        outputs = model_outputs["outputs"]
        target_sizes = model_outputs["target_size"]
        prompt_text = model_outputs["prompt_text"]

        # Use processor's post-processing method
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes.tolist(),
        )[0]  # Get first batch element

        # Convert to expected output format
        final_results = []
        if len(results["scores"]) > 0:
            for i in range(len(results["scores"])):
                score = results["scores"][i].item()
                box_tensor = results["boxes"][i]
                mask_tensor = results["masks"][i]

                result = {
                    "score": score,
                    "box": self._get_bounding_box(box_tensor),
                    "mask": mask_tensor,
                }

                # Optionally add label if text prompt was provided
                if prompt_text is not None:
                    result["label"] = prompt_text

                final_results.append(result)

        # Sort results by score in descending order
        final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)

        # Apply top_k filtering
        if top_k is not None and len(final_results) > top_k:
            final_results = final_results[:top_k]

        return final_results

    def _get_bounding_box(self, box: "torch.Tensor") -> dict[str, int]:
        xmin, ymin, xmax, ymax = box.int().tolist()
        return {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
