from typing import Any, Dict, List, Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, ChunkPipeline
import numpy as np

if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    from ..models.auto.modeling_auto import (
        MODEL_FOR_ZERO_SHOT_IMAGE_SEGMENTATION_MAPPING,
    )

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotSegmentationPipeline(ChunkPipeline):
    """
    Zero shot segmentation pipeline using `SamForSegmentation`. This pipeline predicts segmentation masks for an image,
    given an image or potentially additional inputs.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> detector = pipeline(model="facebook/sam-vit-h", task="zero-shot-object-detection")
    >>> detector(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ...     candidate_labels=["cat", "couch"],
    ... )
    [{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]

    >>> detector(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     candidate_labels=["head", "bird"],
    ... )
    [{'score': 0.119, 'label': 'bird', 'box': {'xmin': 71, 'ymin': 170, 'xmax': 410, 'ymax': 508}}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-object-detection"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.framework == "tf":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_ZERO_SHOT_IMAGE_SEGMENTATION_MAPPING)

    def _sanitize_parameters(self, **kwargs):
        preprocessor_kwargs = {}
        postprocess_kwargs = {}
        if "subtask" in kwargs:
            postprocess_kwargs["subtask"] = kwargs["subtask"]
            preprocessor_kwargs["subtask"] = kwargs["subtask"]
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "mask_threshold" in kwargs:
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]
        if "overlap_mask_area_threshold" in kwargs:
            postprocess_kwargs["overlap_mask_area_threshold"] = kwargs["overlap_mask_area_threshold"]

        return preprocessor_kwargs, {}, postprocess_kwargs

    def __call__(
        self,
        images: Union[str, "Image.Image", List[Dict[str, Any]]],
        segmentation_prompts: Union[str, List[str]] = None,
        **kwargs,
    ):
        """
        Perform zero-shot image segmentation (detect masks & classes) in the image(s) passed as inputs.

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
        """

        if isinstance(images, (str, Image.Image)):
            inputs = {"image": images, "segmentation_prompts": segmentation_prompts}
        else:
            inputs = images
        results = super().__call__(inputs, **kwargs)
        return results

    def _sanitize_parameters(self, **kwargs):
        postprocess_params = {}
        if "threshold" in kwargs:
            postprocess_params["threshold"] = kwargs["threshold"]
        if "top_k" in kwargs:
            postprocess_params["top_k"] = kwargs["top_k"]
        return {}, {}, postprocess_params

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
        candidate_label = model_inputs.pop("candidate_label")
        is_last = model_inputs.pop("is_last")

        outputs = self.model(**model_inputs)

        model_outputs = {"target_size": target_size, "candidate_label": candidate_label, "is_last": is_last, **outputs}
        return model_outputs

    def postprocess(self, model_outputs, threshold=0.1, top_k=None):
        results = []
        for model_output in model_outputs:
            label = model_output["candidate_label"]
            model_output = BaseModelOutput(model_output)
            outputs = self.image_processor.post_process_object_detection(
                outputs=model_output, threshold=threshold, target_sizes=model_output["target_size"]
            )[0]

            for index in outputs["scores"].nonzero():
                score = outputs["scores"][index].item()
                box = self._get_bounding_box(outputs["boxes"][index][0])

                result = {"score": score, "label": label, "box": box}
                results.append(result)

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        if top_k:
            results = results[:top_k]

        return results

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
