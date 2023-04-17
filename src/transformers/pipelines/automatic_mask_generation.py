from ..image_utils import load_image, to_numpy_array
from ..models.sam.image_processing_sam import _filter_masks, _generate_crop_boxes, _postprocess_for_amg
from ..utils import (
    add_end_docstrings,
    is_torch_available,
    logging,
    requires_backends,
)
from .base import PIPELINE_INIT_ARGS, ChunkPipeline


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_AUTOMATIC_MASK_GENERATION_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class AutomaticMaskGenerationPipeline(ChunkPipeline):
    """
    Automatic mask generation for images using `SamForMaskGeneration`. This pipeline predicts binary masks for an
    image, given an image and potentially additional inputs such as points, bounding boxes or .

    The pipeline used the following functions:
        1. pre_process: calls the processor with image, points labels etc
            - generate_crop_boxes
        2. forward: feed the output of the processor to the model
            - get_image_embeddings
            - forward sequentually (loop)
        3. post_process:
            - postprocess_masks (loop)
            - filter_masks (loop)
            - postprocess_masks_for_amg

    Example:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="facebook/sam-vit-h", task="automatic-mask-generation")
    >>> generator(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ... )
    []

    >>> generator("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", point_batch_size=16)
    []
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"automatic-mask-generation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=automatic-mask-generation).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        requires_backends(self, "vision")
        requires_backends(self, "torch")

        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        self.check_model_type(MODEL_FOR_AUTOMATIC_MASK_GENERATION_MAPPING)

    def _sanitize_parameters(self, **kwargs):
        """ """
        preprocessor_kwargs = {}
        postprocess_kwargs = {}
        if "n_layers" in kwargs:
            postprocess_kwargs["n_layers"] = kwargs["n_layers"]
        if "overlap_ratio" in kwargs:
            postprocess_kwargs["overlap_ratio"] = kwargs["overlap_ratio"]
        if "scale_per_layer" in kwargs:
            postprocess_kwargs["scale_per_layer"] = kwargs["scale_per_layer"]
        if "pred_iou_thresh" in kwargs:
            postprocess_kwargs["pred_iou_thresh"] = kwargs["pred_iou_thresh"]
        if "stability_score_offset" in kwargs:
            postprocess_kwargs["stability_score_offset"] = kwargs["stability_score_offset"]
        if "box_nms_thresh" in kwargs:
            postprocess_kwargs["box_nms_thresh"] = kwargs["box_nms_thresh"]
        if "mask_threshold" in kwargs:
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]
        if "point_batch_size" in kwargs:
            preprocessor_kwargs["point_batch_size"] = kwargs["point_batch_size"]

        return preprocessor_kwargs, {}, postprocess_kwargs

    def __call__(self, image, *args, num_workers=None, batch_size=None, **kwargs):
        image = load_image(image)
        image = to_numpy_array(image)
        self._postprocess_params["image"] = image
        return super().__call__(image, *args, num_workers=num_workers, batch_size=batch_size, **kwargs)

    def preprocess(self, image, point_batch_size, **kwargs):
        """
        Since the pipeline inherits from `chunkPipeline` is meas that setting a `max_batch_points` allows the user to
        run the model sequentially on the specified number of points.

        Args:
            - image: an input image.
            - max_batch_points : number maxium of points to process at the same time
        """
        target_size = self.image_processor.target_size
        crop_boxes, points_per_crop, cropped_images = _generate_crop_boxes(image, target_size, **kwargs)
        model_inputs = self.image_processor(images=cropped_images, return_tensors="pt").to("cuda")

        pixel_values = model_inputs["pixel_values"]
        input_labels = torch.ones_like(points_per_crop[:, :, 0], dtype=torch.long)

        if point_batch_size:
            for i in range(0, points_per_crop.shape[0], point_batch_size):
                batched_points = points_per_crop[i : i + point_batch_size, :, :]
                labels = input_labels[i : i + point_batch_size]
                is_last = i == points_per_crop.shape[0] - point_batch_size
                yield {
                    "pixel_values": pixel_values,
                    "input_points": batched_points,
                    "input_labels": labels,
                    "input_boxes": crop_boxes,
                    "is_last": is_last,
                }
        else:
            yield {
                "pixel_values": pixel_values,
                "input_points": points_per_crop,
                "input_labels": input_labels,
                "input_boxes": crop_boxes,
                "is_last": True,
            }

    def forward(self, model_inputs, **forward_params):
        with self.device_placement():
            if self.framework == "pt":
                inference_context = self.get_inference_context()
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    if "image_embeddings" not in forward_params.keys():
                        image_embeddings = self.model.get_image_embeddings(model_inputs.pop("pixel_values"))
                        forward_params["image_embeddings"] = image_embeddings
                    model_outputs = self._forward(model_inputs, **forward_params)
                    model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
        return model_outputs

    def _forward(self, model_inputs, **forward_kwargs):
        input_boxes = model_inputs.pop("input_boxes")
        is_last = model_inputs.pop("is_last")
        model_outputs = self.model(**model_inputs, **forward_kwargs)
        return {"is_last": is_last, "crop_boxes": input_boxes, **model_outputs}

    def postprocess(self, model_outputs, **kwargs):
        raw_image = kwargs.pop("image")
        original_height, original_width = raw_image.shape[:2]
        total_iou_scores = []
        total_masks = []
        total_boxes = []
        for model_output in model_outputs:
            iou_scores = model_output["iou_scores"].flatten(0, 1)
            crop_boxes = model_output["crop_boxes"]
            low_resolution_masks = model_output.pop("low_resolution_masks")
            iou_scores = model_output["iou_scores"].flatten(0, 1)
            masks = self.image_processor.postprocess_masks(raw_image, low_resolution_masks, binarize=False).flatten(
                0, 1
            )
            masks, iou_scores, boxes = _filter_masks(masks, iou_scores, original_height, original_width, crop_boxes[0])
            total_iou_scores.append(iou_scores)
            total_masks.extend(masks)
            total_boxes.append(boxes)

        total_iou_scores = torch.cat(total_iou_scores)
        total_boxes = torch.cat(total_boxes)
        outputs = _postprocess_for_amg(total_masks, total_iou_scores, total_boxes)

        return outputs, raw_image
