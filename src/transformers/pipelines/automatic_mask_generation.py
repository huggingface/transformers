from collections import defaultdict
from typing import List, Optional

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
    image, given an image. It is a `ChunkPipeline` because you can seperate the points in a mini-batch in order to
    avoid OOM issues. Use the `points_per_batch` argument to control the number of points that will be processed at the
    same time. Default is `64`.

    The pipeline works in 3 steps:
        1. `preprocess`: A grid of 1024 points evenly separated is generated along with boundinx boxes and point
           labels.
            For more details on how the points and bounding boxes are created, check the `_generate_crop_boxes`
            function. The image is also preprocessed using the `image_processor`. This function `yields` a minibatch of
            `points_per_batch`.

        2. `forward`: feeds the outputs of `preprocess` to the model. The image embedding is computed only once.
            Calls both `self.model.get_image_embeddings` and makes sure that the gradients are not computed, and the
            tensors and models are on the same device.

        3. `postprocess`: The most important part of the automatic mask generation happends here. Three steps
            are induced:
                - postprocess_masks (run on each minibatch loop): takes in the raw output masks, resizes them according
                  to the image size, and transforms there to binary masks.
                - _filter_masks (on each minibatch loop): uses both `pred_iou_thresh` and `stability_scores`. Also
                  applies
                 a variaty of filtesr based on non maximum suppresion to remove bad masks.
                - _postprocess_masks_for_amg:

    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        feature_extractor ([`SequenceFeatureExtractor`]):
            The feature extractor that will be used by the pipeline to encode waveform for the model.
        points_per_batch (*optional*, int, default to 64): Sets the number of points run simultaneously
                by the model. Higher numbers may be faster but use more GPU memory.
        output_bboxes_mask defaults to False

        output_rle_masks defaults to False


    Example:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="facebook/sam-vit-h", task="automatic-mask-generation")
    >>> generator(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ... )
    []

    >>> generator("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", points_per_batch=128)
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
        # preprocess args
        if "points_per_batch" in kwargs:
            preprocessor_kwargs["points_per_batch"] = kwargs["points_per_batch"]
        if "points_per_crop" in kwargs:
            preprocessor_kwargs["points_per_crop"] = kwargs["points_per_crop"]
        if "n_layers" in kwargs:
            preprocessor_kwargs["n_layers"] = kwargs["n_layers"]
        if "overlap_ratio" in kwargs:
            preprocessor_kwargs["overlap_ratio"] = kwargs["overlap_ratio"]
        if "scale_per_layer" in kwargs:
            preprocessor_kwargs["scale_per_layer"] = kwargs["scale_per_layer"]
        # postprocess args
        if "pred_iou_thresh" in kwargs:
            postprocess_kwargs["pred_iou_thresh"] = kwargs["pred_iou_thresh"]
        if "stability_score_offset" in kwargs:
            postprocess_kwargs["stability_score_offset"] = kwargs["stability_score_offset"]
        if "box_nms_thresh" in kwargs:
            postprocess_kwargs["box_nms_thresh"] = kwargs["box_nms_thresh"]
        if "mask_threshold" in kwargs:
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]
        if "output_rle_mask" in kwargs:
            postprocess_kwargs["output_rle_mask"] = kwargs["output_rle_mask"]
        if "output_bboxes_mask" in kwargs:
            postprocess_kwargs["output_bboxes_mask"] = kwargs["output_bboxes_mask"]
        return preprocessor_kwargs, {}, postprocess_kwargs

    def __call__(self, image, *args, num_workers=None, batch_size=None, **kwargs):
        """
        Generates binary segmentation masks

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):

            points_per_side (int or None):
                The number of points to be sampled along one side of the image. The total number of points is
                points_per_side**2. If None, 'point_grids' must provide explicit point sampling.
            pred_iou_thresh (float): A filtering threshold in [0,1], using the
                model's predicted mask quality.
            stability_score_thresh (float): A filtering threshold in [0,1], using
                the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
            stability_score_offset (float): The amount to shift the cutoff when
                calculated the stability score.
            box_nms_thresh (float): The box IoU cutoff used by non-maximal
                suppression to filter duplicate masks.
            crops_n_layers (int): If >0, mask prediction will be run again on
                crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image
                crops.
            crops_nms_thresh (float): The box IoU cutoff used by non-maximal
                suppression to filter duplicate masks between different crops.
            crop_overlap_ratio (float): Sets the degree to which crops overlap.
                In the first crop layer, crops will overlap by this fraction of the image length. Later layers with
                more crops scale down this overlap.
            crop_n_points_downscale_factor (int): The number of points-per-side
                sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            point_grids (list(np.ndarray) or None): A list over explicit grids
                of points used for sampling, normalized to [0,1]. The nth grid in the list is used in the nth crop
                layer. Exclusive with points_per_side.
            min_mask_region_area (int): If >0, postprocessing will be applied
                to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires
                opencv.
            output_mode (str): The form masks are returned in. Can be 'binary_mask',
                'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools. For large resolutions,
                'binary_mask' may consume large amounts of memory.

        Return:
            `Dict`: A dictionary with the following keys:
                - **mask** (`PIL.Image`) -- A binary mask of the detected object as a Pil Image of shape (width,
                  height) of
                the original image. Returns a mask filled with zeros if no object is found.
                - **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of
                  the
                "object" described by the label and the mask..

        """
        if isinstance(image, List):
            raise ValueError("The input should be a single image, the pipeline does not support multiple inputs")
        image = load_image(image)
        image = to_numpy_array(image)
        self._postprocess_params["image"] = image
        return super().__call__(image, *args, num_workers=num_workers, batch_size=batch_size, **kwargs)

    def preprocess(
        self,
        image,
        points_per_batch,
        n_layers: int = 0,
        overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = 32,
        scale_per_layer: Optional[List[int]] = 1,
    ):
        target_size = self.image_processor.target_size
        crop_boxes, grid_points, cropped_images = _generate_crop_boxes(
            image, target_size, n_layers, overlap_ratio, points_per_crop, scale_per_layer
        )
        grid_points = grid_points[None, :, :, :]
        model_inputs = self.image_processor(images=cropped_images, return_tensors="pt").to("cuda")

        pixel_values = model_inputs["pixel_values"]
        input_labels = torch.ones_like(grid_points[:, :, :, 0], dtype=torch.long)

        if points_per_batch:
            for i in range(0, grid_points.shape[1], points_per_batch):
                batched_points = grid_points[:, i : i + points_per_batch, :, :]
                labels = input_labels[:, i : i + points_per_batch]
                is_last = i == grid_points.shape[1] - points_per_batch
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
                "input_points": grid_points,
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

    def postprocess(
        self,
        model_outputs,
        image,
        output_rle_mask=False,
        output_bboxes_mask=False,
        box_nms_thresh=0.7,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
    ):
        raw_image = image
        original_height, original_width = raw_image.shape[:2]
        all_scores = []
        all_masks = []
        all_boxes = []
        for model_output in model_outputs:
            low_resolution_masks = model_output.pop("low_resolution_masks")[0]
            crop_boxes = model_output.pop("crop_boxes")[0]
            iou_scores = model_output.pop("iou_scores")[0]

            masks = self.image_processor.postprocess_masks(
                raw_image, low_resolution_masks, mask_threshold, binarize=False
            )

            masks, iou_scores, boxes = _filter_masks(
                masks,
                iou_scores,
                original_height,
                original_width,
                crop_boxes,
                pred_iou_thresh,
                stability_score_thresh,
                mask_threshold,
                stability_score_offset,
            )
            all_scores.append(iou_scores)
            all_masks.extend(masks)
            all_boxes.append(boxes)

        all_scores = torch.cat(all_scores)
        all_boxes = torch.cat(all_boxes)
        output_masks, iou_scores, rle_mask, bounding_boxes = _postprocess_for_amg(
            all_masks, all_scores, all_boxes, box_nms_thresh
        )

        extra = defaultdict(list)
        for output in model_outputs:
            # output.pop("is_last", None)
            for k, v in output.items():
                extra[k].append(v)

        optional = {}
        if output_rle_mask:
            optional["rle_mask"] = rle_mask

        if output_bboxes_mask:
            optional["bounding_boxes"] = bounding_boxes

        return {"masks": output_masks, "scores": iou_scores, **optional, **extra}
