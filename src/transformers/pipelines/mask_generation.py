import math
from collections import defaultdict
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..image_utils import load_image, to_numpy_array
from ..utils import (
    add_end_docstrings,
    is_torch_available,
    is_torchvision_available,
    logging,
    requires_backends,
)
from .base import PIPELINE_INIT_ARGS, ChunkPipeline


if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_MASK_GENERATION_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class MaskGenerationPipeline(ChunkPipeline):
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

    >>> generator = pipeline(model="facebook/sam-vit-h", task="mask-generation")
    >>> generator(
    ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
    ... )
    []

    >>> generator("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", points_per_batch=128)
    []
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"mask-generation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=mask-generation).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        requires_backends(self, "vision")
        requires_backends(self, "torch")

        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        self.check_model_type(MODEL_FOR_MASK_GENERATION_MAPPING)

    def _sanitize_parameters(self, **kwargs):
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
        return super().__call__(image, *args, num_workers=num_workers, batch_size=batch_size, **kwargs)

    def preprocess(
        self,
        image,
        points_per_batch=64,
        n_layers: int = 0,
        overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = 32,
        scale_per_layer: Optional[List[int]] = 1,
    ):
        image = load_image(image)
        target_size = self.image_processor.target_size
        crop_boxes, grid_points, cropped_images, input_labels = _generate_crop_boxes(
            image, target_size, n_layers, overlap_ratio, points_per_crop, scale_per_layer
        )
        model_inputs = self.image_processor(images=cropped_images, return_tensors="pt")

        if points_per_batch:
            for i in range(0, grid_points.shape[1], points_per_batch):
                batched_points = grid_points[:, i : i + points_per_batch, :, :]
                labels = input_labels[:, i : i + points_per_batch]
                is_last = i == grid_points.shape[1] - points_per_batch
                yield {
                    "input_points": batched_points,
                    "input_labels": labels,
                    "input_boxes": crop_boxes,
                    "is_last": is_last,
                    **model_inputs,
                }
        else:
            yield {
                "input_points": grid_points,
                "input_labels": input_labels,
                "input_boxes": crop_boxes,
                "is_last": True,
                **model_inputs,
            }

    def _forward(self, model_inputs, **forward_params):
        if "image_embeddings" not in forward_params.keys():
          image_embeddings = self.model.get_image_embeddings(model_inputs.pop("pixel_values"))
          model_inputs["image_embeddings"] = image_embeddings
        input_boxes = model_inputs.pop("input_boxes")
        is_last = model_inputs.pop("is_last")
        original_sizes = model_inputs.pop("original_sizes")
        reshaped_input_sizes = model_inputs.pop("reshaped_input_sizes")

        model_outputs = self.model(**model_inputs)
        return {
            "is_last": is_last,
            "crop_boxes": input_boxes,
            "original_sizes": original_sizes,
            "reshaped_input_sizes": reshaped_input_sizes,
            **model_outputs,
        }

    def postprocess(
        self,
        model_outputs,
        output_rle_mask=False,
        output_bboxes_mask=False,
        box_nms_thresh=0.7,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
    ):
        all_scores = []
        all_masks = []
        all_boxes = []
        for model_output in model_outputs:
            original_sizes = model_output.pop("original_sizes")
            reshaped_input_sizes = model_output.pop("reshaped_input_sizes")
            low_resolution_masks = model_output.pop("pred_masks")
            masks = self.image_processor.postprocess_masks(
                original_sizes, reshaped_input_sizes, low_resolution_masks, mask_threshold, binarize=False
            )

            crop_boxes = model_output.pop("crop_boxes")
            iou_scores = model_output.pop("iou_scores")
            masks, iou_scores, boxes = _filter_masks(
                masks[0],
                iou_scores[0],
                original_sizes[0],
                crop_boxes[0],
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
            for k, v in output.items():
                extra[k].append(v)

        optional = {}
        if output_rle_mask:
            optional["rle_mask"] = rle_mask

        if output_bboxes_mask:
            optional["bounding_boxes"] = bounding_boxes

        return {"masks": output_masks, "scores": iou_scores, **optional, **extra}


def _build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def _normalize_coordinates(target_size, coords: np.ndarray, original_size, is_bounding_box=False) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format.
    """
    old_h, old_w = original_size

    scale = target_size * 1.0 / max(old_h, old_w)
    new_h, new_w = old_h * scale, old_w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)

    coords = deepcopy(coords).astype(float)

    if is_bounding_box:
        # reshape to .reshape(-1, 2, 2)
        coords = coords.reshape(-1, 2, 2)

    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)

    if is_bounding_box:
        # reshape back to .reshape(-1, 4)
        coords = coords.reshape(-1, 4)

    return coords


def _generate_crop_boxes(
    image,
    target_size,
    n_layers: int = 0,
    overlap_ratio: float = 512 / 1500,
    points_per_crop: Optional[int] = 32,
    scale_per_layer: Optional[List[int]] = 1,
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.
    """
    if isinstance(image, list):
        raise ValueError("Only one image is allowed for crop generation.")
    image = to_numpy_array(image)
    original_size = image.shape[:2]

    points_grid = []
    for i in range(n_layers + 1):
        n_points = int(points_per_crop / (scale_per_layer**i))
        points_grid.append(_build_point_grid(n_points))

    crop_boxes, layer_idxs = [], []
    im_h, im_w = original_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = int(math.ceil((overlap * (n_crops_per_side - 1) + im_w) / n_crops_per_side))
        crop_h = int(math.ceil((overlap * (n_crops_per_side - 1) + im_h) / n_crops_per_side))

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    # generate cropped images
    cropped_images = []
    total_points_per_crop = []
    for i, crop_box in enumerate(crop_boxes):
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_images.append(cropped_im)

        cropped_im_size = cropped_im.shape[:2]
        points_scale = np.array(cropped_im_size)[None, ::-1]

        total_points_per_crop.append(points_grid[layer_idxs[i]] * points_scale)

    normalized_total_points_per_crop = []
    for points_per_crop in total_points_per_crop:
        normalized_total_points_per_crop.append(
            [_normalize_coordinates(target_size, point, original_size) for point in points_per_crop]
        )

    crop_boxes = torch.tensor(crop_boxes, dtype=torch.float32)
    normalized_total_points_per_crop = np.array([normalized_total_points_per_crop])
    points_per_crop = torch.tensor(normalized_total_points_per_crop)
    points_per_crop = points_per_crop.permute(0, 2, 1, 3)

    input_labels = torch.ones_like(points_per_crop[:, :, :, 0], dtype=torch.long)

    return crop_boxes, points_per_crop, cropped_images, input_labels


def _uncrop_masks(masks, crop_box: List[int], orig_h: int, orig_w: int):
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad, value=0)


def _is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0):
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)

    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    boxes = (boxes + offset).float()

    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def _batched_mask_to_box(masks):
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for an empty mask. For input shape C1xC2x...xHxW,
    the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case

    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    height, width = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(height, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + height * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(width, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + width * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out


def _mask_to_rle_pytorch(tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by pycoco tools.
    """
    # Put in fortran order and flatten h,w
    batch_size, height, width = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(batch_size):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([height * width], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [height, width], "counts": counts})
    return out


def _rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    height, width = rle["size"]
    mask = np.empty(height * width, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity = not parity
    mask = mask.reshape(width, height)
    return mask.transpose()  # Put in C order


def _filter_masks(
    masks,
    iou_scores,
    original_sizes,
    cropped_box_image,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    mask_threshold=0,
    stability_score_offset=1,
):
    original_height, original_width = original_sizes
    iou_scores = iou_scores.flatten(0, 1)
    masks = masks.flatten(0, 1)

    if masks.shape[0] != iou_scores.shape[0]:
        raise ValueError("masks and iou_scores must have the same batch size.")

    if masks.device != iou_scores.device:
        iou_scores = iou_scores.to(masks.device)

    batch_size = masks.shape[0]

    keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)

    if pred_iou_thresh > 0.0:
        keep_mask = keep_mask & (iou_scores > pred_iou_thresh)

    # Calculate stability score
    if stability_score_thresh > 0.0:
        # One mask is always contained inside the other.
        # Save memory by preventing unnecesary cast to torch.int64
        intersections = (
            (masks > (mask_threshold + stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        )
        unions = (
            (masks > (mask_threshold - stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        )
        stability_scores = intersections / unions
        keep_mask = keep_mask & (stability_scores > stability_score_thresh)

    scores = iou_scores[keep_mask]
    masks = masks[keep_mask]

    # binarize masks
    masks = masks > mask_threshold
    converted_boxes = _batched_mask_to_box(masks)

    keep_mask = ~_is_box_near_crop_edge(converted_boxes, cropped_box_image, [0, 0, original_width, original_height])

    scores = scores[keep_mask]
    masks = masks[keep_mask]
    converted_boxes = converted_boxes[keep_mask]

    masks = _uncrop_masks(masks, cropped_box_image, original_height, original_width)
    # conversion to rle is necessary to run non-maximum suppresion
    masks = _mask_to_rle_pytorch(masks)

    return masks, scores, converted_boxes


def _postprocess_for_amg(rle_masks, iou_scores, mask_boxes, amg_box_nms_thresh=0.7):
    keep_by_nms = batched_nms(
        mask_boxes.float(),
        iou_scores,
        torch.zeros(mask_boxes.shape[0]),  # categories
        iou_threshold=amg_box_nms_thresh,
    )

    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    masks = [_rle_to_mask(rle) for rle in rle_masks]

    return masks, iou_scores, rle_masks, mask_boxes
