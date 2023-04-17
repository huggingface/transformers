import math
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np

from ..image_utils import load_image, to_numpy_array
from ..utils import (
    add_end_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import PIPELINE_INIT_ARGS, ChunkPipeline


if is_vision_available():
    pass

if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms
if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_AUTOMATIC_MASK_GENERATION_MAPPING

logger = logging.get_logger(__name__)

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def build_all_layer_point_grids(
    points_per_crop: int = None, n_layers: int = None, scale_per_layer: int = None
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(points_per_crop / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer

def generate_crop_boxes(
    image,
    n_layers: int = None,
    overlap_ratio: float = None,
    points_per_crop: int = None,
    scale_per_layer: int = None,
    return_tensors="pt",
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.
    """
    if isinstance(image, list):
        raise ValueError("Only one image is allowed for crop generation.")
    image = to_numpy_array(image)
    image.shape[:2]

    points_grid = build_all_layer_point_grids(
        points_per_crop=points_per_crop, n_layers=n_layers, scale_per_layer=scale_per_layer
    )
    crop_boxes, layer_idxs = [], []
    im_h, im_w = image.shape[:2]
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

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
    # for points_per_crop in total_points_per_crop:
    #     normalized_total_points_per_crop.append(
    #         [normalize_coordinates(point, original_size) for point in points_per_crop]
    #     )
    # normaliwation should happen inside the processor

    if return_tensors == "pt":
        import torch

        crop_boxes = torch.tensor(crop_boxes, dtype=torch.float32)
        points_per_crop = torch.cat(
            [torch.tensor(p).unsqueeze(0) for p in np.array(normalized_total_points_per_crop)], dim=0
        )
    else:
        raise ValueError("Only 'pt' is supported for return_tensors.")

    return crop_boxes, points_per_crop, cropped_images

def calculate_stability_score(masks, mask_threshold, threshold_offset):
    """
    Computes the stability score for a batch of masks. The stability score is the IoU between the binary masks
    obtained by thresholding the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecesary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    )
    unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions

def uncrop_boxes_xyxy(boxes, crop_box):
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset

def uncrop_masks(masks, crop_box: List[int], orig_h: int, orig_w: int):
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad, value=0)

def is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0):
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)

def batched_mask_to_box(masks):
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for an empty mask. For input shape
    C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    requires_backends("torch")

    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
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

def mask_to_rle_pytorch(tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out

def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def filter_masks(
    masks,
    iou_scores,
    original_height,
    original_width,
    cropped_box_image,
):
    r"""
    Filters the masks and iou_scores for the AMG algorithm.
    """
    requires_backends("torch")

    if masks.shape[0] != iou_scores.shape[0]:
        raise ValueError("masks and iou_scores must have the same batch size.")

    if masks.device != iou_scores.device:
        iou_scores = iou_scores.to(masks.device)

    batch_size = masks.shape[0]

    keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)

    if amg_pred_iou_thresh > 0.0:
        keep_mask = keep_mask & (iou_scores > amg_pred_iou_thresh)

    # Calculate stability score
    if amg_stability_score_thresh > 0.0:
        stability_scores = calculate_stability_score(
            masks, mask_threshold, amg_stability_score_offset
        )
        keep_mask = keep_mask & (stability_scores > amg_stability_score_thresh)

    scores = iou_scores[keep_mask]
    masks = masks[keep_mask]

    # binarize masks
    masks = masks > mask_threshold
    converted_boxes = batched_mask_to_box(masks)

    keep_mask = ~is_box_near_crop_edge(
        converted_boxes, cropped_box_image, [0, 0, original_width, original_height]
    )

    scores = scores[keep_mask]
    masks = masks[keep_mask]
    converted_boxes = converted_boxes[keep_mask]

    masks = uncrop_masks(masks, cropped_box_image, original_height, original_width)
    masks = mask_to_rle_pytorch(masks)

    return masks, scores, converted_boxes


def postprocess_for_amg(rle_masks, iou_scores, mask_boxes, amg_box_nms_thresh):
    keep_by_nms = batched_nms(
        mask_boxes.float(),
        iou_scores,
        torch.zeros(mask_boxes.shape[0]),  # categories
        iou_threshold=amg_box_nms_thresh,
    )

    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    masks = [rle_to_mask(rle) for rle in rle_masks]

    return masks, rle_masks, iou_scores, mask_boxes

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

    >>> generator(
    ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    ...     point_batch_size = 16
    ... )
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
        """

        """
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

        return preprocessor_kwargs, {}, postprocess_kwargs


    def preprocess(self, image, point_batch_size, **kwargs):
        """
        Since the pipeline inherits from `chunkPipeline` is meas that setting a `max_batch_points` allows the user to run the model
        sequentially on the specified number of points.

        Args:
            - image: an input image.
            - max_batch_points : number maxium of points to process at the same time
        """
        image = load_image(image)
        crop_boxes, points_per_crop, cropped_images = self.processor.generate_crop_boxes(image)
        model_inputs = self.processor(images=cropped_images, return_tensors="pt").to("cuda")
        target_size = model_inputs["target_size"]
        image_embeddings = self.model.get_image_embeddings(model_inputs["pixel_values"])
        input_labels = torch.ones_like(points_per_crop, dtype=torch.long)
        if point_batch_size:
            for i, (batched_points, labels) in enumerate(zip(points_per_crop,input_labels)):
                input_labels = torch.ones_like(batched_points[:, :, 0], dtype=torch.long)

                yield {
                    "image_embeddings": image_embeddings,
                    "batched_points": batched_points,
                    "input_labels": labels,
                    "target_size": target_size,
                    "crop_boxes":crop_boxes
                }
        else:
            yield {"image_embeddings": image_embeddings, "target_size": target_size, "points_per_crop": points_per_crop, "input_labels":input_labels, "crop_boxes":crop_boxes}



    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        original_height, original_width  = model_inputs.pop("original_size")
        
        points_per_crop = model_inputs.pop("points_per_crop")
        image_embeddings = model_inputs.pop("image_embeddings")
        crop_boxes = model_inputs.pop("crop_boxes")
        with torch.no_grad():
            model_outputs = self.model(**model_inputs)
            

        return model_outputs

    def postprocess(self):
        iou_scores = model_outputs.iou_scores.flatten(0, 1)
        masks = self.processor.postprocess_masks(target_size, model_outputs.low_resolution_masks, binarize=False).flatten(
                0, 1
            )
        
        masks, iou_scores, boxes = self.processor.filter_masks_for_amg(
                masks, iou_scores, original_height, original_width, crop_boxes[0]
            )
        
        outputs = self.processor.postprocess_masks_for_amg(
            iou_scores, total_iou_scores, total_boxes
        )
                
        model_outputs = {"target_size": target_size, "outputs":outputs}




