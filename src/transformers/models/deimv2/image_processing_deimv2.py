from typing import List, Dict, Any, Union
import torch
from PIL import Image
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, normalize, to_channel_dimension_format

import numpy as np

import torch
def is_torch_tensor(x):
    return isinstance(x, torch.Tensor)

class Deimv2ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(self, size: int = 1024, image_mean=None, image_std=None, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]

    def preprocess(self, images: List[Union[Image.Image, "np.ndarray", torch.Tensor]], return_tensors="pt", **kwargs) -> BatchFeature:
        pixel_values = []
        for img in images:
            # If tensor already, assume it is CxHxW or HxWxC depending on data
            if is_torch_tensor(img):
                t = img
                if t.ndim == 3 and t.shape[0] in (1, 3):  # channels_first
                    t = t.to(torch.float32)
                else:
                    t = t.permute(2, 0, 1).to(torch.float32)
            else:
                # Convert to PIL.Image if it's numpy array
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img.astype(np.uint8))
                img = resize(img, size={"shortest_edge": self.size})
                arr = to_channel_dimension_format(img, "channels_first")  # likely returns numpy array
                # convert to tensor and scale to [0,1]
                t = torch.tensor(arr, dtype=torch.float32) / 255.0

            # normalize (expects channels_first tensor)
            t = normalize(t, mean=self.image_mean, std=self.image_std)
            pixel_values.append(torch.as_tensor(t, dtype=torch.float32))

        pixel_values = torch.stack(pixel_values, dim=0)
        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)

    def post_process_object_detection(self, outputs, threshold: float = 0.5, target_sizes=None) -> List[Dict[str, Any]]:
        # Minimal passthrough; replace with real box/logit decoding for final PR
        logits = outputs["logits"]
        boxes = outputs["pred_boxes"]
        probs = logits.sigmoid()
        results = []
        for prob, box in zip(probs, boxes):
            keep_mask = prob.max(dim=-1).values > threshold
            kept_scores = prob[keep_mask]
            if kept_scores.numel() == 0:
                results.append({"scores": torch.tensor([]), "labels": torch.tensor([]), "boxes": torch.tensor([])})
                continue
            # for each kept index, take max score and label
            scores, _ = kept_scores.max(dim=-1)
            labels = kept_scores.argmax(dim=-1)
            kept_boxes = box[keep_mask]
            results.append({"scores": scores, "labels": labels, "boxes": kept_boxes})

        if target_sizes is not None:
            for result, size in zip(results, target_sizes):
                img_h, img_w = size
                boxes = result["boxes"]
                if isinstance(boxes, torch.Tensor) and boxes.numel() != 0:
                    # Expect boxes normalized as cxcywh or similar—user must keep consistent format
                    # Here we assume boxes are normalized [cx, cy, w, h] and convert to pixel coords [x1,y1,x2,y2]
                    # If boxes are already in xyxy, remove the conversion step.
                    cxcywh = boxes
                    cx = cxcywh[:, 0] * img_w
                    cy = cxcywh[:, 1] * img_h
                    w = cxcywh[:, 2] * img_w
                    h = cxcywh[:, 3] * img_h
                    x1 = cx - 0.5 * w
                    y1 = cy - 0.5 * h
                    x2 = cx + 0.5 * w
                    y2 = cy + 0.5 * h
                    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    result["boxes"] = boxes_xyxy
        return results
