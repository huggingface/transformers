from typing import List, Dict, Any, Union
import torch
from PIL import Image
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, normalize, to_channel_dimension_format
from ...utils.torch_utils import is_torch_tensor

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
            if not is_torch_tensor(img):
                img = Image.fromarray(img) if not isinstance(img, Image.Image) else img
                img = resize(img, size={"shortest_edge": self.size})
                img = to_channel_dimension_format(img, "channels_first")
                img = normalize(img, mean=self.image_mean, std=self.image_std)
            pixel_values.append(torch.as_tensor(img, dtype=torch.float32))
        pixel_values = torch.stack(pixel_values, dim=0)
        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)

    def post_process_object_detection(self, outputs, threshold: float = 0.5, target_sizes=None) -> List[Dict[str, Any]]:
        # Minimal passthrough; replace with real box/logit decoding
        logits = outputs["logits"]
        boxes = outputs["pred_boxes"]
        probs = logits.sigmoid()
        results = []
        for prob, box in zip(probs, boxes):
            keep = prob.max(dim=-1).values > threshold
            results.append({"scores": prob[keep].max(dim=-1).values, "labels": prob[keep].argmax(dim=-1), "boxes": box[keep]})
        return results
        if target_sizes is not None:
            for result, size in zip(results, target_sizes):
                img_h, img_w = size
                boxes = result["boxes"]
                boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=boxes.dtype, device=boxes.device)
                result["boxes"] = boxes
        return results
