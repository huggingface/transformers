import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms as T

from transformers import ConvNextMaskRCNNForObjectDetection
from transformers.models.convnext_maskrcnn.feature_extraction_convnext_maskrcnn import ConvNextMaskRCNNFeatureExtractor


test_cfg = {
    "score_thr": 0.05,
    "nms": {"type": "nms", "iou_threshold": 0.5},
    "max_per_img": 100,
    "mask_thr_binary": 0.5,
}

feature_extractor = ConvNextMaskRCNNFeatureExtractor(test_cfg=test_cfg)
model = ConvNextMaskRCNNForObjectDetection.from_pretrained("nielsr/convnext-tiny-maskrcnn")

url = "https://miro.medium.com/max/1000/0*w1s81z-Q72obhE_z"
image = Image.open(requests.get(url, stream=True).raw)

# standard PyTorch mean-std input image normalization
transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

pixel_values = transform(image)
pixel_values = torch.stack([pixel_values, pixel_values], dim=0)

width, height = image.size
pixel_values_height, pixel_values_width = pixel_values.shape[-2:]
width_scale = pixel_values_width / width
height_scale = pixel_values_height / height

img_metas = [
    {
        "img_shape": pixel_values.shape[1:],
        "scale_factor": np.array([width_scale, height_scale, width_scale, height_scale], dtype=np.float32),
        "ori_shape": (3, height, width),
    },
    {
        "img_shape": pixel_values.shape[1:],
        "scale_factor": np.array([width_scale, height_scale, width_scale, height_scale], dtype=np.float32),
        "ori_shape": (3, height, width),
    },
]

# forward pass
with torch.no_grad():
    outputs = model(pixel_values, img_metas=img_metas)


det_bboxes, det_labels = feature_extractor.post_process_object_detection(outputs, img_metas=img_metas)

print("Detected boxes:", det_bboxes[0].shape)

#     bbox_results = outputs.results[0][0]

# detections = []
# for label in range(len(bbox_results)):
#     if len(bbox_results[label]) > 0:
#         for detection in bbox_results[label]:
#             detections.append((label, detection))

# print("Number of detections:", len(detections))
