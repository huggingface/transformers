from transformers.models.rtdetrv2.convert_test import get_rt_detr_v2_config
from transformers.models.rt_detr.image_processing_rt_detr import RTDetrImageProcessor
from transformers.models.rtdetrv2.modular_rtdetrv2 import RTDetrv2Config, RTDetrv2ForObjectDetection

from PIL import Image
import requests
import torch
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
model = RTDetrv2ForObjectDetection.from_pretrained("/home/user/app/transformers/src/transformers/models/rtdetrv2/output")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

# forward pass
outputs = model(**inputs)

logits = outputs.logits
list(logits.shape)

boxes = outputs.pred_boxes
list(boxes.shape)

# convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    0
]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )