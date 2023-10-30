from transformers.models.vitpose import ViTPoseModel, ViTPoseConfig, ViTPoseForPoseEstimation
from transformers.models.vitpose.image_processing_vitpose import ViTPoseImageProcessor
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw

#im1 = image.resize((256,192))
#url = "https://images.pexels.com/photos/4045762/pexels-photo-4045762.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
# model predicts bounding boxes and corresponding COCO classes
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Load and preprocess the image
image_path = "./test.jpg"
image = Image.open(image_path).resize((256,192))
inputs = processor(images=image, return_tensors="pt")

# Perform object detection
with torch.no_grad():
    outputs = model(**inputs)

# Extract bounding box coordinates and class predictions
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=.70)[0]

# Draw bounding boxes on the image
#draw = ImageDraw.Draw(image)
#for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#    box = [round(i, 2) for i in box.tolist()]
#    draw.rectangle(box, outline="red", width=3)
#    draw.text((box[0], box[1]), f"Class: {label.item()}", fill="red")
#    draw.text((box[0], box[1] - 20), f"Score: {round(score.item(), 2)}", fill="red")

# Save the annotated image
#annotated_image_path = "path_to_save_annotated_image.jpg"
#image.save(annotated_image_path)
print("result boxes", results['boxes'])


processor = ViTPoseImageProcessor.from_pretrained("shauray/vitpose")
anothermodel = ViTPoseForPoseEstimation.from_pretrained("shauray/ViTPose")
print(anothermodel)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
  result = anothermodel(**inputs,pred_boxes=results["boxes"])
print(result)
show = processor.post_processing(image, result)
to_pil_image(show).save("pred.png")

def some(result):
    for i in result:
        print(i['bbox'])
        boxes_xywh = i['bbox'].reshape(1,-1)
        boxes_xyxy = box_convert(boxes_xywh, 'xywh', 'xyxy')
        im = to_pil_image(
            draw_bounding_boxes(
                pil_to_tensor(im1),
                boxes_xyxy,
                colors="red",
            )
        )

        im.save("123.jpg")

        print(i)
        break

