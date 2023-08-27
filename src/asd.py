from transformers.models.vitpose import ViTPoseModel, ViTPoseConfig, ViTPoseForPoseEstimation
from transformers.models.vitpose.image_processing_vitpose import ViTPoseImageProcessor
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
from PIL import Image
import requests
from torchvision.transforms.functional import to_pil_image

url = "https://images.pexels.com/photos/16931314/pexels-photo-16931314/free-photo-of-portrait-of-a-senior-man-at-a-table-in-a-bar.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
url = "./test.jpg"
image = Image.open(url)
im1 = image.resize((256,192))
url = "https://images.pexels.com/photos/4045762/pexels-photo-4045762.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
image = Image.open(requests.get(url, stream=True).raw)
im1 = image
im1 = image.resize((256,192))

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")

print(model)
inputs = processor(images=image, return_tensors="pt",size={"height":256, "width":192})

outputs = model(**inputs)
target_sizes = torch.tensor([inputs.pixel_values.shape[2:]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
print("result boxes", results['boxes'])
processor = ViTPoseImageProcessor.from_pretrained("shauray/vitpose")
anothermodel = ViTPoseForPoseEstimation.from_pretrained("shauray/ViTPose")
print(anothermodel)
inputs = processor(images=im1, return_tensors="pt")

result = anothermodel(**inputs,pred_boxes=results["boxes"])
print(result)
show = processor.post_processing(im1, result)
to_pil_image(show).save("pred.png")
def some():
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

