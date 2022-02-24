from transformers import CvtConfig, CvtForImageClassification, CvtFeatureExtractor
from transformers import BeitForImageClassification
from CvT.lib.models.cls_cvt import ConvolutionalVisionTransformer as CVT
import torch
import yaml
from PIL import Image
import requests

with open('C:\\Users\AH87766\Documents\CvT\experiments\imagenet\cvt\cvt-13-224x224.yaml', 'r') as f:
    original_config = yaml.load(f, Loader=yaml.FullLoader)

cvt_hugging_config = CvtConfig()
model1 = CvtForImageClassification(cvt_hugging_config)
model3 = CVT(spec=original_config['MODEL']['SPEC'])

model1.load_state_dict(torch.load('anugunj/cvt-base-patch13-224.pth'))
model2 = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
model3.load_state_dict(torch.load('CvT-13-224x224-IN-1k.pth', map_location=torch.device('cpu')))

model1.eval()
model2.eval()
model3.eval()

url = "000000039769.jpg"
image = Image.open(r"000000039769.jpg")

feature_extractor = CvtFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model1(**inputs)
logits1 = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits1.argmax(-1).item()
print("Predicted class:", model2.config.id2label[predicted_class_idx])

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model3(inputs.pixel_values)
logits3 = outputs
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits3.argmax(-1).item()
print("Predicted class:", model2.config.id2label[predicted_class_idx])

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model2(**inputs)
logits2 = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits2.argmax(-1).item()
print("Predicted class:", model2.config.id2label[predicted_class_idx])
