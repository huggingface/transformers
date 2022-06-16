from PIL import Image
import torch
import torch.nn as nn

from src.transformers.models.regnet.modeling_regnet import RegNetForImageClassification
from transformers import AutoFeatureExtractor

def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/regnet-y-040")
model = RegNetForImageClassification.from_pretrained("facebook/regnet-y-040")
print(len([x for x in model.modules() if isinstance(x, nn.BatchNorm2d)]))
print(all([x.momentum is not None for x in model.modules() if isinstance(x, nn.BatchNorm2d)]))
print(sum(p.numel() for p in model.parameters()))

image = prepare_img()
inputs = feature_extractor(images=image, return_tensors="pt")
# batch_size, num_channels, height, width = inputs["pixel_values"].shape
# inputs["labels"] = tf.zeros((batch_size, height, width))
outputs = model(**inputs)

print(outputs.logits.shape)
# print(outputs.loss.shape)

expected_slice = torch.tensor([-0.4180, -1.5051, -3.4836]).to("cpu")
torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)