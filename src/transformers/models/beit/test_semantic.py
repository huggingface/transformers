from datasets import load_dataset
from PIL import Image

from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation


# load image + ground truth map
ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
image = Image.open(ds[0]["file"])
segmentation_map = Image.open(ds[1]["file"])

# load model
model_name = "nielsr/beit-base-finetuned-ade20k"
feature_extractor = BeitFeatureExtractor(do_resize=True, size=640, do_center_crop=False)
model = BeitForSemanticSegmentation.from_pretrained(model_name)

pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
outputs = model(pixel_values)
logits = outputs.logits

print("Shape of logits:", outputs.logits.shape)
print("First elements of logits:", logits[0, :3, :3, :3])
