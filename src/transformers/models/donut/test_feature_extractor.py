from datasets import load_dataset
from PIL import Image

from transformers import DonutFeatureExtractor


dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
image = Image.open(dataset["test"][0]["file"]).convert("RGB")

feature_extractor = DonutFeatureExtractor(do_align_long_axis=True)

encoding = feature_extractor(image, return_tensors="pt")

print(encoding.pixel_values.shape)
