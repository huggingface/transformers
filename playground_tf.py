from PIL import Image
import tensorflow as tf 

from src.transformers.models.data2vec.modeling_tf_data2vec_vision import (
    TFData2VecVisionForImageClassification,
    TFData2VecVisionForSemanticSegmentation,
)
from transformers import BeitFeatureExtractor


def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


feature_extractor = BeitFeatureExtractor.from_pretrained(
    "facebook/data2vec-vision-base-ft1k"
)
model = TFData2VecVisionForSemanticSegmentation.from_pretrained(
    "facebook/data2vec-vision-base",
)


image = prepare_img()
inputs = feature_extractor(images=image, return_tensors="tf")
batch_size, num_channels, height, width = inputs["pixel_values"].shape
inputs["labels"] = tf.zeros((batch_size, height, width))
outputs = model(**inputs)

print(outputs.logits.shape)
print(outputs.loss.shape)