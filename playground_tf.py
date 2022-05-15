import numpy as np
import tensorflow as tf
from PIL import Image

from src.transformers.models.data2vec.modeling_tf_data2vec_vision import TFData2VecVisionForImageClassification, TFData2VecVisionForSemanticSegmentation
from transformers import BeitFeatureExtractor


def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


feature_extractor = BeitFeatureExtractor.from_pretrained("facebook/data2vec-vision-base-ft1k")

# model = TFData2VecVisionForImageClassification.from_pretrained("facebook/data2vec-vision-base-ft1k", from_pt=True)
model = TFData2VecVisionForSemanticSegmentation.from_pretrained("facebook/data2vec-vision-base", from_pt=True)


image = prepare_img()
inputs = feature_extractor(images=image, return_tensors="tf")
outputs = model(**inputs)

print(outputs.logits.shape)

# # verify the logits
# assert outputs.logits.shape == (1, 1000)
# print("Shapes matched.")

# expected_slice = [0.3277, -0.1395, 0.0911]
# np.testing.assert_allclose(outputs.logits[0, :3].numpy(), expected_slice, atol=1e-4)
# print("Logits matched.")

# expected_top2 = [model.config.label2id[i] for i in ["remote control, remote", "tabby, tabby cat"]]
# np.testing.assert_equal(tf.nn.top_k(outputs.logits[0], 2).indices.numpy().tolist(), expected_top2)
# print("Indices matched.")
