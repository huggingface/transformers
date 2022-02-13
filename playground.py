import tensorflow as tf
from transformers import AutoFeatureExtractor

# import your TFConvNextForImageClassification class here, we will take care
# of adding the boilerplate to run `from transformers import
# TFConvNextForImageClassification` later
from src.transformers.models.convnext.modeling_tf_convnext import TFConvNextForImageClassification
from transformers import ConvNextForImageClassification

from PIL import Image

# model = ConvNextForImageClassification.from_pretrained(
#     "facebook/convnext-tiny-224",
# )
# print(f"Model State Dict:\n")
# all_keys = list(model.state_dict().keys())
# print([k for k in all_keys if "layer_scale" in k])

model = TFConvNextForImageClassification.from_pretrained(
    "facebook/convnext-tiny-224",
    from_pt=True,
)  # notice the `from_pt` argument
print(model.summary(expand_nested=True))


# feature_extractor = AutoFeatureExtractor.from_pretrained(
#     "facebook/convnext-tiny-224"
# )  # don't know if this is supposed to work with TF as well, change this as needed

# image = Image.open("tests/fixtures/tests_samples/COCO/000000039769.png")  # you might need to change the relative path
# inputs = feature_extractor(images=image, return_tensors="tf")

# # forward pass
# outputs = model(**inputs)

# # verify the logits
# assert outputs.logits.shape == [1, 1000]
# tf.debugging.assert_near(outputs.logits[0, :3], [-0.0260, -0.4739, 0.1911], atol=1e-4)
