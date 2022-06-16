from PIL import Image
import tensorflow as tf
import numpy as np
import re
import torch

from transformers import RegNetConfig, ResNetConfig
from transformers import TFRegNetForImageClassification, TFRegNetModel
from src.transformers.models.regnet.modeling_regnet import RegNetForImageClassification
from transformers import AutoFeatureExtractor
from transformers.modeling_tf_pytorch_utils import convert_tf_weight_name_to_pt_weight_name
from transformers.configuration_utils import PretrainedConfig


def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


checkpoint = "facebook/regnet-y-040"
feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
tf_model = TFRegNetForImageClassification.from_pretrained(checkpoint, from_pt=True)
pt_model = RegNetForImageClassification.from_pretrained(checkpoint)

config_class = RegNetConfig()
print(f"RegNet Config class type: {type(config_class)}.")
print(f"RegNet Config is an instance of PretrainedConfig: {isinstance(config_class, PretrainedConfig)}")
class_from_config = TFRegNetModel(config_class)
another_class_from_config = TFRegNetForImageClassification(config_class)
print("Model class from config was initialized.")

image = prepare_img()
inputs = feature_extractor(images=image, return_tensors="tf")
outputs = tf_model(**inputs, training=False)

print(outputs.logits.shape)
print(f"PT model params: {sum(p.numel() for p in pt_model.parameters())}")
print(f"TF model params: {tf_model.count_params()}")

expected_slice = np.array([-0.4180, -1.5051, -3.4836])

np.testing.assert_allclose(outputs.logits[0, :3].numpy(), expected_slice, atol=1e-4)
