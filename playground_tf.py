from PIL import Image
import numpy as np
from transformers import RegNetConfig
from transformers import AutoFeatureExtractor, TFRegNetModel
from tests.models.regnet.test_modeling_tf_regnet import TFRegNetModelTester
from tests.test_modeling_tf_common import TFModelTesterMixin

def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image

# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/regnet-y-040")
# model = TFRegNetForImageClassification.from_pretrained("facebook/regnet-y-040", from_pt=True)

# image = prepare_img()
# inputs = feature_extractor(images=image, return_tensors="tf") 
# outputs = model(**inputs, training=False)

# print(outputs.logits.shape)

# expected_slice = np.array([-0.4180, -1.5051, -3.4836])

# np.testing.assert_allclose(outputs.logits[0, :3].numpy(), expected_slice, atol=1e-4)

model_tester = TFRegNetModelTester(TFModelTesterMixin)
config_and_inputs = model_tester.prepare_config_and_inputs()
config, pixel_values, _ = config_and_inputs
model = TFRegNetModel(config=config)
result = model(pixel_values, training=False)
print(result.last_hidden_state.shape)