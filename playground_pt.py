import numpy as np
import torch


np.random.seed(2)
from PIL import Image

from transformers import ViTFeatureExtractor, ViTMAEConfig, ViTMAEForPreTraining


image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
inputs = feature_extractor(images=image, return_tensors="pt")

# Initialize ViTMAEConfig to calculate the total number of patches without
# `cls` token.
vit_mae_config = ViTMAEConfig()
num_patches = int((vit_mae_config.image_size // vit_mae_config.patch_size) ** 2)

# Generate a noise vector to control randomness in masking. This is needed
# to ensure that the PT and TF models operate with same inputs.
noise = np.random.uniform(size=(1, num_patches))

with torch.no_grad():
    outputs = model(**inputs, noise=torch.from_numpy(noise))

expected_shape = torch.Size((1, 196, 768))
assert outputs.logits.shape == expected_shape

# print(outputs.logits[0, :3, :3])

expected_slice_cpu = [
    [-0.0548, -1.7023, -0.9325],
    [0.3721, -0.5670, -0.2233],
    [0.8235, -1.3878, -0.3524],
]

np.testing.assert_allclose(
    outputs.logits[0, :3, :3], expected_slice_cpu, atol=1e-4
)
