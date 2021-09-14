import torch
from transformers import PerceiverImagePreprocessor, PerceiverConfig, PerceiverForImageClassification


# TEST 1: testing PerceiverImagePreprocessor
pixel_values = torch.randn((1, 3, 224, 224))
config = PerceiverConfig()
processor = PerceiverImagePreprocessor(config, 
    prep_type="conv1x1", out_channels=256, spatial_downsample=1, concat_or_add_pos="concat"
)

inputs  = processor(pixel_values)
print(inputs.shape)

# TEST 2: testing PerceiverForImageClassification
config.num_labels = 1000
config.num_latents = 512
config.d_latents = 1024
config.d_model = 512
model = PerceiverForImageClassification(config)

outputs = model(pixel_values)
print(outputs.logits.shape)
