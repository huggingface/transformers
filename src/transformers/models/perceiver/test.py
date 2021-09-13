import torch
from transformers import PerceiverImagePreprocessor

pixel_values = torch.randn((1,3,224,224))

processor = PerceiverImagePreprocessor(prep_type='conv1x1',
                                       out_channels=256,
                                       spatial_downsample=1,
                                       concat_or_add_pos='concat')

inputs, modality_sizes, inputs_without_pos = processor(pixel_values)
print(inputs.shape)
print(inputs_without_pos.shape)

# from transformers import PerceiverConfig, PerceiverModel


# config = PerceiverConfig()
# model = PerceiverModel(config)

# # assuming we have already turned our input_ids into embeddings
# inputs = torch.randn((2, 2048, 768))
# outputs = model(inputs)

# print("Shape of outputs:", outputs.last_hidden_state.shape)
