import numpy as np
import torch

from transformers import PerceiverConfig, PerceiverForMultimodalAutoencoding


config = PerceiverConfig()

config.num_latents = 28 * 28 * 1
config.d_latents = 512
config.d_model = 704
config.num_blocks = 1
config.num_self_attends_per_block = 8
config.num_self_attention_heads = 8
config.num_cross_attention_heads = 1
config.num_labels = 700

# create dummy input
images = torch.randn((1, 16, 3, 224, 224))
audio = torch.randn((1, 30720, 1))
nchunks = 128
image_chunk_size = np.prod((16, 224, 224)) // nchunks
audio_chunk_size = audio.shape[1] // config.samples_per_patch // nchunks
# process the first chunk
chunk_idx = 0
subsampling = {
    "image": torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
    "audio": torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
    "label": None,
}

# define model
model = PerceiverForMultimodalAutoencoding(config, subsampling=subsampling)

# forward pass
inputs = dict(image=images, audio=audio, label=torch.zeros((images.shape[0], 700)))
outputs = model(inputs=inputs, subsampled_output_points=subsampling)
