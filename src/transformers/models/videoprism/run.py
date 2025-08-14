from transformers import VideoPrismClip, VideoPrismConfig
config = VideoPrismConfig()
model = VideoPrismClip(config)


import torch
video_inputs = torch.randn(1, 16, 3, 288, 288)
text_token_ids = torch.randint(0, 100, (5, 64), dtype=torch.long) # Example text token ID
padding = None
outputs = model(video_inputs, text_token_ids, padding)

print(outputs[0].shape, outputs[1].shape)