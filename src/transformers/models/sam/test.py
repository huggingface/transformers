import requests
from PIL import Image

from transformers import (
    SamConfig,
    SamMaskDecoderConfig,
    SamModel,
    SamProcessor,
    SamPromptEncoderConfig,
    SamVisionAutoBackboneConfig,
    TinyVitConfig,
)


backbone_config = TinyVitConfig(
    image_size=1024,
    hidden_sizes=[64, 128, 160, 320],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
)
vision_config = SamVisionAutoBackboneConfig(backbone_config=backbone_config)
prompt_encoder_config = SamPromptEncoderConfig()
mask_decoder_config = SamMaskDecoderConfig()

config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)

model = SamModel(config=config)

processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt")

for k, v in inputs.items():
    print(k, v.shape)

outputs = model(**inputs)
