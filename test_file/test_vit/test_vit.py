import os
import torch
from PIL import Image
from types import SimpleNamespace
from transformers import AutoConfig
from modeling_siglip_tokenizer import create_anyres_preprocess, SiglipTokenizer

base_dir = "/opensource/X-Omni-En"
config = AutoConfig.from_pretrained(base_dir, local_files_only=True, trust_remote_code=True)
vision_config = SimpleNamespace(**config.vision_config)
encoder_config = SimpleNamespace(**vision_config.encoder)
som_token, eom_token, img_token = config.mm_special_tokens[:3]
encoder_config.siglip_path = os.path.join(base_dir, encoder_config.siglip_path)
encoder_config.projector_path = os.path.join(base_dir, encoder_config.projector_path)

dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
dtype = dtype_map[vision_config.dtype]
device = torch.device("cuda")

image_transform = create_anyres_preprocess(**vision_config.transform)

# 模型加载：一次到位
image_tokenizer = SiglipTokenizer(**vars(encoder_config)).eval().to(device=device, dtype=dtype)

image_path = "/mnt/transformers/test_file/web.png"
image = Image.open(image_path)
image = image_transform(image)

# 图像处理：只加一次 batch 维度
image = image[None, ...].to(device=device, dtype=dtype)

features, (h, w), _ = image_tokenizer.vit(image)
breakpoint()

tokens = image_tokenizer.encode(image)
B, H, W = tokens.shape
tokens = tokens.view(B, -1).cpu().tolist()[0]
token_str = ''.join(map(lambda x: f'<MM-Token-{x}>', tokens))
image_str = f'{som_token}{H} {W}{img_token}{token_str}{eom_token}'
print(tokens)
breakpoint()