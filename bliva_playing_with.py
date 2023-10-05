import torch
ref_proc_image = torch.load(open("/home/rafael/huggingface/code/BLIVA/processed_image.torch", "rb"))

import requests
from PIL import Image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

from transformers import BLIVAImageProcessor, AutoImageProcessor
vit_image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
bliva_image_processor = BLIVAImageProcessor.from_pretrained("rafaelpadilla/porting_bliva")
res = bliva_image_processor(image)
a = 123