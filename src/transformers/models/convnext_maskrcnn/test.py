from PIL import Image
import requests
from torchvision import transforms as T
import numpy as np

url = "https://miro.medium.com/max/1000/0*w1s81z-Q72obhE_z"
image = Image.open(requests.get(url, stream=True).raw)

# standard PyTorch mean-std input image normalization
transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

pixel_values = transform(image).unsqueeze(0)

img_metas = [
    dict(
        img_shape=(800, 1067, 3),
        scale_factor=np.array([1.6671875, 1.6666666, 1.6671875, 1.6666666], dtype=np.float32),
        ori_shape=(480, 640, 3),
    )
]