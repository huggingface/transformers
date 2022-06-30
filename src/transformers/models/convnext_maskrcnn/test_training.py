import torch
from transformers import ConvNextMaskRCNNForObjectDetection
from PIL import Image
import requests
import numpy as np

import torchvision.transforms as T

model = ConvNextMaskRCNNForObjectDetection.from_pretrained("nielsr/convnext-tiny-maskrcnn")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

transforms = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
img1 = transforms(image)
img2 = transforms(image)

img = torch.stack([img1, img2], dim=0)
img_metas = [{'filename': './drive/MyDrive/ConvNeXT MaskRCNN/COCO/val2017/000000039769.jpg', 'ori_filename': '000000039769.jpg', 'ori_shape': (480, 640, 3), 'img_shape': (480, 640, 3), 'pad_shape': (480, 640, 3), 'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}, {'filename': './drive/MyDrive/ConvNeXT MaskRCNN/COCO/val2017/000000039769.jpg', 'ori_filename': '000000039769.jpg', 'ori_shape': (480, 640, 3), 'img_shape': (704, 939, 3), 'pad_shape': (704, 960, 3), 'scale_factor': np.array([1.4671875, 1.4666667, 1.4671875, 1.4666667], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]
labels = dict()

labels["gt_bboxes"] = [torch.tensor([[332.8900,  79.5700, 371.5400, 185.8800],
        [ 41.8800,  74.2000, 175.0300, 119.3000]]), torch.tensor([[488.4121, 116.7027, 545.1188, 272.6240],
        [ 61.4458, 108.8267, 256.8018, 174.9733]])]
labels["gt_labels"] = [torch.tensor([65, 65]), torch.tensor([65, 65])]

outputs = model(pixel_values=img, img_metas=img_metas, labels=labels)