import numpy as np
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


def preprocess(img, target_image_size=256):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f"min dim for image {s} < {target_image_size}")

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img


def preprocess_vqgan(x):
    x = 2.0 * x - 1.0
    return x


def custom_to_pil(x, process=True, mode="RGB"):
    x = x.detach().cpu()
    if process:
        x = post_process_tensor(x)
    x = x.numpy()
    if process:
        x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == mode:
        x = x.convert(mode)
    return x


def post_process_tensor(x):
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0)
    return x


def loop_post_process(x):
    x = post_process_tensor(x.squeeze())
    return x.permute(2, 0, 1).unsqueeze(0)
