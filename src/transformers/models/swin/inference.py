import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import requests
from transformers import SwinConfig, SwinForMaskedImageModeling


# define model
IMAGE_SIZE = 192
EMBED_DIM = 128
DEPTHS = [2, 2, 18, 2]
NUM_HEADS = [4, 8, 16, 32]
WINDOW_SIZE = 6
config = SwinConfig.from_pretrained(
    "microsoft/swin-base-patch4-window7-224",
    image_size=IMAGE_SIZE,
    embed_dim=EMBED_DIM,
    depths=DEPTHS,
    num_heads=NUM_HEADS,
    window_size=WINDOW_SIZE,
)
model = SwinForMaskedImageModeling(config)

# define image transformations
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

transforms = T.Compose(
    [
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
    ]
)

# prepare image + mask
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = transforms(image).unsqueeze(0)


class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1, where 1 indicates "masked".
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())


mask_generator = MaskGenerator(
    input_size=192,
    mask_patch_size=32,
    model_patch_size=model.config.patch_size,
    mask_ratio=0.6,
)
mask = mask_generator().unsqueeze(0)

# forward pass
outputs = model(pixel_values, bool_masked_pos=mask)

print("Outputs:", outputs.keys())
