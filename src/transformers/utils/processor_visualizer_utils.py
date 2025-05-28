import torch
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoProcessor, AutoConfig
from PIL import Image
from enum import Enum

import torchvision.transforms.functional as F

class ImageMode(Enum):
    SINGLE = "single"
    TILED = "tiled"

class ImageVisualizer:
    def __init__(self, repo_id: str):
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.config = AutoConfig.from_pretrained(repo_id)
        self.means = getattr(self.processor.image_processor, "image_mean", [0.485, 0.456, 0.406])
        self.stds = getattr(self.processor.image_processor, "image_std", [0.229, 0.224, 0.225])
        self.image_token = getattr(self.processor, "image_token_index", "<image>")
        self.default_prompt = f"{self.image_token} How does it look?"
        self.patch_size = getattr(self.config.vision_config, "patch_size", 14)

    def _to_pil(self, x):
        if isinstance(x, str):
            return Image.open(x)
        elif isinstance(x, np.ndarray):
            return Image.fromarray(x)
        return x

    def _unnormalize(self, pixel_values: torch.Tensor) -> np.ndarray:
        pixel_values = pixel_values.float()
        c = pixel_values.shape[-3]
        mean = torch.tensor(self.means[:c], dtype=pixel_values.dtype, device=pixel_values.device).view(-1, 1, 1)
        std = torch.tensor(self.stds[:c],  dtype=pixel_values.dtype, device=pixel_values.device).view(-1, 1, 1)
        pixel_values = pixel_values * std + mean
        pixel_values = pixel_values.clamp(0, 1)
        if pixel_values.ndim == 3:
            return pixel_values.permute(1, 2, 0).cpu().numpy()
        elif pixel_values.ndim == 4:
            return pixel_values.permute(0, 2, 3, 1).cpu().numpy()
        else:
            raise ValueError(f"Expected 3D or 4D, got shape {pixel_values.shape}")

    def _display_single_image(self, arr: np.ndarray, add_grid: bool, figsize=(7, 7)):
        plt.figure(figsize=figsize)
        plt.imshow(arr)
        plt.xticks([])
        plt.yticks([])
        if add_grid:
            h, w = arr.shape[:2]
            step = max(1, min(h, w) // self.patch_size)
            for x in range(0, w, step):
                plt.axvline(x, color='red', linewidth=0.5)
            for y in range(0, h, step):
                plt.axhline(y, color='red', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def _display_tiled_images(
        self,
        arr: np.ndarray,
        original_image: Image,
        rows: int = None,
        cols: int = None,
        aspect_ratio: float = 1.0,
        add_grid=True,
        figsize=(7, 7),
    ):
        num_tiles = arr.shape[0]
        num_tiles = arr.shape[0]
        orig_patch_index = None
        if original_image is not None and num_tiles > 0:
            resized = original_image.convert('RGB').resize(arr[0].shape[:2])
            if np.allclose(arr[0], resized, atol=1e-2):
                orig_patch_index = 0
            elif np.allclose(arr[-1], resized, atol=1e-2):
                orig_patch_index = num_tiles - 1
        saved_orig_tile = None


        # FIXME global image detection is broken - infer from processor class/docstring?

        orig_patch_index = num_tiles - 1 # Will fail in many cases
        if orig_patch_index is not None:
            saved_orig_tile = arr[orig_patch_index]
            arr = np.delete(arr, orig_patch_index, axis=0)
            num_tiles -= 1
        if rows is None or cols is None:
            if aspect_ratio >= 1:
                guess_cols = int(np.ceil(np.sqrt(num_tiles * aspect_ratio)))
                guess_rows = int(np.ceil(num_tiles / guess_cols))
            else:
                guess_rows = int(np.ceil(np.sqrt(num_tiles / aspect_ratio)))
                guess_cols = int(np.ceil(num_tiles / guess_rows))
            rows = rows if rows is not None else guess_rows
            cols = cols if cols is not None else guess_cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        idx = 0
        for r in range(rows):
            for c_ in range(cols):
                ax = axes[r, c_]
                if idx < num_tiles:
                    tile = arr[idx]
                    ax.imshow(tile)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if add_grid:
                        h, w = tile.shape[:2]
                        step = max(1, min(h, w) // self.patch_size)
                        for x in range(0, w, step):
                            ax.axvline(x, color='red', linewidth=0.5)
                        for y in range(0, h, step):
                            ax.axhline(y, color='red', linewidth=0.5)
                else:
                    ax.axis("off")
                idx += 1
        plt.tight_layout()
        plt.show()
        if saved_orig_tile is not None:
            plt.figure(figsize=figsize)
            plt.title("Original Image (Detected as one of the patches)")
            plt.imshow(saved_orig_tile)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()

    def visualize(self, images, rows=None, cols=None, add_grid=True, figsize=(12, 12)):
        if not isinstance(images, list):
            images = [images]
        else:
            if len(images) > 1:
                raise ValueError("You passed a list of several images. Only single images are accepted by the visualizer.")
        width, height = images[0].size
        aspect_ratio = width / height
        pil_imgs = [self._to_pil(x) for x in images]
        processed = self.processor(images=[pil_imgs], text=[self.default_prompt], return_tensors="pt")
        pixel_values = processed["pixel_values"]
        pixel_values = pixel_values.squeeze(0)
        arr = self._unnormalize(pixel_values)
        if arr.ndim == 3:
            self._display_single_image(arr, add_grid=add_grid, figsize=figsize)
        elif arr.ndim == 4:
            self._display_tiled_images(
                arr,
                images[0],
                rows=rows,
                cols=cols,
                aspect_ratio=aspect_ratio,
                add_grid=add_grid,
                figsize=figsize,
            )
        else:
            raise ValueError(f"Unsupported shape after squeeze: {arr.shape}.")
