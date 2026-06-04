import math

import torch
from PIL import Image

from ...image_processing_base import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import ImageInput, ImageType, get_image_type, make_list_of_images
from ...utils import TensorType


__all__ = ["NemotronH_Nano_Omni_Reasoning_V3ImageProcessor"]


class NemotronH_Nano_Omni_Reasoning_V3ImageProcessor(BaseImageProcessorFast):
    """
    Dynamic-resolution image processor for the V3 omni model.

    Each image is resized to a single tile whose patch-grid `(h_patches, w_patches)` is chosen to
    land between `min_num_patches` and `max_num_patches` (on a 16×16-pixel grid), respecting
    aspect ratio. This matches the algorithm in vLLM's `DynamicResolutionImageTiler`
    (`vllm/model_executor/models/nano_nemotron_vl.py`) so HF and vLLM inference see identical pixel
    inputs.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        norm_mean=None,
        norm_std=None,
        patch_size=16,
        downsample_ratio=0.5,
        min_num_patches=1024,
        max_num_patches=13312,
        max_model_len=16384,
        video_target_num_patches=1024,
        video_maintain_aspect_ratio=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        # Integer reduction factor for pixel_shuffle (downsample_ratio = 0.5 → factor 2).
        self._downsample_factor = int(round(1.0 / downsample_ratio))
        # Per-image patch-grid bounds (on the pre-pixel-shuffle 16×16 grid).
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.max_model_len = max_model_len
        # Video frames use a separate (fixed) target-patch budget with aspect-ratio preserved.
        # Matches vLLM's `_compute_aspect_preserving_size` in `nano_nemotron_vl.py`.
        self.video_target_num_patches = video_target_num_patches
        self.video_maintain_aspect_ratio = video_maintain_aspect_ratio

    # Keep the PIL image through to `_preprocess` — we need PIL.resize (bicubic) to match vLLM's
    # algorithm exactly; resizing a tensor via `torchvision.transforms.Resize` uses different
    # kernels and breaks bit-exact agreement.
    def _process_image(self, image: ImageInput, **kwargs):
        if get_image_type(image) == ImageType.PIL:
            if image.mode != "RGB":
                image = image.convert("RGB")
        return image

    # transformers 5.6 renamed this hook from `_process_image` to `process_image`; alias both.
    process_image = _process_image

    # Toggled by `processing.py` around video calls (the strict `ImagesKwargs` validator won't let
    # us thread a new kwarg down, so we use a flag on the instance instead).
    _is_video_mode: bool = False

    def _preprocess(
        self,
        images,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Port of vLLM's `DynamicResolutionImageTiler._images_to_pixel_values_lst`.

        When `self._is_video_mode=True` (flipped by `processing.py` before the video call), each
        input is resized using the **video** target-size rule (`video_target_num_patches`,
        aspect-ratio preserved) instead of the image dynamic-res rule. This matches vLLM's split
        between `video_to_pixel_values` (video path) and `DynamicResolutionImageTiler` (image
        path).
        """
        is_video = self._is_video_mode
        images = make_list_of_images(images)

        target_sizes = []
        if is_video:
            for img in images:
                target_w_patches, target_h_patches = self._compute_target_patches_video(img)
                target_sizes.append((target_w_patches, target_h_patches))
        else:
            # Image path: per-image budget bounded by [min_num_patches, max_num_patches], with a
            # global cap derived from `max_model_len` × pixel-shuffle factor².
            num_tokens_available = self.max_model_len - 4  # match vLLM's reserve
            budget = num_tokens_available * (self._downsample_factor**2)
            budget = max(budget, self.min_num_patches * len(images))
            max_budget = self.max_num_patches if (self.max_num_patches and self.max_num_patches > 0) else float("inf")
            per_image_budget = [max(min(budget, max_budget), self.min_num_patches) for _ in images]
            # Single-pass — vLLM has an iterative scale-down for the batch, but it rarely binds in
            # single-image / small-batch inference.
            for img, tokens_for_media in zip(images, per_image_budget):
                target_w_patches, target_h_patches = self._compute_target_patches(img, tokens_for_media)
                target_sizes.append((target_w_patches, target_h_patches))

        import numpy as np

        norm_mean = torch.tensor(self.norm_mean).view(1, 3, 1, 1)
        norm_std = torch.tensor(self.norm_std).view(1, 3, 1, 1)

        pixel_values_list = []
        num_tokens_per_image = []
        imgs_sizes = []
        for img, (wp, hp) in zip(images, target_sizes):
            target_w = wp * self.patch_size
            target_h = hp * self.patch_size
            # Use torch's antialiased bicubic interpolation to match vLLM's
            # `_bicubic_resize_and_normalize` (`torch.nn.functional.interpolate`, `antialias=True`).
            # PIL's bicubic uses a different kernel (and no antialiasing), producing visibly different
            # pixel values that amplify through the 52-layer ViT / mamba stack and cause HF/vLLM
            # outputs to diverge past the first few tokens.
            arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)  # (1, 3, H, W)
            if t.shape[-2] != target_h or t.shape[-1] != target_w:
                t = torch.nn.functional.interpolate(
                    t, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True
                )
            t = (t / 255.0 - norm_mean) / norm_std
            pixel_values_list.append(t.squeeze(0))  # (3, H, W)
            num_tokens_per_image.append((wp * hp) // (self._downsample_factor**2))
            imgs_sizes.append((target_h, target_w))

        # Stack if all images have the same target size (common for same-aspect-ratio batches);
        # otherwise keep as a list of (3, H_i, W_i) tensors. The outer model's `extract_feature`
        # handles both.
        all_same_shape = all(t.shape == pixel_values_list[0].shape for t in pixel_values_list)
        if all_same_shape:
            pixel_values = torch.stack(pixel_values_list, dim=0)
        else:
            pixel_values = pixel_values_list

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                # One tile per image in dynamic mode — `num_tokens` is what the text-side
                # placeholder expansion should use.
                "num_patches": [1] * len(images),
                "num_tokens": num_tokens_per_image,
                "imgs_sizes": imgs_sizes,
            },
            tensor_type=(return_tensors if all_same_shape else None),
        )

    def _compute_target_patches(self, img: Image.Image, tokens_available: int):
        """Port of `DynamicResolutionImageTiler.process_media` (image-only, no thumbnail)."""
        orig_w, orig_h = img.width, img.height
        # Ceil-ish: `round(x + 0.5)` == `floor(x) + 1` for non-integer x, `x` for integer.
        closest_patch_h = round(orig_h / self.patch_size + 0.5)
        closest_patch_w = round(orig_w / self.patch_size + 0.5)
        patches = closest_patch_h * closest_patch_w

        # Downscale to fit the token budget.
        factor = min(math.sqrt(tokens_available / patches), 1.0)
        target_h = math.floor(factor * closest_patch_h)
        target_w = math.floor(factor * closest_patch_w)

        # Scale up if below the per-image minimum.
        if tokens_available > self.min_num_patches and target_h * target_w < self.min_num_patches:
            up = math.sqrt(self.min_num_patches / (target_h * target_w))
            target_h = math.ceil(up * target_h)
            target_w = math.ceil(up * target_w)

        # Round each dim to a multiple of the pixel_shuffle factor so tokens divide evenly.
        divisor = self._downsample_factor
        rem_h = target_h % divisor
        if rem_h:
            inc_h = divisor - rem_h
            if (target_h + inc_h) * target_w <= tokens_available:
                target_h += inc_h
            else:
                target_h = max(divisor, target_h - rem_h)
        rem_w = target_w % divisor
        if rem_w:
            inc_w = divisor - rem_w
            if target_h * (target_w + inc_w) <= tokens_available:
                target_w += inc_w
            else:
                target_w = max(divisor, target_w - rem_w)

        return target_w, target_h

    def _compute_target_patches_video(self, img: Image.Image):
        """Port of vLLM's `_compute_aspect_preserving_size` for video frames.

        Each frame is resized to roughly `video_target_num_patches` (default 1024) on the 16×16
        grid, with aspect ratio preserved and dims snapped to a multiple of the pixel_shuffle
        factor. For `maintain_aspect_ratio=False`, it falls back to a square of sqrt(target)
        patches.
        """
        orig_w, orig_h = img.width, img.height
        target = self.video_target_num_patches
        divisor = self._downsample_factor  # 2 for pixel_shuffle
        if self.video_maintain_aspect_ratio:
            aspect_wh = orig_w / max(orig_h, 1)
            ph = max(round(math.sqrt(target / aspect_wh)), 1)
            pw = max(round(math.sqrt(target * aspect_wh)), 1)
            if divisor > 1:
                rem_h = ph % divisor
                rem_w = pw % divisor
                ph_up = ph + (divisor - rem_h if rem_h else 0)
                ph_down = ph - rem_h
                pw_up = pw + (divisor - rem_w if rem_w else 0)
                pw_down = pw - rem_w
                # Prefer rounding up when the up-rounded patch count still fits the target;
                # otherwise round down (mirrors vLLM's logic exactly).
                if ph_up * pw_up <= target:
                    ph, pw = ph_up, pw_up
                else:
                    ph = max(divisor, ph_down)
                    pw = max(divisor, pw_down)
        else:
            side = int(math.sqrt(target))
            side = max(divisor, (side // divisor) * divisor)
            ph = pw = side
        return pw, ph
