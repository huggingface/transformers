import re
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from ..image_transforms import convert_to_rgb, to_pil_image, unnormalize
from ..image_utils import ChannelDimension, infer_channel_dimension_format
from ..models.auto import AutoConfig, AutoProcessor


# archs failing that should raise immediately for this util:

INCOMPATIBLE_MODELS = [
    "bit",
    "colpali",
    "colqwen2",
    "convnext",
    "d_fine",
    "data2vec",
    "efficientloftr",
    "efficientnet",
    "fuyu",
    "gemma3",
    "glm4v",
    "glpn",
    "hgnet_v2",
    "hiera",
    "internvl",
    "janus",
    "layoutlmv3",
    "levit",
    "lightglue",
    "llama4",
    "mistral3",
    "mllama",
    "mobilevit",
    "mobilevitv2",
    "musicgen",
    "musicgen_melody",
    "oneformer",
    "perceiver",
    "perception_lm",
    "phi4_multimodal",
    "regnet",
    "resnet",
    "superglue",
    "superpoint",
    "swin2sr",
    "timm_wrapper",
    "tvp",
    "udop",
    "vitmatte",
    "vitpose",
    "vjepa2",
    "whisper",
]


DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/hf-logo-224x224.png"
)


def _looks_like_global(tile: np.ndarray, base: Image.Image, *, mae_tol: float = 0.05) -> bool:
    """
    Heuristic to check if a tile is a downscaled version of the original image.
    Uses mean absolute error with a strict threshold.
    """
    base_r = base.convert("RGB").resize(tile.shape[:2][::-1], Image.BILINEAR)
    base_np = np.asarray(base_r).astype(np.float32) / 255.0

    tile_f32 = tile.astype(np.float32)
    if tile_f32.max() > 1.5:
        tile_f32 /= 255.0

    mae = np.abs(tile_f32 - base_np).mean()
    return mae < mae_tol


def _find_global_tile_index(tiles: np.ndarray, base: Image.Image) -> Optional[int]:
    """
    Find which tile (if any) is the global/downscaled image.
    Checks first and last tiles only, as models place global images at these positions.

    Returns:
        Index of global tile (0 or len-1), or None if not found
    """
    if tiles.shape[0] <= 1:
        return None

    if _looks_like_global(tiles[0], base):
        return 0

    if _looks_like_global(tiles[-1], base):
        return tiles.shape[0] - 1

    return None


class ImageVisualizer:
    def __init__(self, repo_id: str):
        self.processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=False)
        self.config = AutoConfig.from_pretrained(repo_id, trust_remote_code=False)

        if hasattr(self.processor, "image_processor"):
            image_processor = self.processor.image_processor
        elif hasattr(self.processor, "image_mean"):
            image_processor = self.processor  # weak test, but works most of the time
        else:
            raise ValueError(f"No image processor found for {repo_id}.")

        self.channel_means = getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])
        self.channel_stds = getattr(image_processor, "image_std", [0.229, 0.224, 0.225])
        if hasattr(self.processor, "image_token"):
            self.image_token_marker = self.processor.image_token
        elif hasattr(self.processor, "image_token_id"):
            self.image_token_marker = self.processor.decode(self.processor.image_token_id)
        else:
            self.image_token_marker = "<image>"

        self.default_prompt = f"{self.image_token_marker} How does it look?"

        self.vision_config = getattr(self.config, "vision_config", None)
        self.patch_size = getattr(self.vision_config, "patch_size", getattr(image_processor, "patch_size", 14))
        self.merge_size = getattr(image_processor, "merge_size", 1)

    def _prepare_images_for_display(self, image_array: np.ndarray) -> np.ndarray:
        """
        Convert unnormalized images to NHWC format for display, flattening any extra batch dimensions.

        Args:
            image_array: Array of shape [..., C, H, W] or [..., H, W, C]

        Returns:
            Array of shape [N, H, W, C] suitable for plotting
        """
        input_format = infer_channel_dimension_format(image_array)

        if input_format == ChannelDimension.FIRST:
            if image_array.ndim == 3:
                image_array = image_array[np.newaxis, ...]
            elif image_array.ndim > 4:
                batch_size = int(np.prod(image_array.shape[: image_array.ndim - 3]))
                num_channels, height, width = image_array.shape[-3:]
                image_array = image_array.reshape(batch_size, num_channels, height, width)

            if image_array.ndim == 4:
                image_array = np.transpose(image_array, (0, 2, 3, 1))
        else:
            if image_array.ndim == 3:
                image_array = image_array[np.newaxis, ...]
            elif image_array.ndim > 4:
                batch_size = int(np.prod(image_array.shape[: image_array.ndim - 3]))
                height, width, num_channels = image_array.shape[-3:]
                image_array = image_array.reshape(batch_size, height, width, num_channels)

        return image_array

    def _display_single_image(
        self,
        image_array: np.ndarray,
        show_patch_grid: bool,
        figsize=(7, 7),
        patch_grid_rows=None,
        patch_grid_cols=None,
    ):
        plt.figure(figsize=figsize)
        plt.imshow(image_array)
        plt.xticks([])
        plt.yticks([])

        if show_patch_grid:
            height, width = image_array.shape[:2]

            if patch_grid_rows is not None and patch_grid_cols is not None:
                step_h = height / patch_grid_rows
                step_w = width / patch_grid_cols
                for i in range(1, patch_grid_cols):
                    plt.axvline(i * step_w, color="red", linewidth=0.5)
                for i in range(1, patch_grid_rows):
                    plt.axhline(i * step_h, color="red", linewidth=0.5)
            else:
                step = max(1, min(height, width) // self.patch_size)
                for x_pos in range(0, width, step):
                    plt.axvline(x_pos, color="red", linewidth=0.5)
                for y_pos in range(0, height, step):
                    plt.axhline(y_pos, color="red", linewidth=0.5)

        caption = f"{width}×{height} | mean={', '.join(f'{m:.3f}' for m in self.channel_means)} | std={', '.join(f'{s:.3f}' for s in self.channel_stds)}"
        plt.tight_layout()
        plt.figtext(0.5, -0.02, caption, ha="center", va="top", fontsize=12)
        plt.show()

    def _display_tiled_images(
        self,
        tiles_array: np.ndarray,
        source_image: Image.Image,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        aspect_ratio: float = 1.0,
        add_grid: bool = True,
        figsize=(7, 7),
        global_tile: Optional[np.ndarray] = None,
    ):
        """
        Display a grid of image tiles with optional global image display.

        Args:
            tiles_array: Array of tiles to display in grid format
            source_image: Original source image
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            aspect_ratio: Aspect ratio for grid layout calculation
            add_grid: Whether to add patch grid overlay
            figsize: Figure size for matplotlib
            global_tile: Optional global/downscaled image to display separately
        """
        num_tiles = tiles_array.shape[0]

        # Infer grid if not specified
        grid_rows, grid_cols = rows, cols
        if grid_rows is None or grid_cols is None:
            if aspect_ratio >= 1:
                guessed_cols = int(np.ceil(np.sqrt(num_tiles * aspect_ratio)))
                guessed_rows = int(np.ceil(num_tiles / max(guessed_cols, 1)))
            else:
                guessed_rows = int(np.ceil(np.sqrt(num_tiles / max(aspect_ratio, 1e-8))))
                guessed_cols = int(np.ceil(num_tiles / max(guessed_rows, 1)))
            grid_rows = grid_rows if grid_rows is not None else guessed_rows
            grid_cols = grid_cols if grid_cols is not None else guessed_cols

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize, squeeze=False)
        tile_index = 0
        for row_idx in range(grid_rows):
            for col_idx in range(grid_cols):
                ax = axes[row_idx, col_idx]
                if tile_index < num_tiles:
                    tile_image = tiles_array[tile_index]
                    ax.imshow(tile_image)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if add_grid:
                        height, width = tile_image.shape[:2]
                        step = max(1, min(height, width) // self.patch_size)
                        for x_pos in range(0, width, step):
                            ax.axvline(x_pos, color="red", linewidth=0.5)
                        for y_pos in range(0, height, step):
                            ax.axhline(y_pos, color="red", linewidth=0.5)
                else:
                    ax.axis("off")
                tile_index += 1

        unique = sorted({f"{t.shape[1]}×{t.shape[0]}" for t in tiles_array})
        sizes = ", ".join(unique)
        caption = f"{tiles_array.shape[0]} patches | {sizes} | mean={', '.join(f'{m:.3f}' for m in self.channel_means)} | std={', '.join(f'{s:.3f}' for s in self.channel_stds)}"
        plt.tight_layout()
        fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=12)
        plt.show()

        if global_tile is not None:
            fig2, ax2 = plt.subplots(figsize=figsize)
            ax2.imshow(global_tile)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_aspect("equal", adjustable="box")
            fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
            h0, w0 = global_tile.shape[:2]
            caption = f"Global: {w0}×{h0} | mean={', '.join(f'{m:.3f}' for m in self.channel_means)} | std={', '.join(f'{s:.3f}' for s in self.channel_stds)}"
            fig2.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=12)
            plt.show()

    def default_message(self, full_output: bool = False) -> str:
        """
        Build a single formatted prompt string using the processor's chat template.
        Contains one image (HF logo) and one user text message.
        If available, adds the generation prompt as well.
        Falls back to a minimal '<image>' string if no template is available.
        """
        # ensure this is a multimodal processor with image + tokenizer
        if not (
            hasattr(self.processor, "attributes")
            and "image_processor" in self.processor.attributes
            and "tokenizer" in self.processor.attributes
        ):
            raise RuntimeError(
                "Processor does not expose both 'image_processor' and 'tokenizer'; cannot build multimodal example."
            )

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/hf-logo-224x224.png",
                    },
                    {"type": "text", "text": "Please describe this image."},
                ],
            }
        ]

        try:
            print("For a 224x224 RGB png image: \n")
            decoded_message = self.processor.batch_decode(
                self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    truncation=False,
                ),
                skip_special_tokens=False,
            )[0]

            image_token_string = getattr(self.processor, "image_token", "<image>")
            token_escaped = re.escape(image_token_string)
            image_token_run_pattern = re.compile(rf"(?:{token_escaped})(?:\s*{token_escaped}){{2,}}")

            def compress_image_token_run(match: re.Match) -> str:
                n_tokens = match.group(0).count(image_token_string)
                return f"{image_token_string}[...{n_tokens} tokens...]{image_token_string}"

            if full_output:
                return decoded_message
            else:
                return image_token_run_pattern.sub(compress_image_token_run, decoded_message)

        except ValueError:
            image_token_string = getattr(
                self.processor,
                "image_token",
                getattr(getattr(self.processor, "tokenizer", None), "image_token", "<image>"),
            )
            return f"{image_token_string} {'Please describe this image.'}"

    def visualize(
        self,
        images: Optional[Union[Image.Image, np.ndarray, str, list[Union[Image.Image, np.ndarray, str]]]] = None,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        add_grid: bool = True,
        figsize=(12, 12),
    ):
        """
        Visualize the model-processed image(s). Only single images are supported.
        If the processor returns multiple tiles, display them in a grid with optional patch grid overlay.
        """
        if images is None:
            images = Image.open(requests.get(DEFAULT_IMAGE_URL, stream=True).raw)

        if not isinstance(images, list):
            images = [images]
        else:
            if len(images) > 1:
                raise ValueError(
                    "You passed a list of several images. Only single images are accepted by the visualizer."
                )

        pil_images = [convert_to_rgb(to_pil_image(x)) for x in images]
        img_width, img_height = pil_images[0].size
        aspect_ratio = img_width / max(img_height, 1)

        processed_inputs = self.processor(images=pil_images, text=self.default_prompt, return_tensors="pt")
        pixel_values = processed_inputs["pixel_values"]

        grid_rows = None
        grid_cols = None
        patch_grid_rows = None
        patch_grid_cols = None

        if hasattr(self.processor, "image_processor") and hasattr(
            self.processor.image_processor, "get_num_patches_from_image_size"
        ):
            num_patches, grid_rows, grid_cols = self.processor.image_processor.get_num_patches_from_image_size(
                img_width, img_height
            )

        if pixel_values.ndim == 2 and "image_grid_thw" in processed_inputs:
            num_patches, flattened_size = pixel_values.shape
            grid_thw = processed_inputs["image_grid_thw"][0]
            temporal_frames, patch_grid_h, patch_grid_w = grid_thw.tolist()

            patch_size = getattr(self.processor.image_processor, "patch_size", 14)
            temporal_patch_size = getattr(self.processor.image_processor, "temporal_patch_size", 1)
            merge_size = getattr(self.processor.image_processor, "merge_size", 2)

            expected_size = temporal_patch_size * 3 * patch_size * patch_size
            if flattened_size == expected_size:
                pixel_values = pixel_values.reshape(num_patches, temporal_patch_size, 3, patch_size, patch_size)
                pixel_values = pixel_values[:, 0, :, :, :]

                super_grid_h = patch_grid_h // merge_size
                super_grid_w = patch_grid_w // merge_size

                pixel_values = pixel_values.reshape(
                    super_grid_h, super_grid_w, merge_size, merge_size, 3, patch_size, patch_size
                )
                pixel_values = pixel_values.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
                pixel_values = pixel_values.reshape(
                    super_grid_h * merge_size, super_grid_w * merge_size, 3, patch_size, patch_size
                )
                pixel_values = pixel_values.permute(0, 3, 1, 4, 2).contiguous()
                pixel_values = pixel_values.reshape(patch_grid_h * patch_size, patch_grid_w * patch_size, 3)
                pixel_values = pixel_values.unsqueeze(0)

                patch_grid_rows = patch_grid_h
                patch_grid_cols = patch_grid_w
            else:
                raise ValueError(
                    f"Cannot reshape pixel_values: expected flattened size {expected_size} "
                    f"(temporal={temporal_patch_size} × channels=3 × patch={patch_size}×{patch_size}), "
                    f"but got {flattened_size}"
                )
        elif pixel_values.ndim == 5:
            batch_size, num_tiles, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.view(batch_size * num_tiles, num_channels, height, width)
        elif pixel_values.ndim == 4:
            pass
        elif pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

        unnormalized = unnormalize(pixel_values, mean=self.channel_means, std=self.channel_stds)
        display_ready = self._prepare_images_for_display(unnormalized)

        if display_ready.shape[0] == 1:
            self._display_single_image(
                display_ready[0],
                show_patch_grid=add_grid,
                figsize=figsize,
                patch_grid_rows=patch_grid_rows,
                patch_grid_cols=patch_grid_cols,
            )
            return

        num_tiles = display_ready.shape[0]
        global_tile = None

        if grid_rows is not None and grid_cols is not None and grid_rows * grid_cols + 1 == num_tiles:
            global_tile = display_ready[-1]
            display_ready = display_ready[:-1]
            num_tiles = display_ready.shape[0]
            if rows is None:
                rows = grid_rows
            if cols is None:
                cols = grid_cols
        else:
            global_idx = _find_global_tile_index(display_ready, pil_images[0])
            if global_idx is not None:
                global_tile = display_ready[global_idx]
                if global_idx == 0:
                    display_ready = display_ready[1:]
                else:
                    display_ready = display_ready[:-1]
                num_tiles = display_ready.shape[0]

        if rows is None or cols is None:
            tile_h, tile_w = display_ready.shape[1:3]
            tile_aspect = tile_w / tile_h if tile_h > 0 else 1.0
            target_aspect = aspect_ratio / tile_aspect

            best_rows, best_cols = 1, num_tiles
            min_diff = float("inf")
            for r in range(1, num_tiles + 1):
                c = int(np.ceil(num_tiles / r))
                diff = abs((c / r) - target_aspect)
                if diff < min_diff:
                    min_diff = diff
                    best_rows, best_cols = r, c

            rows = best_rows
            cols = best_cols

        self._display_tiled_images(
            display_ready,
            pil_images[0],
            rows=rows,
            cols=cols,
            aspect_ratio=aspect_ratio,
            add_grid=add_grid,
            figsize=figsize,
            global_tile=global_tile,
        )
