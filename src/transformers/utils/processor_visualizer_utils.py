import re
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from transformers import AutoConfig, AutoProcessor


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
    "qwen2_5_omni",
    "qwen2_5_vl",
    "qwen2_vl",
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


def _looks_like_global(tile: np.ndarray, base: Image.Image, *, mae_tol: float = 0.3) -> bool:
    """
    Very simple visualizer heuristic.
    """
    base_r = base.convert("RGB").resize(tile.shape[:2][::-1], Image.BILINEAR)
    base_np = np.asarray(base_r).astype(np.float32) / 255.0

    tile_f32 = tile.astype(np.float32)
    if tile_f32.max() > 1.5:
        tile_f32 /= 255.0

    mae = np.abs(tile_f32 - base_np).mean()
    return mae < mae_tol


class ImageVisualizer:
    def __init__(self, repo_id: str):
        self.processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=False)
        self.config = AutoConfig.from_pretrained(repo_id, trust_remote_code=False)

        # infer processor
        if hasattr(self.processor, "image_processor"):
            image_processor = self.processor.image_processor
        elif hasattr(self.processor, "image_mean"):
            image_processor = self.processor  # weak test, but works most of the time
        else:
            raise ValueError(f"No image processor found for {repo_id}.")

        # Image normalization parameters
        self.channel_means = getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])
        self.channel_stds = getattr(image_processor, "image_std", [0.229, 0.224, 0.225])

        # Image token marker used in prompts
        self.image_token_marker = getattr(self.processor, "image_token_index", "<image>")
        self.default_prompt = f"{self.image_token_marker} How does it look?"

        # Vision configuration
        self.vision_config = getattr(self.config, "vision_config", None)
        self.patch_size = getattr(self.vision_config, "patch_size", getattr(image_processor, "patch_size", 14))
        self.merge_size = getattr(image_processor, "merge_size", 1)

    def _pixel_values_as_tensor(
        self, pixel_values: Union[torch.Tensor, np.ndarray, list[np.ndarray], list[torch.Tensor]]
    ):
        """
        Normalize input to a 4D tensor with shape (batch, channels, height, width).
        Supports input of shape:
          - (B, C, H, W)
          - (B, N, C, H, W)  -> flattened to (B*N, C, H, W)
          - (C, H, W)        -> expanded to (1, C, H, W)
          - list/tuple of arrays or tensors
        """
        if isinstance(pixel_values, (list, tuple)):
            tensor_list = [pv if isinstance(pv, torch.Tensor) else torch.tensor(pv) for pv in pixel_values]
            pixel_values = torch.stack(tensor_list, dim=0)

        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)

        if pixel_values.ndim == 5:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.view(batch_size * num_images, num_channels, height, width)
        elif pixel_values.ndim == 4:
            pass
        elif pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected pixel tensor shape {pixel_values.shape}")

        return pixel_values

    def _to_pil(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _unnormalize(
        self, pixel_values: Union[torch.Tensor, np.ndarray, list[np.ndarray], list[torch.Tensor]]
    ) -> np.ndarray:
        """
        Inverse-normalize pixel values using stored means/stds and return numpy array(s) in HWC.
        """
        tensor_pixels = self._pixel_values_as_tensor(pixel_values).float()

        num_channels = tensor_pixels.shape[-3]
        means = torch.tensor(
            self.channel_means[:num_channels], dtype=tensor_pixels.dtype, device=tensor_pixels.device
        ).view(-1, 1, 1)
        stds = torch.tensor(
            self.channel_stds[:num_channels], dtype=tensor_pixels.dtype, device=tensor_pixels.device
        ).view(-1, 1, 1)

        tensor_pixels = tensor_pixels * stds + means
        tensor_pixels = tensor_pixels.clamp(0, 1)

        if tensor_pixels.ndim == 4:
            return tensor_pixels.permute(0, 2, 3, 1).cpu().numpy()
        elif tensor_pixels.ndim == 3:
            return tensor_pixels.permute(1, 2, 0).cpu().numpy()
        else:
            raise ValueError(f"Expected 3D or 4D image tensor after normalization, got {tensor_pixels.shape}")

    def _display_single_image(self, image_array: np.ndarray, show_patch_grid: bool, figsize=(7, 7)):
        plt.figure(figsize=figsize)
        plt.imshow(image_array)
        plt.xticks([])
        plt.yticks([])

        if show_patch_grid:
            height, width = image_array.shape[:2]
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
    ):
        """
        Display a grid of image tiles. Attempts to detect and preserve the original/global image tile,
        which is then shown separately at the end.
        """
        num_tiles = tiles_array.shape[0]

        original_tile_index = None
        saved_original_tile = None

        for idx in (0, num_tiles - 1):
            if _looks_like_global(tiles_array[idx], source_image):
                original_tile_index = idx
                break

        if original_tile_index is not None:
            saved_original_tile = tiles_array[original_tile_index]
            tiles_array = np.delete(tiles_array, original_tile_index, axis=0)
            num_tiles -= 1

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

        if saved_original_tile is not None:
            plt.figure(figsize=figsize)
            plt.title("Original Image (Detected as one of the patches)")
            plt.imshow(saved_original_tile)
            plt.xticks([])
            plt.yticks([])
            h0, w0 = saved_original_tile.shape[:2]
            caption = f"{w0}×{h0} | mean={', '.join(f'{m:.3f}' for m in self.channel_means)} | std={', '.join(f'{s:.3f}' for s in self.channel_stds)}"
            plt.tight_layout()
            plt.figtext(0.5, -0.02, caption, ha="center", va="top", fontsize=12)
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
        images: Union[Image.Image, np.ndarray, str, list[Union[Image.Image, np.ndarray, str]]],
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        add_grid: bool = True,
        figsize=(12, 12),
    ):
        """
        Visualize the model-processed image(s). Only single images are supported.
        If the processor returns multiple tiles, display them in a grid with optional patch grid overlay.
        """
        if not isinstance(images, list):
            images = [images]
        else:
            if len(images) > 1:
                raise ValueError(
                    "You passed a list of several images. Only single images are accepted by the visualizer."
                )

        pil_images = [self._to_pil(x) for x in images]
        img_width, img_height = pil_images[0].size
        aspect_ratio = img_width / max(img_height, 1)

        processed_inputs = self.processor(images=pil_images, text=self.default_prompt, return_tensors="pt")
        pixel_values = processed_inputs["pixel_values"]
        unnormalized = self._unnormalize(pixel_values)
        if unnormalized.ndim == 3 or unnormalized.shape[0] == 1:
            self._display_single_image(
                unnormalized[0] if unnormalized.ndim == 4 else unnormalized,
                show_patch_grid=add_grid,
                figsize=figsize,
            )
            return
        elif unnormalized.ndim != 4:
            raise ValueError(f"Unsupported shape after unnormalization: {unnormalized.shape}")

        num_tiles = unnormalized.shape[0]

        if rows is None or cols is None:
            tile_h, tile_w = unnormalized.shape[1:3]
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
                unnormalized,
                images[0],
                rows=rows,
                cols=cols,
                aspect_ratio=aspect_ratio,
                add_grid=add_grid,
                figsize=figsize,
            )
