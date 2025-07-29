import re
from typing import Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class Cohere2VisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
            "padding_side": "left",
            "truncation": True,
        },
        "images_kwargs": {
            "do_pad": True,
        },
    }


class Cohere2VisionProcessor(ProcessorMixin):
    """
    A processor that handles all the pre-processing steps needed to convert raw images and text into the format required by the Cohere2Vision model.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Cohere2VisionImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template: Optional[str] = None,
        **kwargs: Unpack[Cohere2VisionProcessorKwargs],
    ):
        assert tokenizer is not None, "tokenizer must be provided"
        assert image_processor is not None, "image_processor must be provided"

        self.image_processor = image_processor
        self.tokenizer = tokenizer

        super().__init__(
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            chat_template=chat_template if chat_template else None,
        )

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        **kwargs: Unpack[Cohere2VisionProcessorKwargs],
    ) -> BatchFeature:
        """
        TODO
        """
        if text is None:
            raise ValueError("You have to specify text.")

        if not isinstance(text, (list, tuple)):
            text = [text]

        if not isinstance(images, (list, tuple)):
            images = [images] if images is not None else None

        if images and not isinstance(images[0], (list, tuple)):
            images = [images]

        if images is None:
            text_prompts = text
            all_image_patches = [[] for _ in range(len(text))]
            image_num_patches = [0 for _ in range(len(text))]
            return self._make_batch_feature(
                text_prompts=text_prompts,
                all_image_patches=all_image_patches,
                image_num_patches=image_num_patches,
                **kwargs,
            )

        text_prompts = []
        all_image_patches = []
        image_num_patches = []

        # if some batch items don't have images then len(images) < len(text)
        assert len(images) <= len(text), (
            f"Number of images ({len(images)}) cannot exceed number of text prompts ({len(text)})."
        )

        img_idx = 0
        for text_idx in range(len(text)):
            prompt = text[text_idx]
            if "<image>" in prompt:
                imgs = images[img_idx]
                img_idx += 1
            else:
                imgs = []

            text_prompt, image_patches, num_patches = self.prepare_sample(
                images=imgs,
                text=prompt,
            )
            text_prompts.append(text_prompt)
            all_image_patches.append(image_patches)
            image_num_patches.append(num_patches)

        return self._make_batch_feature(
            text_prompts=text_prompts,
            all_image_patches=all_image_patches,
            image_num_patches=image_num_patches,
            **kwargs,
        )

    def prepare_sample(
        self,
        images: list[Image.Image],
        text: str,
    ) -> tuple[str, list[np.ndarray], int]:
        """
        Process a single text + its list of images.

        - text: may contain N occurrences of "<image>"
        - images: length N list of PIL images

        Returns:
            text_prompt:        text with each "<image>" replaced by the model's image-token string
            image_patches:      flat List[np.ndarray] of all patches, in order
            image_num_patches:  int, total number of patches across all images
        """
        splitter = re.compile(r"(<image>)")

        parts = splitter.split(text)
        n_tags = parts.count("<image>")
        if n_tags != len(images):
            raise ValueError(f"Expected {n_tags} images (one per '<image>'), but got {len(images)}.")

        out_text = []
        image_patches: list[np.ndarray] = []
        image_num_patches: int = 0
        img_idx = 0

        for chunk in parts:
            if chunk == "<image>":
                img = images[img_idx]
                img_idx += 1

                token_str, patch_list, _ = self.image_processor.process_image(img)

                # insert the token placeholder
                out_text.append(token_str)

                # collect patches
                image_patches.extend(patch_list)
                image_num_patches += len(patch_list)
            else:
                # plain text
                out_text.append(chunk)

        # join all text pieces back into one string
        text_prompt = "".join(out_text)
        return text_prompt, image_patches, image_num_patches

    def _make_batch_feature(
        self,
        text_prompts: list[str],
        all_image_patches: list[list[np.ndarray]],
        image_num_patches: list[int],
        **kwargs: Unpack[Cohere2VisionProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Cohere2VisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]

        text_inputs = self.tokenizer(text_prompts, **text_kwargs)

        # Zero-pad the image patches across the batch
        max_num_patches = max(image_num_patches)
        padded_image_patches = []
        img_size = self.image_processor.img_size
        padding_image = np.zeros((img_size, img_size, 3))
        for i, patch_seq in enumerate(all_image_patches):
            # Normalize the patch images
            patch_seq = self.image_processor.normalize_patches(patch_seq)
            padded = patch_seq + [padding_image for _ in range(max_num_patches - len(patch_seq))]
            padded_image_patches.append(padded)

        # If there are *any* images in the batch, convert to np.array
        if any(len(p) > 0 for p in padded_image_patches):
            # shape = (batch_size, num_patches, H, W, C)
            padded_image_patches = np.array(padded_image_patches)
            # We want (batch_size, num_patches, channels, height, width)
            padded_image_patches = padded_image_patches.transpose(0, 1, 4, 2, 3)
            padded_image_patches = padded_image_patches.astype(np.float16)

            # Return the final BatchFeature with text + image data
            return BatchFeature(
                data={
                    **text_inputs,
                    "pixel_values": padded_image_patches,
                    "image_num_patches": np.array(image_num_patches),
                },
                tensor_type="pt",
            )
        else:
            # No images in the entire batch (text-only). Return text
            return BatchFeature(
                data={**text_inputs},
                tensor_type="pt",
            )

    def batch_decode(self, *args, **kwargs):
        """
        Forwards all arguments to the tokenizer's batch_decode method.

        This method forwards all its arguments to PreTrainedTokenizerBase's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.

        Returns:
            List[str]: A list of decoded texts.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        Forwards all arguments to the tokenizer's decode method.

        This method forwards all its arguments to PreTrainedTokenizerBase's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["Cohere2VisionProcessor", "Cohere2VisionProcessorKwargs"]
