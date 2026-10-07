# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Pixtral.
"""

import os

import numpy as np

from ...processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
)
from ...utils import auto_docstring, cached_file, is_vision_available, logging
from ...utils.import_utils import requires


if is_vision_available():
    from .image_processing_pixtral import get_resize_output_image_size


logger = logging.get_logger(__name__)


class PixtralProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
@requires(backends=("torchvision", "torch"))
class PixtralProcessor(ProcessorMixin):
    valid_processor_kwargs = PixtralProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        chat_template=None,
        image_token="[IMG]",  # set the default and let users change if they have peculiar special tokens in rare cases
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to 16):
            Patch size from the vision tower.
        spatial_merge_size (`int`, *optional*, defaults to 1):
            The downsampling factor for the spatial merge operation.
        image_token (`str`, *optional*, defaults to `"[IMG]"`):
            Special token used to denote image location.
        image_break_token (`str`, *optional*, defaults to `"[IMG_BREAK]"`):
            Special token used to denote the end of a line of pixels in an image.
        image_end_token (`str`, *optional*, defaults to `"[IMG_END]"`):
            Special token used to denote the end of an image input.
        """
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token = image_break_token
        self.image_end_token = image_end_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token_id = tokenizer.convert_tokens_to_ids(self.image_break_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)

    @property
    def image_token_ids(self) -> list[int]:
        return [self.image_token_id, self.image_break_token_id, self.image_end_token_id]

    @classmethod
    def _load_tokenizer_from_pretrained(cls, sub_processor_type, pretrained_model_name_or_path, **kwargs):
        r"""Pop `mistral_format` (already resolved by [`from_pretrained`]) and load
        a [`TokenizersBackend`] directly from `tokenizer.json`.

        Using [`TokenizersBackend`] instead of [`AutoTokenizer`] avoids being misled by a
        `tokenizer_config.json` that references the wrong tokenizer class (e.g.
        `LlamaTokenizerFast` in the Mistral3 test checkpoint), which preserves the
        prior behaviour before this override existed.

        `mistral_format` is consumed and pinned to `False` by [`from_pretrained`]
        before this method is called, so there is no mistral branch here; native checkpoints
        are handled entirely inside [`from_pretrained`] via
        [`convert_tekken_image_processor`].
        """
        kwargs.pop("mistral_format", None)  # consumed upstream; prevent forwarding to AutoTokenizer
        from ...tokenization_utils_tokenizers import TokenizersBackend

        return TokenizersBackend.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> "PixtralProcessor":
        r"""Instantiate a [`PixtralProcessor`] from a pretrained checkpoint.

        In addition to the standard HuggingFace processor files, this method supports
        native Mistral checkpoints that contain `tekken.json` and `params.json` instead
        of `processor_config.json` / `tokenizer.json`.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Path, model id, or Hub identifier.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Where to cache downloaded files.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force re-download.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only look at local files.
            token (`str` or `bool`, *optional*):
                Authentication token for the Hub.
            revision (`str`, *optional*, defaults to `"main"`):
                Git revision to use.
            **kwargs:
                Additional keyword arguments. The extra keyword `mistral_format`
                (`bool`, *optional*) controls format detection — see
                [`resolve_mistral_format`] for semantics.
        """
        from ...integrations.mistral import resolve_mistral_format

        mistral_format = kwargs.pop("mistral_format", None)

        _cache_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        if token is not None:
            _cache_kwargs["token"] = token

        use_mistral, tekken_file = resolve_mistral_format(
            pretrained_model_name_or_path, mistral_format, **_cache_kwargs
        )

        if not use_mistral:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **kwargs,
            )

        # Native format: need params.json too
        params_file = cached_file(
            pretrained_model_name_or_path,
            "params.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            **_cache_kwargs,
        )
        if params_file is None:
            raise OSError(
                f"Cannot find 'params.json' at '{pretrained_model_name_or_path}'. "
                "Both 'tekken.json' and 'params.json' are required to load a native Mistral processor."
            )

        chat_template = None
        try:
            processor_dict, _ = cls.get_processor_dict(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **kwargs,
            )
            chat_template = processor_dict.get("chat_template")
        except OSError:
            pass

        from ...integrations.mistral import convert_tekken_image_processor

        return convert_tekken_image_processor(
            tokenizer_file=tekken_file,
            params_file=params_file,
            chat_template=chat_template,
        )

    def apply_chat_template(self, conversation, **kwargs):
        r"""Applies a chat template to the conversation.

        When the tokenizer is a [`MistralCommonBackend`], delegates directly to it
        (using `mistral-common`'s chat completion protocol). Otherwise falls back
        to the Jinja2-based [`ProcessorMixin`] implementation.

        Note:
            `tokenize` defaults to `False` on both paths, matching
            [`ProcessorMixin.apply_chat_template`].  [`MistralCommonBackend`]
            natively defaults to `True`; this method normalises that so callers
            get consistent behaviour regardless of which backend is active.

        Args:
            conversation: The conversation to apply the template to.
            **kwargs: Additional keyword arguments forwarded to the underlying
                `apply_chat_template` implementation.  Pass `tokenize=True`
                explicitly to receive token IDs instead of a string.

        Returns:
            The formatted conversation output.
        """
        from ...tokenization_mistral_common import MistralCommonBackend

        if not isinstance(self.tokenizer, MistralCommonBackend):
            return super().apply_chat_template(conversation, **kwargs)

        kwargs.setdefault("tokenize", False)
        kwargs.pop("return_token_type_ids", None)
        return self.tokenizer.apply_chat_template(conversation, **kwargs)

    def _process_images(self, images, **images_kwargs):
        images_kwargs["patch_size"] = self.patch_size * self.spatial_merge_size
        return super()._process_images(images, **images_kwargs)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        patch_size = self.patch_size * self.spatial_merge_size
        height, width = image_inputs["image_sizes"][image_idx]
        num_height_tokens = height // patch_size
        num_width_tokens = width // patch_size
        replace_tokens = [[self.image_token] * num_width_tokens + [self.image_break_token]] * num_height_tokens
        replace_tokens = [item for sublist in replace_tokens for item in sublist]
        replace_tokens[-1] = self.image_end_token
        return "".join(replace_tokens)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = PixtralProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size
            patch_size = self.patch_size * self.spatial_merge_size

            num_image_tokens = []
            for height, width in image_sizes:
                resized_height, resized_width = get_resize_output_image_size(
                    np.zeros((height, width, 3)),
                    size=(size["longest_edge"], size["longest_edge"]),
                    patch_size=(patch_size, patch_size),
                )
                num_height_tokens = resized_height // patch_size
                num_width_tokens = resized_width // patch_size
                num_image_tokens.append((num_width_tokens + 1) * num_height_tokens)

            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return tokenizer_input_names + image_processor_input_names + ["image_sizes"]


__all__ = ["PixtralProcessor"]
