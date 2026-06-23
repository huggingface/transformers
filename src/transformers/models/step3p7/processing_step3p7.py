from __future__ import annotations

from transformers.feature_extraction_utils import BatchFeature, TensorType
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_tokenizers import TokenizersBackend
from transformers.utils import auto_docstring, is_vision_available
from transformers.utils.import_utils import is_torchvision_available, requires


if is_vision_available() and is_torchvision_available():
    from .image_processing_step3p7 import Step3VisionProcessor


class Step3VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "images_kwargs": {
            "is_patch": False,
        },
    }


@auto_docstring
@requires(backends=("vision", "torch", "torchvision"))
class Step3VLProcessor(ProcessorMixin):
    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        return TokenizersBackend.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            **kwargs,
        )

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        try:
            return super()._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
        except OSError as error:
            if "Can't load image processor" not in str(error):
                raise

        kwargs = kwargs.copy()
        subfolder = kwargs.pop("subfolder", "")
        tokenizer = cls._load_tokenizer_from_pretrained(
            "tokenizer", pretrained_model_name_or_path, subfolder=subfolder, **kwargs
        )
        return [Step3VisionProcessor(), tokenizer]

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs) -> None:
        if image_processor is None:
            image_processor = kwargs.pop("image_preprocessor", None)
        image_processor = image_processor if image_processor is not None else Step3VisionProcessor()
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template, **kwargs)
        self.image_preprocessor = self.image_processor

        self.num_image_feature_size = 169
        self.num_patch_feature_size = 81
        self.image_token = "<im_patch>"
        self.image_feature_placeholder = self.image_token * self.num_image_feature_size
        self.patch_feature_placeholder = self.image_token * self.num_patch_feature_size

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.image_token]

    def get_num_image_tokens(self, img_width: int, img_height: int) -> int:
        num_patches, num_newlines = self.image_processor.get_num_patches(img_width, img_height)

        return num_patches * (self.num_patch_feature_size + 2) + self.num_image_feature_size + 2 + num_newlines

    def _get_patch_repl(
        self,
        num_patches: int,
        patch_newline_mask: list[bool] | None,
    ) -> str:
        text = ""
        for i in range(num_patches):
            assert len(patch_newline_mask) == num_patches
            text += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"
            if patch_newline_mask and patch_newline_mask[i]:
                text += "<patch_newline>"
        return text

    def _get_image_repl(self, num_images: int) -> str:
        text = f"<im_start>{self.image_feature_placeholder}<im_end>"
        return text * num_images

    def _get_image_repl_features(
        self,
        num_images: int,
        num_patches: int,
        patch_new_line_idx: list[bool] | None,
    ) -> str:
        patch_repl = self._get_patch_repl(num_patches, patch_new_line_idx) if num_patches > 0 else ""
        return patch_repl + self._get_image_repl(num_images)

    def replace_placeholder(self, text: str, placeholder: str, repls: list[str]) -> str:
        parts = text.split(placeholder)

        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."  # noqa: E501
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    @auto_docstring
    def __call__(
        self,
        text: str | list[str] | None = None,
        images: ImageInput | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Step3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if self.tokenizer is not None else {},
            **kwargs,
        )
        images_kwargs = output_kwargs["images_kwargs"]
        images_kwargs.pop("is_patch", None)
        images_kwargs.setdefault("return_tensors", "pt")
        text_kwargs = output_kwargs["text_kwargs"]
        if return_tensors is not None:
            text_kwargs["return_tensors"] = return_tensors

        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]

        if images is None:
            image_inputs = {}
            text_inputs = self.tokenizer(text, **text_kwargs)
        else:
            image_inputs, patch_newline_masks = self.image_processor.prepare_image_inputs(images, **images_kwargs)
            if not image_inputs:
                text_inputs = self.tokenizer(text, **text_kwargs)
            else:
                image_repl_str_lst = [
                    self._get_image_repl_features(1, num_patches, patch_newline_mask)
                    for num_patches, patch_newline_mask in zip(image_inputs["num_patches"], patch_newline_masks)
                ]
                text = [self.replace_placeholder(t, self.image_token, image_repl_str_lst) for t in text]
                text_inputs = self.tokenizer(text, **text_kwargs)

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


Step3p7Processor = Step3VLProcessor

__all__ = ["Step3VLProcessor", "Step3p7Processor"]
