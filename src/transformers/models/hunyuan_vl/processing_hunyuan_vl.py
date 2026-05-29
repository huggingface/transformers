import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class HunYuanVLImagesKwargs(ImagesKwargs):
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None


class HunYuanVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: HunYuanVLImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


@auto_docstring
class HunYuanVLProcessor(ProcessorMixin):
    valid_processor_kwargs = HunYuanVLProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.tokenizer = tokenizer

        required_attrs = ["image_token", "image_start_token", "image_end_token", "vocab_size", "pad_token"]
        for attr in required_attrs:
            if not hasattr(tokenizer, attr):
                raise ValueError(
                    f"Tokenizer missing required attribute '{attr}'. "
                    "Please add the corresponding field mapping to `extra_special_tokens` in tokenizer_config.json"
                )

        self.image_token = tokenizer.image_token
        self.image_token_id = self.tokenizer.encode(self.tokenizer.image_token)[0]
        self.image_start_token = tokenizer.image_start_token
        self.image_start_token_id = self.tokenizer.encode(self.tokenizer.image_start_token)[0]
        self.image_end_token = tokenizer.image_end_token
        self.image_end_token_id = self.tokenizer.encode(self.tokenizer.image_end_token)[0]
        self.placeholder_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.vocab_size - 1)
        self.pad_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]

        self.cat_extra_token = kwargs.pop("cat_extra_token", True)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _get_spatial_patch_size(self) -> int:
        return getattr(self.image_processor, "spatial_patch_size", 1)

    def _get_image_token_count(self, grid_h: int, grid_w: int) -> tuple[int, int, int]:
        spatial_patch_size = self._get_spatial_patch_size()
        patch_h = grid_h // self.image_processor.merge_size // spatial_patch_size
        patch_w = grid_w // self.image_processor.merge_size // spatial_patch_size
        num_image_tokens = patch_h * (patch_w + 1) + (2 if self.cat_extra_token else 0)
        return patch_h, patch_w, num_image_tokens

    @staticmethod
    def _has_wrappers(prompt: str, token_start: int, start_token: str, token: str, end_token: str) -> bool:
        start_index = token_start - len(start_token)
        end_index = token_start + len(token)
        return (
            start_index >= 0
            and prompt[start_index:token_start] == start_token
            and prompt[end_index : end_index + len(end_token)] == end_token
        )

    def _build_position_ids(self, input_ids: torch.LongTensor, image_grid_thw=None) -> torch.LongTensor:
        seq_len = input_ids.shape[-1]
        device = input_ids.device

        position_ids = torch.arange(seq_len, dtype=torch.int64, device=device)
        position_ids_w = torch.arange(seq_len, dtype=torch.int64, device=device)
        position_ids_h = torch.arange(seq_len, dtype=torch.int64, device=device)
        position_ids_t = torch.arange(seq_len, dtype=torch.int64, device=device)

        if image_grid_thw is None:
            return torch.stack([position_ids, position_ids_w, position_ids_h, position_ids_t])

        image_start_token_pos_indices = torch.where(input_ids == self.image_start_token_id)[0]
        image_index = 0

        for start_pos in image_start_token_pos_indices:
            if image_index >= len(image_grid_thw):
                raise ValueError(
                    "Mismatch between image placeholders and `image_grid_thw`: "
                    f"image_index={image_index}, total_images={len(image_grid_thw)}."
                )

            _, grid_h, grid_w = [int(value) for value in image_grid_thw[image_index]]
            patch_h, patch_w, _ = self._get_image_token_count(grid_h, grid_w)
            token_start = start_pos + 1 + int(self.cat_extra_token)
            replace_num = patch_h * (patch_w + 1)

            position_ids_w[token_start : token_start + replace_num] = torch.tensor(
                list(range(patch_w + 1)) * patch_h,
                dtype=torch.int64,
                device=device,
            )
            patch_h_list = []
            for h_idx in range(patch_h):
                patch_h_list += [h_idx] * (patch_w + 1)
            position_ids_h[token_start : token_start + replace_num] = torch.tensor(
                patch_h_list,
                dtype=torch.int64,
                device=device,
            )
            image_index += 1

        return torch.stack([position_ids, position_ids_w, position_ids_h, position_ids_t])

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[HunYuanVLProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError(f"You need to provide at least one input to call {self.__class__.__name__}")

        output_kwargs = self._merge_kwargs(
            HunYuanVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        image_inputs = {}
        image_grid_thw = None
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if text is None:
            return BatchFeature(data={**image_inputs})

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        if images is not None and any(not isinstance(prompt, str) for prompt in text):
            raise ValueError(
                "`HunYuanVLProcessor` expects string prompts when multimodal inputs are provided so that multimodal "
                "placeholder tokens can be expanded before tokenization."
            )

        image_counts_per_prompt = [0] * len(text)
        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    token_start = text[i].index(self.image_token)
                    has_wrappers = self._has_wrappers(
                        text[i], token_start, self.image_start_token, self.image_token, self.image_end_token
                    )
                    _, grid_h, grid_w = [int(value) for value in image_grid_thw[index]]
                    _, _, num_image_tokens = self._get_image_token_count(grid_h, grid_w)
                    replacement = self.placeholder_token * num_image_tokens
                    if not has_wrappers:
                        replacement = self.image_start_token + replacement + self.image_end_token
                    text[i] = text[i].replace(self.image_token, replacement, 1)
                    image_counts_per_prompt[i] += 1
                    index += 1
                text[i] = text[i].replace(self.placeholder_token, self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, add_special_tokens=False, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        input_id_tensors = [torch.tensor(ids, dtype=torch.int64) for ids in text_inputs["input_ids"]]
        position_id_tensors = []
        attention_masks = []
        imgs_pos = []
        image_offset = 0

        for input_ids, image_count in zip(input_id_tensors, image_counts_per_prompt):
            sample_image_grid_thw = None
            if image_grid_thw is not None:
                sample_image_grid_thw = image_grid_thw[image_offset : image_offset + image_count]
                image_offset += image_count

            position_id_tensors.append(self._build_position_ids(input_ids, image_grid_thw=sample_image_grid_thw))
            attention_masks.append(input_ids.ne(self.pad_id).long())
            imgs_pos.append(self.get_imgs_pos(input_ids.tolist()))

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        text_inputs["position_ids"] = [position_ids.tolist() for position_ids in position_id_tensors]
        text_inputs["attention_mask"] = [attention_mask.tolist() for attention_mask in attention_masks]
        text_inputs["imgs_pos"] = imgs_pos

        if return_tensors is not None:
            try:
                text_inputs["input_ids"] = torch.stack(input_id_tensors)
                text_inputs["position_ids"] = torch.stack(position_id_tensors)
                text_inputs["attention_mask"] = torch.stack(attention_masks)
            except RuntimeError as error:
                raise ValueError(
                    "Unable to convert `HunYuanVLProcessor` outputs to tensors. Use `padding=True` or "
                    "`return_tensors=None` when batching variable-length prompts."
                ) from error

        return BatchFeature(
            data={**text_inputs, **image_inputs},
            tensor_type=return_tensors,
            skip_tensor_conversion={"imgs_pos"},
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal image inputs with the given sizes.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = HunYuanVLProcessorKwargs._defaults.get("images_kwargs", {}).copy()
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size") or self.image_processor.merge_size

            num_image_patches_size = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs) for image_size in image_sizes
            ]
            num_image_tokens = [
                patch_hw[0] // merge_size * (patch_hw[1] // merge_size + 1) + (2 if self.cat_extra_token else 0)
                for patch_hw in num_image_patches_size
            ]
            num_image_patches = [(patch_hw[0] * patch_hw[1]) for patch_hw in num_image_patches_size]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def get_imgs_pos(self, doc_ids):
        doc_ids = np.array(doc_ids, dtype=np.int64).reshape(-1)
        img_begin_index = np.where(doc_ids == self.image_start_token_id)[0]
        img_end_index = np.where(doc_ids == self.image_end_token_id)[0]
        if len(img_begin_index) == 0 or len(img_end_index) == 0:
            return []
        imgs_pos = np.concatenate(
            (np.reshape(img_begin_index + 1, (-1, 1)), np.reshape(img_end_index, (-1, 1))), axis=-1
        ).tolist()
        return imgs_pos

    @property
    def model_input_names(self):
        return list(dict.fromkeys(super().model_input_names + ["position_ids", "imgs_pos"]))


__all__ = ["HunYuanVLProcessor"]
