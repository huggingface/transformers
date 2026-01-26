"""
Processor class for Molmo2.
"""

import numpy as np

from ... import AutoTokenizer
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from ...video_utils import VideoInput
from .image_processing_molmo2 import Molmo2ImageProcessor, Molmo2ImagesKwargs
from .video_processing_molmo2 import Molmo2VideoProcessor, Molmo2VideoProcessorKwargs


logger = logging.get_logger(__name__)


# Special tokens, these should be present in any tokenizer we use since the preprocessor uses them
IMAGE_PATCH_TOKEN = "<im_patch>"  # Where to insert high-res tokens
IMAGE_LOW_RES_TOKEN = "<im_low>"  # Where to insert low-res tokens
IM_START_TOKEN = "<im_start>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
FRAME_START_TOKEN = "<frame_start>"
IM_END_TOKEN = "<im_end>"
FRAME_END_TOKEN = "<frame_end>"
IM_COL_TOKEN = "<im_col>"
IMAGE_PROMPT = "<|image|>"
VIDEO_PROMPT = "<|video|>"

IMAGE_TOKENS = [
    IMAGE_PATCH_TOKEN,
    IM_COL_TOKEN,
    IM_START_TOKEN,
    LOW_RES_IMAGE_START_TOKEN,
    FRAME_START_TOKEN,
    IM_END_TOKEN,
    FRAME_END_TOKEN,
    IMAGE_LOW_RES_TOKEN,
]


class Molmo2ProcessorKwargs(ProcessingKwargs, total=False):
    """Molmo2 processor kwargs"""

    images_kwargs: Molmo2ImagesKwargs
    videos_kwargs: Molmo2VideoProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Molmo2Processor(ProcessorMixin):
    attributes = ["image_processor", "video_processor", "tokenizer"]
    optional_attributes = [
        "chat_template",
        "time_mode",
        "image_use_col_tokens",
        "use_single_crop_col_tokens",
        "use_single_crop_start_token",
        "video_use_col_tokens",
        "use_frame_special_tokens",
    ]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Molmo2ImageProcessor = None,
        video_processor: Molmo2VideoProcessor = None,
        tokenizer: AutoTokenizer = None,
        chat_template: str | None = None,
        image_use_col_tokens: bool | None = True,
        use_single_crop_col_tokens: bool | None = None,
        use_single_crop_start_token: bool | None = True,
        video_use_col_tokens: bool | None = False,
        use_frame_special_tokens: bool | None = True,
        **kwargs,
    ) -> None:
        super().__init__(
            image_processor,
            video_processor,
            tokenizer,
            chat_template=chat_template,
            image_use_col_tokens=image_use_col_tokens,
            use_single_crop_col_tokens=use_single_crop_col_tokens,
            use_single_crop_start_token=use_single_crop_start_token,
            video_use_col_tokens=video_use_col_tokens,
            use_frame_special_tokens=use_frame_special_tokens,
        )

        self.image_placeholder_token = IMAGE_PROMPT
        self.video_placeholder_token = VIDEO_PROMPT
        self.image_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in IMAGE_TOKENS]

    def get_image_tokens(self, image_grid: np.ndarray):
        resized_h, resized_w, height, width = image_grid
        per_row = np.full(width, IMAGE_PATCH_TOKEN)
        if self.image_use_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
        joint = [
            [IM_START_TOKEN],
            np.tile(per_row, [height]),
            [IM_END_TOKEN],
        ]
        per_row = np.full(resized_w, IMAGE_PATCH_TOKEN)
        use_single_crop_col_tokens = (
            self.image_use_col_tokens if self.use_single_crop_col_tokens is None else self.use_single_crop_col_tokens
        )
        image_start_token = LOW_RES_IMAGE_START_TOKEN if self.use_single_crop_start_token else IM_START_TOKEN
        if use_single_crop_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
        joint = [
            [image_start_token],
            np.tile(per_row, [resized_h]),
            [IM_END_TOKEN],
        ] + joint

        return np.concatenate(joint)

    def get_video_string(
        self,
        video_grid: np.ndarray,
        timestamps: np.ndarray,
    ):
        if self.use_frame_special_tokens:
            start_token_id = FRAME_START_TOKEN
            end_token_id = FRAME_END_TOKEN
        else:
            start_token_id = IM_START_TOKEN
            end_token_id = IM_END_TOKEN

        num_frames, h, w = video_grid
        video_string: str = ""
        for frame_idx, frame_time in enumerate(timestamps):
            # `per-frame-compact` time mode
            prev_space = " " if frame_idx > 0 else ""
            frame_prefix = prev_space + f"{frame_time:.1f} "  # explicit whitespace before/after image tokens

            video_string += frame_prefix
            per_row = np.full(w, IMAGE_PATCH_TOKEN)
            if self.video_use_col_tokens:
                per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
            extra_tokens = np.tile(per_row, [h])
            video_tokens = [
                [start_token_id],
                extra_tokens,
                [end_token_id],
            ]
            video_string += "".join(np.concatenate(video_tokens, 0))

        return video_string

    def insert_bos(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        bos_token_id: int,
        pad_token_id: int,
    ):
        """
        Args:
            input_ids: [B, S] array with left padding
            attention_mask: [B, S] array (0 for pad, 1 for valid)
            bos_token_id: int
            pad_token_id: int
        Returns:
            input_ids_out: [B, S] or [B, S+1] array with bos inserted if needed
            attention_mask_out: same shape as input_ids_out
        """

        need_to_expand = len(input_ids.shape) == 1
        if need_to_expand:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]

        B, S = input_ids.shape

        # Handle zero-length sequence
        if S == 0:
            new_input_ids = np.full((B, 1), bos_token_id, dtype=input_ids.dtype)
            new_attention_mask = np.ones((B, 1), dtype=attention_mask.dtype)
            if need_to_expand:
                new_input_ids = new_input_ids[0]
                new_attention_mask = new_attention_mask[0]
            return new_input_ids, new_attention_mask

        first_valid_index = (attention_mask == 1).argmax(axis=-1)  # [B]
        bos_already_present = np.all(input_ids[np.arange(B), first_valid_index] == bos_token_id)

        if bos_already_present:
            if need_to_expand:
                input_ids = input_ids[0]
                attention_mask = attention_mask[0]
            return input_ids, attention_mask
        else:
            new_input_ids = np.full((B, S + 1), pad_token_id, dtype=input_ids.dtype)
            new_attention_mask = np.zeros((B, S + 1), dtype=attention_mask.dtype)

            src_idx = np.tile(np.arange(S), (B, 1))  # [B, S]
            valid_mask = src_idx >= first_valid_index[:, None]  # [B, S]
            tgt_idx = src_idx + 1  # shit right
            batch_idx = np.tile(np.arange(B)[:, None], (1, S))  # [B, S]

            # flatten valid_positions
            flat_vals = input_ids[valid_mask]
            flat_batch = batch_idx[valid_mask]
            flat_tgt = tgt_idx[valid_mask]

            new_input_ids[flat_batch, flat_tgt] = flat_vals
            new_attention_mask[flat_batch, flat_tgt] = 1

            insert_pos = first_valid_index
            new_input_ids[np.arange(B), insert_pos] = bos_token_id
            new_attention_mask[np.arange(B), insert_pos] = 1

            if need_to_expand:
                new_input_ids = new_input_ids[0]
                new_attention_mask = new_attention_mask[0]

            return new_input_ids, new_attention_mask

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Molmo2ProcessorKwargs],
    ) -> BatchFeature:
        """

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            videos (`dict[str, Any]` or `list[dict[str, Any]]`):
                The video or batch of videos to be prepared. Each video can be a dictionary with the following keys:
                - `"frames"`: `np.ndarray` of shape (T, H, W, 3)
                - `"timestamps"`: `np.ndarray` of shape (T,)
                - `"sampled_fps"`: `float` (optional)
                - `"sampling_augmentation"`: `str` (optional)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            `BatchFeature`: A [`BatchFeature`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_token_pooling** -- Indices of the patches in `image_grids` to pool for each token in `image_tokens`.
              Returned when `images` is not `None`.
            - **image_grids** -- Grids of images. Returned when `images` is not `None`.
            - **image_num_crops** -- Number of crops for each image. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **video_token_pooling** -- Indices of the patches in `video_grids` to pool for each token in `video_tokens`.
              Returned when `videos` is not `None`.
            - **video_grids** -- Grids of videos. Returned when `videos` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            Molmo2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            image_grids = image_inputs["image_grids"]
        else:
            image_inputs = {}
            image_grids = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grids = videos_inputs["video_grids"]
            # If user has not requested video metadata, pop it
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grids = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if image_grids is not None:
            index = 0
            for i in range(len(text)):
                num_images = text[i].count(self.image_placeholder_token)
                image_grids_i = image_grids[index : index + num_images]
                for image_grid in image_grids_i:
                    image_tokens = self.get_image_tokens(image_grid)
                    image_string = "".join(image_tokens)
                    text[i] = text[i].replace(self.image_placeholder_token, image_string, 1)
                index += num_images

        if video_grids is not None:
            index = 0
            for i in range(len(text)):
                num_videos = text[i].count(self.video_placeholder_token)
                assert num_videos in {0, 1}, "At most one video is supported for now"
                video_grids_i = video_grids[index : index + num_videos]
                metadata_i = video_metadata[index : index + num_videos]
                for video_grid, metadata in zip(video_grids_i, metadata_i):
                    video_string = self.get_video_string(
                        video_grid,
                        metadata.timestamps,
                    )
                    text[i] = text[i].replace(self.video_placeholder_token, video_string, 1)
                index += num_videos

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)

        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids, attention_mask = self.insert_bos(input_ids, attention_mask, bos, self.tokenizer.pad_token_id)

        if return_mm_token_type_ids:
            image_tokens = np.array(self.image_token_ids).astype(input_ids.dtype)
            token_type_ids = np.any(input_ids[:, :, None] == image_tokens[None, None, :], axis=-1)
            text_inputs["token_type_ids"] = token_type_ids.tolist()

        text_inputs["input_ids"] = input_ids.tolist()
        text_inputs["attention_mask"] = attention_mask.tolist()

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


Molmo2Processor.register_for_auto_class()
