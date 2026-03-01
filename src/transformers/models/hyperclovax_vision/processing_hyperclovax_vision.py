from PIL import Image as PILImage
from typing_extensions import Unpack

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import auto_docstring
from transformers.video_utils import VideoInput


class HCXVisionV2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            # "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": False},
    }


@auto_docstring
class HyperClovaXProcessor(ProcessorMixin):
    image_processor_class = "HyperClovaXVisionImageProcessor"
    tokenizer_class = ("GPT2Tokenizer", "GPT2TokenizerFast", "PreTrainedTokenizer", "PreTrainedTokenizerFast")
    video_processor_class = "AutoVideoProcessor"

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[HCXVisionV2ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors of a particular framework. Acceptable values are:
            - `'tf'`: Return TensorFlow `tf.constant` objects.
            - `'pt'`: Return PyTorch `torch.Tensor` objects.
            - `'np'`: Return NumPy `np.ndarray` objects.
            - `'jax'`: Return JAX `jnp.ndarray` objects.
        images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
            The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
            tensor. Both channels-first and channels-last formats are supported.
        text (`str`, `list[str]`, `list[list[str]]`):
            The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
            (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
            `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
            The video or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
            tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
              Returned when `text` is not `None`.
            - **pixel_values** -- Pixel values of images preprocessed and ready to be fed to the vision encoder.
              Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos preprocessed and ready to be fed to the vision
              encoder. Returned when `videos` is not `None`.
            - **image_grid_thw** -- Tensor of shape `(num_images, 3)` containing temporal, height, and width
              grid dimensions for each image. Returned when `images` is not `None`.
            - **video_grid_thw** -- Tensor of shape `(num_videos, 3)` containing temporal, height, and width
              grid dimensions for each video. Returned when `videos` is not `None`.
            - **image_sizes** -- List of original image sizes `{"width": W, "height": H}` before preprocessing.
              Returned when `images` is not `None` and `anyres=True`.
            - **vision_query_lengths** -- List of visual token counts for each image or video input.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import HyperClovaXProcessor

        >>> processor = HyperClovaXProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
        >>> image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        >>> messages = [{"role": "user", "content": [
        ...     {"type": "image"},
        ...     {"type": "text", "text": "What is shown in this image?"}
        ... ]}]
        >>> text = processor.apply_chat_template(messages, add_generation_prompt=True)
        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        ```
        """
        output_kwargs = self._merge_kwargs(
            HCXVisionV2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if images is not None:
            # Normalize images to a flat list so we can index by global image counter
            if isinstance(images, (list, tuple)):
                images_list = []
                for img in images:
                    if isinstance(img, (list, tuple)):
                        images_list.extend(img)
                    else:
                        images_list.append(img)
            else:
                images_list = [images]

            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    # Replace resolution placeholder with actual image size (PIL images only)
                    if index < len(images_list):
                        current_image = images_list[index]
                        from PIL import Image as PILImage

                        if isinstance(current_image, PILImage.Image):
                            text[i] = text[i].replace(
                                '{"resolution": [w, h]}',
                                '{"resolution": ' + str(list(current_image.size)) + "}",
                            )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[tuple[int, int]]`, *optional*):
                Image sizes as `(height, width)` pairs.

        Returns:
            `dict`: A dict with `num_image_tokens` and `num_image_patches` lists.
        """

        vision_data = {}
        if image_sizes is not None:
            # Create dummy images of the requested sizes and run them through the image processor
            # to let it determine the exact number of patches (handles dynamic resizing, merging, etc.)
            dummy_images = [PILImage.new("RGB", (w, h)) for h, w in image_sizes]
            image_inputs = self.image_processor(images=dummy_images)

            if "image_grid_thw" in image_inputs:
                merge_size = getattr(self.image_processor, "merge_size", 1)
                num_image_patches = [int(thw.prod()) for thw in image_inputs["image_grid_thw"]]
                num_image_tokens = [p // merge_size**2 for p in num_image_patches]
            else:
                # Fallback for CLIP-style processors without grid_thw output
                patch_size = self.image_processor.patch_size
                crop_h = self.image_processor.crop_size.get("height", 336)
                crop_w = self.image_processor.crop_size.get("width", 336)
                n = (crop_h // patch_size) * (crop_w // patch_size)
                num_image_patches = [n] * len(image_sizes)
                num_image_tokens = num_image_patches

            vision_data["num_image_tokens"] = num_image_tokens
            vision_data["num_image_patches"] = num_image_patches

        return vision_data

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

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        names_from_processor = list(
            dict.fromkeys(tokenizer_input_names + image_processor_input_names + video_processor_input_names)
        )
        return names_from_processor


__all__ = ["HyperClovaXProcessor"]
