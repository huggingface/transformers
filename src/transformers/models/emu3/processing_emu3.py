# Copyright 2024 HuggingFace Inc. team. All rights reserved.
#
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


from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_vision_available
from ...utils.import_utils import requires


if is_vision_available():
    from .image_processing_emu3 import Emu3ImageProcessorKwargs, smart_resize


class Emu3TextKwargs(TextKwargs, total=False):
    """
    return_for_image_generation (`bool`, *optional*, defaults to `False`):
        Whether the processed text is intended for image generation tasks. When `True`, the processor prepares
        inputs for image generation by appending image start tokens and size information to the prompt, and
        images should not be provided. When `False`, the processor prepares inputs for text generation from
        images and text, requiring both inputs to be provided.
    """

    return_for_image_generation: bool


class Emu3ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Emu3TextKwargs
    images_kwargs: Emu3ImageProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "return_for_image_generation": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "ratio": "1:1",
            "image_area": 518400,
        },
    }


@auto_docstring
@requires(backends=("vision",))
class Emu3Processor(ProcessorMixin):
    valid_processor_kwargs = Emu3ProcessorKwargs

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = tokenizer.image_token  # image_token as placeholder to be replaced by vq-vae tokens
        self.image_token_id = tokenizer.image_token_id
        self.image_start_token = tokenizer.boi_token  # "<|image start|>" fixed tokens for start and end of image
        self.image_end_token = tokenizer.eoi_token  # "<|image end|>"
        self.fake_token_around_image = tokenizer.image_wrapper_token  # "<|image token|>"  every image starts with it
        self.eof_token = tokenizer.eof_token  # "<|extra_201|>"
        self.bos_token = tokenizer.bos_token
        self.downsample_ratio = 8
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[Emu3ProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            Emu3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_for_image_generation = output_kwargs["text_kwargs"].pop("return_for_image_generation", False)
        ratio = output_kwargs["images_kwargs"].pop("ratio", None)
        image_area = output_kwargs["images_kwargs"].pop("image_area", None)

        # take different processing path when generarating images cond on text
        if return_for_image_generation:
            if images is not None:
                raise ValueError("You should not provide `images` when `return_for_image_generation=True`")
            height, width = self.calculate_generate_size(ratio, image_area, self.downsample_ratio)
            image_prompt = f"{self.image_start_token}{height}*{width}{self.fake_token_around_image}"
            if isinstance(text, str):
                text = [text]
            text = [f"{self.bos_token}{sample}{image_prompt}" for sample in text]

        model_inputs = super().__call__(images=images, text=text, **output_kwargs)
        if return_for_image_generation:
            model_inputs["image_sizes"] = [[height, width]] * len(text)
        return model_inputs

    def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
        images, text, *_ = super().prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        if images is not None and text is not None:
            # Add BOS once per sample; GPT tokenizer doesn't add it automatically
            text = [f"{self.bos_token}{sample}" for sample in text]
        return images, text, videos, audio

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        height, width = image_inputs["image_sizes"][image_idx]
        height = height // self.downsample_ratio
        width = width // self.downsample_ratio
        image_seq_length = height * (width + 1)  # +1 for extra row when converting to BPE in modeling code
        return (
            f"{self.image_start_token}{height}*{width}"
            f"{self.fake_token_around_image}"
            f"{self.image_token * image_seq_length}"
            f"{self.eof_token}{self.image_end_token}"
        )

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
            num_image_tokens = []
            for height, width in image_sizes:
                height, width = smart_resize(
                    height,
                    width,
                    self.image_processor.spatial_factor,
                    self.image_processor.min_pixels,
                    self.image_processor.max_pixels,
                )
                height = height // self.downsample_ratio
                width = width // self.downsample_ratio
                image_seq_length = height * (width + 1)  # +1 for extra row when converting to BPE in modeling code
                num_image_tokens.append(image_seq_length)

            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def calculate_generate_size(self, ratio, image_area, spatial_factor):
        width, height = map(int, ratio.split(":"))
        current_area = width * height
        target_ratio = (image_area / current_area) ** 0.5

        token_height = int(round(height * target_ratio / spatial_factor))
        token_width = int(round(width * target_ratio / spatial_factor))
        return token_height, token_width

    def postprocess(self, images: ImageInput, **kwargs):
        return self.image_processor.postprocess(images, **kwargs)

    def post_process_multimodal_output(
        self, generated_outputs, skip_special_tokens=True, generation_mode=None, **kwargs
    ):
        """
        Post-process the output of a multimodal model to return the requested modality output.
        If the model cannot generated the requested modality, an error will be raised.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            generation_mode (`str`, *optional*):
                Generation mode indicated which modality to output and can be one of `["text", "image", "audio"]`.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[Union[str, PIL.Image.Image]]`: The decoded text or generated image.
        """
        if generation_mode is None or generation_mode == "text":
            return self.post_process_image_text_to_text(
                generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs
            )

        elif generation_mode == "image":
            images = self.postprocess(generated_outputs, return_tensors="PIL.Image.Image")
            return images["pixel_values"]

        else:
            raise ValueError(
                f"{self.__class__.__name__} got an unexpected generation_mode={generation_mode}. Supported options are only `text` and `image"
            )


__all__ = ["Emu3Processor"]
