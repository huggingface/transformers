# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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


from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


class MiniCPMV4_6ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "common_kwargs": {
            "return_tensors": "pt",
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
        },
    }


@auto_docstring
class MiniCPMV4_6Processor(ProcessorMixin):
    def __init__(self, image_processor=None, video_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, video_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.slice_mode = self.image_processor.slice_mode
        self.default_use_image_id = self.image_processor.use_image_id
        self.image_token_divisor = 4 if self.image_processor.downsample_mode == "4x" else 16
        self.video_token_divisor = 4 if self.video_processor.downsample_mode == "4x" else 16

        self.image_token = tokenizer.image_pad_token  # "<|image_pad|>"
        self.video_token = tokenizer.video_pad_token  # "<|video_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

        self.image_start_token = tokenizer.image_start_token  # "<image>"
        self.image_end_token = tokenizer.image_end_token  # "</image>"
        self.video_start_token = tokenizer.video_start_token  # "<video>"
        self.video_end_token = tokenizer.video_end_token  # "</video>"
        self.slice_start_token = tokenizer.slice_start_token  # "<slice>"
        self.slice_end_token = tokenizer.slice_end_token  # "</slice>"
        self.image_id_start_token = tokenizer.image_id_start_token  # "<image_id>"
        self.image_id_end_token = tokenizer.image_id_end_token  # "</image_id>"

        special_tokens = [
            self.image_start_token,
            self.image_end_token,
            self.video_start_token,
            self.video_end_token,
            self.slice_start_token,
            self.slice_end_token,
            self.image_id_start_token,
            self.image_id_end_token,
            self.image_token,
            self.video_token,
        ]

        # TODO: move to conversion script before release, we can't add tokens when init a processor
        tokens_to_add = [
            AddedToken(tok, normalized=False, special=True)
            for tok in special_tokens
            if tok not in self.tokenizer.get_vocab()
        ]
        if tokens_to_add:
            self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[MiniCPMV4_6ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- Token ids to be fed to a model.
            - **attention_mask** -- Mask indicating which tokens should be attended to.
            - **pixel_values** -- Processed image patches to be fed to a model.
            - **target_sizes** -- Patch grid sizes for the vision encoder.
        """

        if isinstance(text, str):
            text = [text]
        text = text.copy()

        output_kwargs = self._merge_kwargs(
            MiniCPMV4_6ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        use_image_id = output_kwargs["images_kwargs"].pop("use_image_id", None)
        use_image_id = use_image_id if use_image_id is not None else self.default_use_image_id

        image_inputs = {}
        # TODO: Check what is the actual pattern in jinja and match it here
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

            index = 0
            image_grids = image_inputs.pop("grids")
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_tokens_per_patch = image_inputs["target_sizes"][index].prod(-1) * self.image_token_divisor
                    num_patch_tokens = num_tokens_per_patch[1:].sum()
                    num_rows, num_cols = image_grids[index]

                    image_placeholder = (
                        self.image_start_token + self.image_token * num_tokens_per_patch[0] + self.image_end_token
                    )
                    if use_image_id:
                        image_placeholder = (
                            f"{self.image_id_start_token}{index}{self.image_id_end_token}" + image_placeholder
                        )

                    if self.slice_mode and num_rows > 0 and num_cols > 0:
                        slice_placeholder = (
                            self.slice_start_token + self.image_token * num_patch_tokens + self.slice_end_token
                        )
                        slices = [slice_placeholder * num_cols for _ in range(num_rows)]
                        image_placeholder += "\n".join(slices)

                    text[i] = text[i].replace(self.image_token, image_placeholder, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        video_inputs = {}
        if videos is not None:
            video_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])

            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_tokens_per_patch = image_inputs["target_sizes"][index].prod(-1) * self.video_token_divisor
                    num_patch_tokens = num_tokens_per_patch[1:].sum()
                    num_rows = num_cols = 1  # FIXME: do we really apply cropping on each frame and possibly subframe?

                    video_placeholder = (
                        self.image_start_token + self.video_token * num_tokens_per_patch[0] + self.image_end_token
                    )
                    if use_image_id:
                        video_placeholder = (
                            f"{self.image_id_start_token}{index}{self.image_id_end_token}" + image_placeholder
                        )

                    if self.slice_mode and num_rows > 0 and num_cols > 0:
                        slice_placeholder = (
                            self.slice_start_token + self.video_token * num_patch_tokens + self.slice_end_token
                        )
                        slices = [slice_placeholder * num_cols for _ in range(num_rows)]
                        video_placeholder += "\n".join(slices)

                    text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].get("return_tensors")
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        texts = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)
        return [t.strip() for t in texts]


__all__ = ["MiniCPMV4_6Processor"]
