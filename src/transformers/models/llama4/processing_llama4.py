# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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


from typing import Optional, Union

from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, make_flat_list_of_images
from ...utils import auto_docstring


class Llama4ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
        },
    }


chat_template = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}    \n    {%- if messages[0]['content'] is string %}\n        {%- set system_message = messages[0]['content']|trim %}\n    {%- else %}\n        {#- FIXME: The processor requires an array, always. #}\n        {%- set system_message = messages[0]['content'][0]['text']|trim %}\n    {%- endif %}\n    {%- set messages = messages[1:] %}\n    {%- set user_supplied_system_message = true %}\n{%- else %}\n    {%- set system_message = \"\" %}\n    {%- set user_supplied_system_message = false %}\n{%- endif %}\n\n{#- System message if the user supplied one #}\n{%- if user_supplied_system_message %}\n    {{- \"<|header_start|>system<|header_end|>\n\n\" }}\n    {%- if tools is not none %}\n        {{- \"Environment: ipython\n\" }}\n    {%- endif %}\n    {%- if tools is not none and not tools_in_user_message %}\n        {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n        {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n        {{- \"Do not use variables.\n\n\" }}\n        {%- for t in tools %}\n            {{- t | tojson(indent=4) }}\n            {{- \"\n\n\" }}\n        {%- endfor %}\n    {%- endif %}\n    {{- system_message }}\n    {{- \"<|eot|>\" }}\n{%- endif %}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|header_start|>user<|header_end|>\n\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\n\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\n\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\n\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n    {{- '<|header_start|>' + message['role'] + '<|header_end|>\n\n' }}\n        {%- if message['content'] is string %}\n            {{- message['content'] }}\n        {%- else %}\n            {%- for content in message['content'] %}\n                {%- if content['type'] == 'image' %}\n                    {{- '<|image|>' }}\n                {%- elif content['type'] == 'text' %}\n                    {{- content['text'] }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- \"<|eot|>\" }}\n    {%- elif 'tool_calls' in message and message.tool_calls|length > 0 %}\n       {{- '<|header_start|>assistant<|header_end|>\n\n' -}}\n       {{- '<|python_start|>' }}\n        {%- if message['content'] is string %}\n            {{- message['content'] }}\n        {%- else %}\n            {%- for content in message['content'] %}\n                {%- if content['type'] == 'image' %}\n                    {{- '<|image|>' }}\n                {%- elif content['type'] == 'text' %}\n                    {{- content['text'] }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n       {{- '<|python_end|>' }}\n        {%- for tool_call in message.tool_calls %}\n           {{- '{\"name\": \"' + tool_call.function.name + '\", ' }}\n           {{- '\"parameters\": ' }}\n           {{- tool_call.function.arguments | tojson }}\n           {{- \"}\" }}\n        {%- endfor %}\n       {{- \"<|eot|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|header_start|>ipython<|header_end|>\n\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|header_start|>assistant<|header_end|>\n\n' }}\n{%- endif %}\n"


@auto_docstring
class Llama4Processor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 14,
        pixel_shuffle_ratio: float = 0.5,
        fake_image_token="<|image|>",
        image_token="<|image|>",
        start_of_image_token="<|image_start|>",
        end_of_image_token="<|image_end|>",
        patch_token="<|patch|>",
        tile_x_separator_token="<|tile_x_separator|>",
        tile_y_separator_token="<|tile_y_separator|>",
        chat_template=chat_template,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to 28):
            The size of image patches for tokenization.
        pixel_shuffle_ratio (`float`, *optional*, defaults to `0.5`):
            The ratio used for pixel shuffling when processing images. This controls the downsampling factor
            applied to image patches. The actual downsampling ratio is calculated as `1 / (pixel_shuffle_ratio^2)`.
        fake_image_token (`str`, *optional*, defaults to `"<|image|>"`):
            The placeholder token in the text that will be replaced with actual image tokens. This token serves
            as a marker indicating where images should be inserted in the text sequence.
        image_token (`str`, *optional*, defaults to `"<|image|>"`):
            The token to be used to represent an image in the text.
        start_of_image_token (`str`, *optional*, defaults to `"<|image_start|>"`):
            The special token that marks the beginning of an image sequence in the text. This token is prepended
            to image token sequences to delimit image boundaries.
        end_of_image_token (`str`, *optional*, defaults to `"<|image_end|>"`):
            The special token that marks the end of an image sequence in the text. This token is appended to
            image token sequences to delimit image boundaries.
        patch_token (`str`, *optional*, defaults to `"<|patch|>"`):
            The token used to represent individual image patches. Multiple patch tokens are used to represent
            the full image, with the number depending on the image size and patch configuration.
        tile_x_separator_token (`str`, *optional*, defaults to `"<|tile_x_separator|>"`):
            The token used to separate tiles (patches) horizontally within an image. This token is inserted
            between patches in the same row when images are split into multiple tiles.
        tile_y_separator_token (`str`, *optional*, defaults to `"<|tile_y_separator|>"`):
            The token used to separate tiles (patches) vertically within an image. This token is inserted
            between rows of patches when images are split into multiple tiles.
        """
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.downsample_ratio = int(round(1.0 / (pixel_shuffle_ratio**2)))
        self.patch_size = patch_size

        self.fake_image_token = fake_image_token
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.start_of_img_token = start_of_image_token
        self.end_of_img_token = end_of_image_token
        self.img_patch_token = patch_token
        self.tile_token = tile_x_separator_token
        self.tile_global_token = tile_y_separator_token

    def _prompt_split_image(self, aspect_ratio, num_patches_per_chunk):
        """
        Create a structured string representation of image tokens

        Args:
           num_patches: Number of patches in the image

        Returns:
            String with appropriate image tokens
        """
        img_string = "<|image_start|>"
        ratio_h, ratio_w = aspect_ratio
        if ratio_h * ratio_w > 1:
            for yy in range(ratio_h):
                for xx in range(ratio_w):
                    img_string += "<|patch|>" * num_patches_per_chunk
                    if xx < ratio_w - 1:
                        img_string += "<|tile_x_separator|>"

                img_string += "<|tile_y_separator|>"

        img_string += "<|image|>"
        img_string += "<|patch|>" * num_patches_per_chunk
        img_string += "<|image_end|>"

        return img_string

    @auto_docstring
    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        **kwargs: Unpack[Llama4ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            Llama4ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, (list, tuple)):
            text = [text]

        # Process images
        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_height, image_width = image_inputs["pixel_values"][0].shape[-2:]
            num_patches_per_chunk = int(
                (image_height // self.patch_size) * (image_width // self.patch_size) // self.downsample_ratio
            )
            aspect_ratios = image_inputs.pop("aspect_ratios")

            total_placeholders = sum(prompt.count(self.fake_image_token) for prompt in text)
            if total_placeholders != len(images):
                raise ValueError(
                    f"Found {total_placeholders} placeholders across the batch, "
                    f"but have {len(images)} flattened images."
                )

            image_index = 0
            processed_text = []
            for prompt in text:
                placeholder_count = prompt.count(self.fake_image_token)
                if placeholder_count == 0:
                    # do nothing if there is no image
                    processed_text.append(prompt)
                    continue
                prompt_splits = prompt.split(self.fake_image_token)
                new_prompt = []
                for local_image_index, split_part in enumerate(prompt_splits):
                    new_prompt.append(split_part)
                    if local_image_index < placeholder_count:
                        tokens_for_this_image = self._prompt_split_image(
                            aspect_ratios[image_index], num_patches_per_chunk
                        )
                        image_index += 1
                        new_prompt.append(tokens_for_this_image)
                processed_text.append("".join(new_prompt))

            if image_index != len(images):
                raise ValueError("Number of image placeholders in the prompt does not match the number of images.")

            text = processed_text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)


__all__ = ["Llama4Processor"]
