# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import enum
from typing import Any, Optional, Union, overload

import numpy as np

from ..audio_utils import AudioInput
from ..generation import GenerationConfig
from ..image_utils import ImageInput
from ..processing_utils import ProcessingKwargs, Unpack
from ..utils import (
    add_end_docstrings,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from ..video_utils import VideoInput
from .base import Pipeline, build_pipeline_init_args


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES
    from .pt_utils import KeyDataset

if is_vision_available():
    from PIL import Image

logger = logging.get_logger(__name__)


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


class Chat:
    """This class is intended to just be used internally in this pipeline and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""

    def __init__(self, messages: list[dict]):
        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        self.messages = messages


@add_end_docstrings(build_pipeline_init_args(has_processor=True))
class AnyToAnyPipeline(Pipeline):
    """
    Multimodal Generation pipeline using an `AutoModelForMultimodalLM`. This pipeline generates text given any
    combination of multimodal data and text.When the underlying model is a conversational model, it can also
    accept one or more chats, in which case the pipeline will operate in chat mode and will continue the
    chat(s) by adding its response(s). Each chat takes the form of a list of dicts, where each dict contains
    "role" and "content" keys.

    Unless the model you're using explicitly sets these generation parameters in its configuration files
    (`generation_config.json`), the following default values will be used:
    - max_new_tokens: 256

    Example:

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline(task="any-to-any", model="google/gemma-3n-E4B-it")
    >>> pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", text="A photo of")
    [{'generated_text': 'a photo of two birds'}]
    ```

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline("any-to-any", model="google/gemma-3n-E4B-it")
    >>> messages = [
    >>>     {
    >>>         "role": "user",
    >>>         "content": [
    >>>             {
    >>>                 "type": "image",
    >>>                 "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    >>>             },
    >>>             {"type": "text", "text": "Describe this image."},
    >>>         ],
    >>>     },
    >>>     {
    >>>         "role": "assistant",
    >>>         "content": [
    >>>             {"type": "text", "text": "There is a dog and"},
    >>>         ],
    >>>     },
    >>> ]
    >>> pipe(text=messages, max_new_tokens=20, return_full_text=False)
    [{'input_text': [{'role': 'user',
        'content': [{'type': 'image',
        'url': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
        {'type': 'text', 'text': 'Describe this image.'}]},
    {'role': 'assistant',
        'content': [{'type': 'text', 'text': 'There is a dog and'}]}],
    'generated_text': ' a person in the image. The dog is sitting on the sand, and the person is sitting on'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This multimodal pipeline can currently be loaded from pipeline() using the following task identifier:
    "any-to-any".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=any-to-any).
    """

    _load_processor = True
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False

    _pipeline_calls_generate = True
    # Make sure the docstring is updated when the default generation config is changed
    _default_generation_config = GenerationConfig(
        max_new_tokens=256,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        requires_backends(self, "librosa")
        requires_backends(self, "torchvision")
        self.check_model_type(MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES)

    def _sanitize_parameters(
        self,
        max_new_tokens=None,
        generate_kwargs=None,
        timeout=None,
        return_full_text=None,
        return_tensors=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        stop_sequence=None,
        continue_final_message=None,
        skip_special_tokens=None,
        generation_mode=None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        forward_kwargs = {}
        preprocess_params = {}
        postprocess_params = {}

        # Preprocess params
        preprocess_params.update(kwargs)
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        if continue_final_message is not None:
            preprocess_params["continue_final_message"] = continue_final_message

        # Forward kwargs
        forward_kwargs["generate_kwargs"] = generate_kwargs or {}
        if generation_mode is not None and generation_mode != "text":
            forward_kwargs["generate_kwargs"]["generation_mode"] = generation_mode
        if stop_sequence is not None:
            if isinstance(stop_sequence, str):
                stop_sequence = [stop_sequence]
            forward_kwargs["generate_kwargs"]["stop_strings"] = stop_sequence
            forward_kwargs["generate_kwargs"]["tokenizer"] = self.processor.tokenizer

        if max_new_tokens is not None:
            if generate_kwargs is not None and "max_new_tokens" in generate_kwargs:
                raise ValueError(
                    "'max_new_tokens' is defined twice, once in 'generate_kwargs' and "
                    "once as a direct argument. Please use only one."
                )
            forward_kwargs["generate_kwargs"]["max_new_tokens"] = max_new_tokens

        if return_full_text is not None and return_type is None:
            if return_tensors is not None:
                raise ValueError("`return_full_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        elif return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS
        # We don't want to set the global default to FULLTEXT at init time. That is why
        # `_postprocess_params` is checked before setting the default value
        elif return_type is None and generation_mode in [None, "text"] and hasattr(self, "_postprocess_params"):
            return_type = ReturnType.FULL_TEXT

        # Postprocess params
        if generation_mode not in [None, "text"] and return_type is not None:
            raise ValueError(
                f"`return_type` cannot be set to {return_type} when generation_mode={generation_mode}. "
                "Set `return_type=None` or generation_mode='text'"
            )
        if generation_mode not in [None, "text", "image", "audio"]:
            raise ValueError(
                f"`generation_mode` can be only one of the `text`, `audio`, `image` but got generation_mode[={generation_mode}]"
            )

        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if continue_final_message is not None:
            postprocess_params["continue_final_message"] = continue_final_message
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
        if skip_special_tokens is not None:
            postprocess_params["skip_special_tokens"] = skip_special_tokens
        postprocess_params["generation_mode"] = generation_mode
        return preprocess_params, forward_kwargs, postprocess_params

    @overload
    def __call__(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, "Image.Image"]] = None,
        videos: Optional[Union[str, "np.ndarray", "torch.Tensor"]] = None,
        audio: Optional[Union[str, "np.ndarray"]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]: ...

    @overload
    def __call__(
        self,
        text: Optional[list[str]] = None,
        images: Optional[Union[list[str], list["Image.Image"]]] = None,
        videos: Optional[Union[list[str], list["np.ndarray"], list["torch.Tensor"]]] = None,
        audio: Optional[Union[list[str], list["np.ndarray"]]] = None,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]: ...

    def __call__(
        self,
        text: Union[str, list[str], list[dict]],
        images: Optional[
            Union[
                str,
                list[str],
                list[list[str]],
                ImageInput,
            ]
        ] = None,
        videos: Optional[Union[str, list[str], VideoInput]] = None,
        audio: Optional[Union[str, list[str], AudioInput]] = None,
        **kwargs,
    ) -> Union[list[dict[str, Any]], list[list[dict[str, Any]]]]:
        """
        Generate a text given text and optionally multimodal data passed as inputs.

        Args:
            text (`str`, `list[str]`, `list[dict]`):
                The text to be used for generation. If a list of strings is passed, the length of the list should be
                the same as the number of images. Text can also follow the chat format: a list of dictionaries where
                each dictionary represents a message in a conversation. Each dictionary should have two keys: 'role'
                and 'content'. 'role' should be one of 'user', 'system' or 'assistant'. 'content' should be a list of
                dictionary containing the text of the message and the type of the message.
            images (`str`, `list[str]`, `ImageInput`):
                The pipeline handles three types of images:

                - A string containing a HTTP(s) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Finally, this pipeline also supports
                the chat format (see `text`) containing images and text in this argument.
            videos (`str`, `list[str]`, `VideoInput`):
                The pipeline handles three types of videos:

                - A string containing a HTTP(s) link pointing to a video
                - A string containing a local path to a video
                - A video loaded and decoded to array format

                The pipeline accepts either a single video or a batch of videos. Finally, this pipeline also supports
                the chat format (see `text`) containing videos and text in this argument.
            audio (`str`, `list[str]`, `AudioInput`):
                The pipeline handles three types of audios:

                - A string containing a HTTP(s) link pointing to an audio
                - A string containing a local path to an audio
                - An audio loaded in PIL directly

                The pipeline accepts either a single audios or a batch of audios. Finally, this pipeline also supports
                the chat format (see `text`) containing audios and text in this argument.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Returns the tensors of predictions (as token indices) in the outputs. If set to
                `True`, the decoded text is not returned.
            return_text (`bool`, *optional*):
                Returns the decoded texts in the outputs.
            return_full_text (`bool`, *optional*, defaults to `True`):
                If set to `False` only added text is returned, otherwise the full text is returned. Cannot be
                specified at the same time as `return_text`.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the potential extra spaces in the text output.
            continue_final_message( `bool`, *optional*): This indicates that you want the model to continue the
                last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
                By default this is `True` when the final message in the input chat has the `assistant` role and
                `False` otherwise, but you can manually override that behaviour by setting this flag.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key (cannot
            return a combination of both `generated_text` and `generated_token_ids`):

            - **generated_text** (`str`, present when `return_text=True` and `generation_mode="text") -- The generated text.
            - **generated_audio** (`np.ndarray`, present when `generation_mode="audio") -- The generated audio.
            - **generated_image** (`PIL.Image.Image`, present when `generation_mode="image") -- The generated image.
            - **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True` and `generation_mode="text") -- The token
                ids of the generated text.
            - **input_text** (`str`) -- The input text.
        """
        if images is None and text is None:
            raise ValueError("You must at least provide either text or images.")

        # Do we need this codepath ???
        if text is not None and not (isinstance(text, str) or (isinstance(text, list) and isinstance(text[0], str))):
            """
            Supports the following format
            - {"text": text, "image": image, "video": video, "audio": audio}
            - [{"text": text, "image": image, "video": video, "audio": audio}]
            - Generator and datasets
            This is a common pattern in other multimodal pipelines, so we support it here as well.
            """
            return super().__call__(text, **kwargs)

        if isinstance(text, (list, tuple, KeyDataset)) and isinstance(text[0], (list, tuple, dict)):
            # We have one or more prompts in list-of-dicts format, so this is chat mode
            if isinstance(text[0], dict):
                return super().__call__(Chat(text), **kwargs)
            else:
                chats = [Chat(chat) for chat in text]  # 🐈 🐈 🐈
                return super().__call__(chats, **kwargs)

        # encourage the user to use the chat format if supported
        if getattr(self.processor, "chat_template", None) is not None:
            logger.warning_once(
                "The input data was not formatted as a chat with dicts containing 'role' and 'content' keys, even "
                "though this model supports chat. Consider using the chat format for better results. For more "
                "information, see https://huggingface.co/docs/transformers/en/chat_templating"
            )

        return super().__call__({"text": text, "images": images, "video": videos, "audio": audio}, **kwargs)

    def preprocess(self, inputs=None, timeout=None, continue_final_message=None, **processing_kwargs):
        if isinstance(inputs, Chat):
            # If the user passes a chat that ends in an assistant message, we treat it as a prefill by default
            # because very few models support multiple separate, consecutive assistant messages
            if continue_final_message is None:
                continue_final_message = inputs.messages[-1]["role"] == "assistant"

            # Handle Mistral tokenizer which does not accept processing kwargs
            chat_template_kwargs = {"add_generation_prompt": not continue_final_message, **processing_kwargs}
            if self.processor.tokenizer.__class__.__name__ == "MistralCommonTokenizer":
                chat_template_kwargs = {
                    k: v for k, v in chat_template_kwargs.items() if k in ["padding", "truncation", "max_length"]
                }

            model_inputs = self.processor.apply_chat_template(
                inputs.messages,
                continue_final_message=continue_final_message,
                return_tensors=self.framework,
                tokenize=True,
                return_dict=True,
                **chat_template_kwargs,
            ).to(dtype=self.dtype)
            model_inputs["text"] = inputs
            return model_inputs

        # In case we only have text inputs
        if isinstance(inputs, (list, tuple, str)):
            text = inputs
            inputs = {}
        else:
            inputs = inputs.copy()  # avoid in-place changes if users passed dict
            text = inputs.pop("text")

            # Feature extractor do not load audio files and expect a decode array
            if "audio" in inputs and hasattr(self.processor, "feature_extractor"):
                inputs["audio"] = self.processor.feature_extractor.fetch_audio(inputs["audio"])

        # If batched text inputs, we set padding to True unless specified otherwise
        if isinstance(text, (list, tuple)) and len(text) > 1:
            processing_kwargs.setdefault("padding", True)

        # Multimodal data is loaded in preprocessors so we pass all ipnuts directly to `self.processor`
        model_inputs = self.processor(text=text, **inputs, return_tensors=self.framework, **processing_kwargs).to(
            dtype=self.dtype
        )
        model_inputs["text"] = text
        return model_inputs

    def _forward(self, model_inputs, generate_kwargs=None):
        generate_kwargs = {} if generate_kwargs is None else generate_kwargs
        prompt_text = model_inputs.pop("text")
        input_ids = model_inputs.get("input_ids", model_inputs.get("decoder_input_ids"))

        # User-defined `generation_config` passed to the pipeline call take precedence
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config

        generated_sequence = self.model.generate(**model_inputs, **generate_kwargs)
        return {"generated_sequence": generated_sequence, "prompt_text": prompt_text, "input_ids": input_ids}

    def postprocess(
        self,
        model_outputs,
        return_type=None,
        continue_final_message=None,
        skip_special_tokens=None,
        **postprocess_kwargs,
    ):
        input_texts = model_outputs["prompt_text"]
        input_texts = [input_texts] if isinstance(input_texts, (str, Chat)) else input_texts
        generated_sequence = model_outputs["generated_sequence"]
        input_ids = model_outputs["input_ids"]
        if return_type == ReturnType.TENSORS:
            return [
                {"input_text": input_texts[i], "generated_token_ids": generated_sequence[i]}
                for i in range(len(input_texts))
            ]

        # Decode inputs and outputs the same way to remove input text from generated text if present
        skip_special_tokens = skip_special_tokens if skip_special_tokens is not None else True
        generation_mode = postprocess_kwargs["generation_mode"] or "text"
        if generation_mode == "image" and hasattr(self.model, "decode_image_tokens"):
            generated_sequence = self.model.decode_image_tokens(generated_sequence.to(self.model.device))
        generated_outputs = self.processor.post_process_multimodal_output(
            generated_sequence, skip_special_tokens=skip_special_tokens, **postprocess_kwargs
        )

        # Force consistent behavior for including the input text in the output
        if return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
            # Remove the input text from the generated text if the generated text starts with the input text
            # (accounting for the possibility of a space between the input and generated text)
            new_generated_texts = []
            decoded_inputs = self.processor.post_process_image_text_to_text(
                input_ids, skip_special_tokens=skip_special_tokens, **postprocess_kwargs
            )
            for text_generated, decoded_input in zip(generated_outputs, decoded_inputs):
                # There can be added characters before the input text, so we need to find the beginning of the input text in the generated text
                index_input_text = text_generated.find(decoded_input)
                # Limit the search to 2 residual characters, like spaces or new lines, to avoid removing a large part of the answer
                if 0 <= index_input_text <= 2:
                    # If the input text is found, we remove it
                    new_generated_texts.append(text_generated[index_input_text + len(decoded_input) :])
                else:
                    new_generated_texts.append(text_generated)
            generated_outputs = new_generated_texts
        if return_type == ReturnType.FULL_TEXT:
            full_texts = []
            for prompt_text, generated_text in zip(input_texts, generated_outputs):
                if isinstance(prompt_text, str):
                    generated_text = prompt_text + generated_text
                elif isinstance(prompt_text, Chat):
                    if continue_final_message is None:
                        # If the user passes a chat ending in an assistant message, we treat it as a prefill by
                        # default because very few models support multiple separate, consecutive assistant messages
                        continue_final_message = prompt_text.messages[-1]["role"] == "assistant"
                    if continue_final_message:
                        # With assistant prefill, concat onto the end of the last message
                        new_text = dict(prompt_text.messages[-1]["content"][-1].items())
                        new_text["text"] += generated_text
                        generated_text = list(prompt_text.messages)[:-1] + [
                            {
                                "role": prompt_text.messages[-1]["role"],
                                "content": prompt_text.messages[-1]["content"][:-1] + [new_text],
                            }
                        ]
                    else:
                        # When we're not starting from a prefill, the output is a new assistant message
                        generated_text = list(prompt_text.messages) + [
                            {"role": "assistant", "content": generated_text}
                        ]
                full_texts.append(generated_text)
            generated_outputs = full_texts

        records = [
            {
                "input_text": input_text.messages if isinstance(input_text, Chat) else input_text,
                f"generated_{generation_mode}": generated_output,
            }
            for input_text, generated_output in zip(input_texts, generated_outputs)
        ]

        return records
