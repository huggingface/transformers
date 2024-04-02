# coding=utf-8
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
Processor class for IDEFICS2.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from packaging import version

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import AddedToken, BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, logging


if TYPE_CHECKING:
    from .pipelines.conversational import Conversation


logger = logging.get_logger(__name__)


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)


def build_string_from_input(prompt, image_seq_len, bos_token, image_token, fake_image_token):
    """
    Builds a string from the input prompt and image tokens.

    For example, for the call:

    build_string_from_input(
        prompt=["Initial str", img1, img2, "mid str", img3],
        image_seq_len=2,
        bos_token="<s>",
        image_token="<im>",
        fake_image_token="<fake>"
    )

    The output will be:

    "<s>Initial str<fake><im><im><fake><im><im><fake>mid str<fake><im><im><fake>"

    Args:
        prompt (`List[Union[str, ImageInput]]`): The input prompt.
        image_seq_len (`int`): The length of the image sequence.
        bos_token (`str`): The beginning of sentence token.
        image_token (`str`): The image token.
        fake_image_token (`str`): The fake image token.
    """
    input_strings = []
    input_strings.append(f"{bos_token}")
    open_image_tag = False
    for elem in prompt:
        if is_valid_image(elem):
            input_strings.append(f"{fake_image_token}{image_token * image_seq_len}")
            open_image_tag = True
        else:
            if open_image_tag:
                input_strings.append(f"{fake_image_token}")
                open_image_tag = False
            input_strings.append(elem)
    if open_image_tag:
        input_strings.append(f"{fake_image_token}")
    return "".join(input_strings)


class Idefics2Processor(ProcessorMixin):
    r"""
    Constructs a IDEFICS2 processor which wraps a LLama tokenizer and IDEFICS2 image processor into a single processor.

    [`IdeficsProcessor`] offers all the functionalities of [`Idefics2ImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`Idefics2ImageProcessor`):
            An instance of [`Idefics2ImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`, *optional*):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
        image_seq_len (`int`, *optional*, defaults to 64):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
            This parameter is used to build the string from the input prompt and image tokens and should match the
            config.perceiver_config.resampler_n_latents value for the model used.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Idefics2ImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 64, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True)
        self.image_token = AddedToken("<image>", normalized=False, special=True)
        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True)
        self.image_seq_len = image_seq_len

        tokens_to_add = {
            "additional_special_tokens": [self.fake_image_token, self.image_token, self.end_of_utterance_token]
        }
        tokenizer.add_special_tokens(tokens_to_add)

        # Stores a Jinja template that formats chat histories into tokenizable strings
        self.chat_template = kwargs.pop("chat_template", None)

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        prompts: Union[List[Union[TextInput, ImageInput]], List[List[Union[TextInput, ImageInput]]]],
        image_seq_len: Optional[int] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchEncoding:
        """ """
        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len

        if _is_str_or_image(prompts):
            prompts = [[prompts]]
        elif isinstance(prompts, list) and _is_str_or_image(prompts[0]):
            prompts = [prompts]
        elif isinstance(prompts, list) and isinstance(prompts[0], list) and _is_str_or_image(prompts[0][0]):
            pass
        else:
            raise ValueError(
                "Invalid input prompts. Please provide a string or image, a list of strings and images or "
                "a list of list of strings and images."
            )

        # Build the string from the input prompt and image tokens
        prompt_strings = [
            build_string_from_input(
                prompt=prompt,
                image_seq_len=image_seq_len,
                bos_token=self.tokenizer.bos_token,
                image_token=self.image_token.content,
                fake_image_token=self.fake_image_token.content,
            )
            for prompt in prompts
        ]

        inputs = BatchFeature()
        text_inputs = self.tokenizer(
            text=prompt_strings,
            add_special_tokens=False,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        inputs.update(text_inputs)

        # Extract the images from the prompts, loading them if necessary
        images = []
        for prompt in prompts:
            for elem in prompt:
                if is_valid_image(elem):
                    images.append(elem)
                elif is_url(elem):
                    images.append(load_image(elem))

        image_inputs = self.image_processor(images, return_tensors=return_tensors)
        inputs.update(image_inputs)

        return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def bad_words_ids(self):
        return [[x] for x in self.tokenizer.additional_special_tokens_ids]

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # Copied from transformers.tokenization_utils_base.PreTrainedTokenizerBase.apply_chat_template
    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], "Conversation"],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        **tokenizer_kwargs,
    ) -> Union[str, List[int]]:
        """
        Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids.

        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.apply_chat_template`]. Please
        refer to the docstring of this method for more information.

        for use with chat models, and will read the tokenizer's chat_template attribute to
        determine the format and control tokens to use when converting. When chat_template is None, it will fall back
        to the default_chat_template specified at the class level.

        Args:
            conversation (Union[List[Dict[str, str]], "Conversation"]): A Conversation object or list of dicts
                with "role" and "content" keys, representing the chat history so far.
            chat_template (str, *optional*): A Jinja template to use for this conversion. If
                this is not passed, the model's default chat template will be used instead.
            add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate
                the start of an assistant message. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, defaults to `False`):
                Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            **tokenizer_kwargs: Additional kwargs to pass to the tokenizer.

        Returns:
            `List[int]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`.
        """

        if hasattr(conversation, "messages"):
            # Indicates it's a Conversation object
            conversation = conversation.messages

        # priority: `chat_template` argument > `tokenizer.chat_template` > `tokenizer.default_chat_template`
        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                chat_template = self.default_chat_template

        # Compilation function uses a cache to avoid recompiling the same template
        compiled_template = self._compile_jinja_template(chat_template)

        # Ignore copy
        rendered = compiled_template.render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            image_tokens=self.image_token.content * self.image_seq_len,
            **self.tokenizer.special_tokens_map,
        )
        # We do a hack here - it's not possible to have the same if/else logic in Jinja to match build_string_from_input so
        # we just remove cases when <fake_image_token> has been added twice in a row
        rendered = rendered.replace(
            f"{self.fake_image_token.content}{self.fake_image_token.content}", f"{self.fake_image_token.content}"
        )

        if padding is True:
            padding = "max_length"  # There's only one sequence here, so "longest" makes no sense
        if tokenize:
            if return_dict:
                # Ignore copy
                return self.tokenizer(
                    rendered,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=False,
                    return_tensors=return_tensors,
                    **tokenizer_kwargs,
                )
            else:
                # Ignore copy
                return self.tokenizer.encode(
                    rendered,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=False,
                    return_tensors=return_tensors,
                    **tokenizer_kwargs,
                )
        else:
            return rendered

    @lru_cache
    # Copied from transformers.tokenization_utils_base.PreTrainedTokenizerBase._compile_jinja_template
    def _compile_jinja_template(self, chat_template):
        try:
            import jinja2
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise ImportError("apply_chat_template requires jinja2 to be installed.")

        if version.parse(jinja2.__version__) <= version.parse("3.0.0"):
            raise ImportError(
                "apply_chat_template requires jinja2>=3.0.0 to be installed. Your version is " f"{jinja2.__version__}."
            )

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        # Ignore copy
        jinja_env.filters["is_image"] = is_image_or_image_url
        return jinja_env.from_string(chat_template)

    @property
    def default_chat_template(self):
        """
        This template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content can be a single string or a list of strings and images.
        * If the content element is an image, the template will output a sequence of <image> tokens and <fake_token_around_image> token before and after each image
        * The template will output an <end_of_utterance> token at the end of each message.

        Example:

        ```python
        messages = [
            {"role": "user", "content": ["What is in this Image?", image1, "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"]},
            {"role": "assistant", "content": "This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground."},
            {"role": "user", "content": ["And who is that?"]},
        ]
        ```

        Will create outputs like:
        ```
        User:What is in this Image?<fake_token_around_image><image><image><image><fake_token_around_image><image><image><image><end_of_utterance>
        Assistant:This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>
        User:And who is that?<end_of_utterance>
        Assistant:
        ```
        """
        logger.warning_once(
            "\nNo chat template is defined for this processor - using a default chat template "
            "that implements the ChatML format (without BOS/EOS tokens!). If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # fmt: off
        return (
            "{{ bos_token }}"
            "{% for message in messages %}"
                "{% if message is iterable and message is not string %}"
                    "{{message['role'].capitalize() + ':'}}"
                    "{% for content_elem in message.content %}"
                        "{% if content_elem | is_image %}"
                            "{{'<fake_token_around_image>' + image_tokens + '<fake_token_around_image>'}}"
                        "{% else %}"
                            "{{content_elem}}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<end_of_utterance>\n"
                "{% else %}"
                    "{{message['role'].capitalize() + ':' + message['content'] + '<end_of_utterance>' + '\n'}}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ 'Assistant:' }}"
            "{% endif %}"
        )
        # fmt: on
