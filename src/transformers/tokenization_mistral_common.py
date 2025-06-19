import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub

from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    TextInput,
)
from transformers.utils import logging, to_py_obj
from transformers.utils.hub import PushToHubMixin
from transformers.utils.import_utils import is_tf_available, is_torch_available


if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf
    import numpy as np


logger = logging.get_logger(__name__)


class MistralCommonTokenizer(PushToHubMixin):
    """
    Class to wrap mistral-common tokenizers.

    This class is a wrapper around a `mistral_common.tokens.tokenizers.mistral.MistralTokenizer`.
    It provides a Hugging Face compatible interface to tokenize using the official mistral-common tokenizer.

    Handles several methods for tokenization as well as methods for
    downloading/caching/loading pretrained tokenizers.
    """
    def __init__(
        self, tokenizer_path: Union[str, os.PathLike, Path], mode: ValidationMode = ValidationMode.test, **kwargs
    ):
        """
        Initiales a mistral-common tokenizer.

        Args:
            tokenizer_path (`Union[str, os.PathLike, Path]`):
                Path to the tokenizer file.
            mode (`ValidationMode`, *optional*, defaults to `ValidationMode.test`):
                Validation mode for the tokenizer.
            **kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonTokenizer`.
                Will raise an error if used.
        """
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported to init `MistralCommonTokenizer`.")

        self._tokenizer_path = Path(tokenizer_path)
        self._tokenizer: MistralTokenizer = MistralTokenizer.from_file(str(self._tokenizer_path), mode=mode)

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        This is a lossy conversion. There may be multiple token ids that decode to the same
        string due to partial UTF-8 byte sequences being converted to �.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        return {token: idx for idx, token in enumerate(self._tokenizer.instruct_tokenizer.tokenizer.vocab())}

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        `int`: Size of the vocabulary.
        """
        return self._tokenizer.instruct_tokenizer.tokenizer.n_words

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size

    def encode(
        self,
        text: TextInput,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> List[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to add special tokens when encoding the sequences. This will use the underlying
                `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
                automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
                automatically.
            **kwargs (additional keyword arguments):
                Not supported by `MistralCommonTokenizer.encode`.
                Will raise an error if used.
        """
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.encode`.")

        encoded_text = self._tokenizer.instruct_tokenizer.tokenizer.encode(
            text, bos=add_special_tokens, eos=add_special_tokens
        )

        return encoded_text

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        mode: ValidationMode = ValidationMode.test,
        revision: str = "main",
        **kwargs,
    ):
        if init_inputs:
            raise ValueError("`init_inputs` are not supported by `MistralCommonTokenizer.from_pretrained`.")

        # Handle kwargs and AutoTokenizer case
        if kwargs and not kwargs.keys() == {"_from_auto"}:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.from_pretrained`."
            )

        if not os.path.isfile(pretrained_model_name_or_path):
            tokenizer_path = download_tokenizer_from_hf_hub(
                repo_id=pretrained_model_name_or_path,
                cache_dir=cache_dir,
                token=token,
                revision=revision,
                force_download=force_download,
                local_files_only=local_files_only,
            )
        else:
            tokenizer_path = pretrained_model_name_or_path

        return cls(tokenizer_path=tokenizer_path, mode=mode)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike, Path],
        push_to_hub: bool = False,
        token: Optional[Union[str, bool]] = None,
        commit_message: Optional[str] = None,
        repo_id: Optional[str] = None,
        private: Optional[bool] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str]:
        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.save_pretrained`."
            )

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        shutil.copy(self._tokenizer_path, save_directory)

        if push_to_hub:
            repo_id = repo_id or str(save_directory).split(os.path.sep)[-1]
            repo_id = self._create_repo(
                repo_id, token=token, private=private, repo_url=repo_url, organization=organization
            )
            files_timestamps = self._get_files_timestamps(save_directory)

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

        return (str(save_directory / self._tokenizer_path.name),)

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonTokenizer.decode`.
                Will raise an error if used.

        Returns:
            `str`: The decoded sentence.
        """
        # Convert inputs to python lists
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.decode`.")

        token_ids = to_py_obj(token_ids)

        special_token_policy = SpecialTokenPolicy.IGNORE if skip_special_tokens else SpecialTokenPolicy.KEEP

        decoded_string = self._tokenizer.decode(token_ids, special_token_policy=special_token_policy)
        if clean_up_tokenization_spaces:
            decoded_string = PreTrainedTokenizerBase.clean_up_tokenization(decoded_string)

        return decoded_string

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonTokenizer.batch_decode`.
                Will raise an error if used.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            for seq in sequences
        ]

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        tools: Optional[List[Union[Dict, Callable]]] = None,
        continue_final_message: bool = False,
        tokenize: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> Union[str, List[int], List[str], List[List[int]]]:
        """
        Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids. This method is intended for use with chat models.

        Args:
            conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): A list of dicts
                with "role" and "content" keys, representing the chat history so far.
            tools (`List[Union[Dict, Callable]]`, *optional*):
                A list of tools (callable functions) that will be accessible to the model. If the template does not
                support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
                giving the name, description and argument types for the tool. See our
                [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
                for more information.
            continue_final_message (bool, *optional*):
                If this is set, the chat will be formatted so that the final
                message in the chat is open-ended, without any EOS tokens. The model will continue this message
                rather than starting a new one. This allows you to "prefill" part of
                the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            kwargs (additional keyword arguments, *optional*):
                Not supported by `MistralCommonTokenizer.apply_chat_template`.
                Will raise an error if used.

        Returns:
            `Union[str, List[int], List[str], List[List[int]]]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`.
        """
        if kwargs:
            raise ValueError(
                f"Kwargs {list(kwargs.keys())} are not supported by `MistralCommonTokenizer.apply_chat_template`."
            )

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        outputs = []
        for conversation in conversations:
            truncation = truncation if tokenize else False  # matching the behavior of the HF tokenizer
            chat_request = ChatCompletionRequest.from_openai(
                messages=conversation,
                tools=tools,
                truncate_for_context_length=truncation,
                continue_final_message=continue_final_message,
            )

            max_length = max_length if tokenize else None  # matching the behavior of the HF tokenizer
            tokenized_request = self._tokenizer.encode_chat_completion(chat_request, max_model_input_len=max_length)
            if tokenize:
                outputs.append(tokenized_request.tokens)
            else:
                outputs.append(tokenized_request.text)

        if not is_batched:
            outputs = outputs[0]

        return outputs
