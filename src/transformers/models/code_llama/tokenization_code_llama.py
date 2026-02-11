# Copyright 2023 The HuggingFace Inc. team.
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


from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}

SPIECE_UNDERLINE = "▁"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class CodeLlamaTokenizer(TokenizersBackend):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no normalization.

    ```python
    >>> from transformers import CodeLlamaTokenizer

    >>> tokenizer = CodeLlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    >>> tokenizer.encode("Hello this is a test")
    [1, 15043, 445, 338, 263, 1243]
    ```

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.


    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods. The default configuration match that of
    [meta-llama/CodeLlama-7b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json)
    which supports prompt infilling.

    Args:
        clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`):
            Whether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra
            spaces.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        prefix_token (`str`, *optional*, defaults to `"▁<PRE>"`):
            Prefix token used for infilling.
        middle_token (`str`, *optional*, defaults to `"▁<MID>"`):
            Middle token used for infilling.
        suffix_token (`str`, *optional*, defaults to `"▁<SUF>"`):
            Suffix token used for infilling.
        eot_token (`str`, *optional*, defaults to `"▁<EOT>"`):
            End of text token used for infilling.
        fill_token (`str`, *optional*, defaults to `"<FILL_ME>"`):
            The token used to split the input between the prefix and suffix.
        additional_special_tokens (`list[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether to add a beginning of sequence token at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether to add an end of sequence token at the end of sequences.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used.
        add_prefix_space (`bool`, *optional*):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
        vocab (`str`, `dict` or `list`, *optional*):
            Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.
        merges (`str` or `list`, *optional*):
            Custom merges list. If not provided, merges are loaded from merges_file.
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        prefix_token="▁<PRE>",
        middle_token="▁<MID>",
        suffix_token="▁<SUF>",
        eot_token="▁<EOT>",
        fill_token="<FILL_ME>",
        additional_special_tokens=None,
        use_default_system_prompt: bool = False,
        add_prefix_space: bool | None = True,
        add_bos_token: bool = True,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space if add_prefix_space is not None else True
        self.use_default_system_prompt = use_default_system_prompt
        additional_special_tokens = additional_special_tokens or []
        for token in [prefix_token, middle_token, suffix_token, eot_token, fill_token]:
            additional_special_tokens += [token] if token is not None else []

        self._vocab = (
            vocab
            if vocab is not None
            else {
                str(unk_token): 0,
                str(bos_token): 1,
                str(eos_token): 2,
            }
        )

        self._merges = merges or []
        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                fuse_unk=True,
                byte_fallback=True,
                dropout=None,
                unk_token=str(unk_token),
            )
        )
        prepend_scheme = "first" if self.add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme=prepend_scheme, split=False
        )

        self._tokenizer.decoder = decoders.Sequence(
            [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(content=" ", left=1)]
        )

        super().__init__(
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            use_default_system_prompt=use_default_system_prompt,
            add_prefix_space=add_prefix_space,
            prefix_token=prefix_token,
            middle_token=middle_token,
            suffix_token=suffix_token,
            eot_token=eot_token,
            fill_token=fill_token,
            add_bos_token=add_bos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self._prefix_token = prefix_token
        self._middle_token = middle_token
        self._suffix_token = suffix_token
        self._eot_token = eot_token
        self.fill_token = fill_token

    @property
    def prefix_token(self):
        return self._prefix_token

    @property
    def prefix_id(self):
        if self._prefix_token is None:
            return None
        return self.convert_tokens_to_ids(self.prefix_token)

    @property
    def middle_token(self):
        return self._middle_token

    @property
    def middle_id(self):
        if self._middle_token is None:
            return None
        return self.convert_tokens_to_ids(self.middle_token)

    @property
    def suffix_token(self):
        return self._suffix_token

    @property
    def suffix_id(self):
        if self._suffix_token is None:
            return None
        return self.convert_tokens_to_ids(self.suffix_token)

    @property
    def eot_id(self):
        if self._eot_token is None:
            return None
        return self.convert_tokens_to_ids(self.eot_token)

    @property
    def eot_token(self):
        return self._eot_token

    def set_infilling_processor(self, reset, suffix_first=False, add_special_tokens=True):
        """
        Updates the normalizer to make sure the prompt format for `infilling` is respected. The infilling format is the
        following: if suffix_first
            " <PRE> <SUF>{suf} <MID> {pre}"
        else:
            " <PRE> {pre} <SUF>{suf} <MID>"

        If `reset` is set to `True`, the `normalizer` and `post_processor` are reset to their "normal" behaviour, which
        is to add a prefix space for the normalizer, and add a `bos_token` to the input text for the `post_processor`.
        """
        if reset:
            self._tokenizer.normalizer = normalizers.Sequence(
                [
                    normalizers.Prepend(prepend="▁"),
                    normalizers.Replace(pattern=" ", content="▁"),
                ]
            )
            self.update_post_processor()
            return

        self._tokenizer.normalizer = normalizers.Replace(pattern=" ", content="▁")
        pair = [self.bos_token] if self.add_bos_token and add_special_tokens else []
        special_tokens = [(self.bos_token, self.bos_token_id)] if self.add_bos_token and add_special_tokens else []
        if suffix_first:
            # format as " <PRE> <SUF>{suf} <MID> {pre}"
            pair += [self.prefix_token, self.suffix_token, "$B", self.middle_token, "$A"]
            special_tokens += [
                (self.prefix_token, self.prefix_id),
                (self.suffix_token, self.suffix_id),
                (self.middle_token, self.middle_id),
            ]
        else:
            # format as " <PRE> {pre} <SUF>{suf} <MID>"
            pair += [self.prefix_token, "$A", self.suffix_token, "$B", self.middle_token]
            special_tokens += [
                (self.prefix_token, self.prefix_id),
                (self.suffix_token, self.suffix_id),
                (self.middle_token, self.middle_id),
            ]

        if self.add_eos_token and add_special_tokens:
            pair += [self.eos_token]
            special_tokens += [(self.eos_token, self.eos_token_id)]
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single="$A", pair=pair, special_tokens=special_tokens
        )

    def tokenize(self, text, suffix=None, suffix_first=False, **kwargs):
        # Handle fill_token splitting
        if self.fill_token is not None and self.fill_token in text and suffix is None:
            text, suffix = text.split(self.fill_token)

        # If no suffix, use standard tokenization
        if suffix is None or len(suffix) < 1:
            return super().tokenize(text, **kwargs)

        # Check that infilling tokens are available
        if None in (self.prefix_id, self.middle_id, self.suffix_id):
            raise ValueError(
                "The input either includes a `prefix` and a `suffix` used for the infilling task,"
                f"  or can be split on the {self.fill_token} token, creating a suffix and prefix,"
                " but the model does not support `infilling`."
            )

        # Temporarily set infilling processor
        self.set_infilling_processor(False, suffix_first=suffix_first, add_special_tokens=False)

        # Remove text_pair and pair from kwargs if present to avoid conflict
        kwargs.pop("text_pair", None)
        kwargs.pop("pair", None)

        # Tokenize with infilling format
        # The processor will handle the special token arrangement
        # Use pair=suffix (not text_pair) since base class tokenize expects 'pair' parameter
        result = super().tokenize(" " + text, pair=suffix, **kwargs)

        # Reset processor
        self.set_infilling_processor(True)

        return result

    def _encode_plus(self, text, text_pair=None, suffix=None, suffix_first=False, add_special_tokens=True, **kwargs):
        is_infilling = False

        if suffix is not None:
            text_pair = suffix
            is_infilling = True
        elif "suffix" in kwargs:
            text_pair = kwargs.pop("suffix")
            is_infilling = True

        if isinstance(text, str) and self.fill_token is not None and self.fill_token in text and text_pair is None:
            text, text_pair = text.split(self.fill_token)
            is_infilling = True

        if not is_infilling:
            return super()._encode_plus(text, text_pair=text_pair, add_special_tokens=add_special_tokens, **kwargs)

        if (
            text_pair is None
            or (isinstance(text_pair, str) and len(text_pair) < 1)
            or (isinstance(text_pair, list) and len(text_pair) == 0)
        ):
            return super()._encode_plus(text, text_pair=text_pair, add_special_tokens=add_special_tokens, **kwargs)

        if None in (self.prefix_id, self.middle_id, self.suffix_id):
            raise ValueError(
                "The input includes a `prefix` and a `suffix` used for the infilling task,"
                " the `prefix_id, middle_id, suffix_id` must all be initialized. Current"
                f" values : {self.prefix_id, self.middle_id, self.suffix_id}"
            )

        self.set_infilling_processor(False, suffix_first=suffix_first, add_special_tokens=add_special_tokens)
        kwargs.pop("text_pair", None)

        if isinstance(text, str):
            text = " " + text
        elif isinstance(text, list):
            text = [" " + t if isinstance(t, str) else t for t in text]

        result = super()._encode_plus(text, text_pair=text_pair, add_special_tokens=True, **kwargs)
        self.set_infilling_processor(True)
        return result


__all__ = ["CodeLlamaTokenizer", "CodeLlamaTokenizerFast"]

# Backward alias
CodeLlamaTokenizerFast = CodeLlamaTokenizer
