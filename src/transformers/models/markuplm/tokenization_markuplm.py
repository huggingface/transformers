# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import add_end_docstrings, logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = r"""
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to encode the sequences with the special tokens relative to their model.
            padding (`bool`, `str` or [`~tokenization_utils_base.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
                `None`, this will use the predefined model maximum length if a maximum length is required by one of the
                truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
                truncation/padding to a maximum length will be deactivated.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing tokens returned when
                `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping tokens.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pretokenized (e.g. split into words). Set this to `True` if you are
                passing pretokenized inputs to avoid additional tokenization.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            return_tensors (`str` or [`~tokenization_utils_base.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
"""


class MarkupLMTokenizer(TokenizersBackend):
    r"""
    Construct a MarkupLM tokenizer. Based on byte-level Byte-Pair-Encoding (BPE).

    [`MarkupLMTokenizer`] can be used to turn HTML strings into to token-level `input_ids`, `attention_mask`,
    `token_type_ids`, `xpath_tags_seq` and `xpath_tags_seq`. This tokenizer inherits from [`TokenizersBackend`] which
    contains most of the main methods and ensures a `tokenizers` backend is always instantiated.

    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab (`str` or `dict[str, int]`, *optional*):
            Custom vocabulary dictionary. If not provided, the vocabulary is loaded from `vocab_file`.
        merges (`str` or `list[str]`, *optional*):
            Custom merges list. If not provided, merges are loaded from `merges_file`.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        tags_dict,
        vocab: Optional[Union[str, dict[str, int], list[tuple[str, float]]]] = None,
        merges: Optional[Union[str, list[str]]] = None,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        max_depth=50,
        max_width=1000,
        pad_width=1001,
        pad_token_label=-100,
        only_label_first_subword=True,
        trim_offsets=False,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        if vocab is None:
            vocab = {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
            }
        merges = merges or []
        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        self._vocab = vocab
        self._merges = merges
        self._tokenizer = tokenizer
        super().__init__(
            tags_dict=tags_dict,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            max_depth=max_depth,
            max_width=max_width,
            pad_width=pad_width,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            **kwargs,
        )
        sep_token_str = str(sep_token)
        cls_token_str = str(cls_token)
        cls_token_id = self.cls_token_id
        sep_token_id = self.sep_token_id
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_token_str} $A {sep_token_str}",
            pair=f"{cls_token_str} $A {sep_token_str} $B {sep_token_str}",
            special_tokens=[
                (cls_token_str, cls_token_id),
                (sep_token_str, sep_token_id),
            ],
        )

        self.tags_dict = tags_dict

        # additional properties
        self.max_depth = max_depth
        self.max_width = max_width
        self.pad_width = pad_width
        self.unk_tag_id = len(self.tags_dict)
        self.pad_tag_id = self.unk_tag_id + 1
        self.pad_xpath_tags_seq = [self.pad_tag_id] * self.max_depth
        self.pad_xpath_subs_seq = [self.pad_width] * self.max_depth
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

    def get_xpath_seq(self, xpath):
        """
        Given the xpath expression of one particular node (like "/html/body/div/li[1]/div/span[2]"), return a list of
        tag IDs and corresponding subscripts, taking into account max depth.
        """
        xpath_tags_list = []
        xpath_subs_list = []

        xpath_units = xpath.split("/")
        for unit in xpath_units:
            if not unit.strip():
                continue
            name_subs = unit.strip().split("[")
            tag_name = name_subs[0]
            sub = 0 if len(name_subs) == 1 else int(name_subs[1][:-1])
            xpath_tags_list.append(self.tags_dict.get(tag_name, self.unk_tag_id))
            xpath_subs_list.append(min(self.max_width, sub))

        xpath_tags_list = xpath_tags_list[: self.max_depth]
        xpath_subs_list = xpath_subs_list[: self.max_depth]
        xpath_tags_list += [self.pad_tag_id] * (self.max_depth - len(xpath_tags_list))
        xpath_subs_list += [self.pad_width] * (self.max_depth - len(xpath_subs_list))

        return xpath_tags_list, xpath_subs_list

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]],
        text_pair: Optional[Union[PreTokenizedInput, list[PreTokenizedInput]]] = None,
        xpaths: Optional[Union[list[list[int]], list[list[list[int]]]]] = None,
        node_labels: Optional[Union[list[int], list[list[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences with nodes, xpaths and optional labels.

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
                (words of a single example or questions of a batch of examples) or a list of list of strings (batch of
                words).
            text_pair (`list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
                (pretokenized string).
            xpaths (`list[list[int]]`, `list[list[list[int]]]`):
                Node-level xpaths. Each bounding box should be normalized to be on a 0-1000 scale.
            node_labels (`list[int]`, `list[list[int]]`, *optional*):
                Node-level integer labels (for token classification tasks).
            is_split_into_words (`bool`, *optional*):
                Set to `True` if the inputs are already provided as pretokenized word lists.
        """

        placeholder_xpath = "/document/node"

        if isinstance(text, tuple):
            text = list(text)
        if text_pair is not None and isinstance(text_pair, tuple):
            text_pair = list(text_pair)

        if xpaths is None and not is_split_into_words:
            nodes_source = text if text_pair is None else text_pair
            if isinstance(nodes_source, tuple):
                nodes_source = list(nodes_source)
            processed_nodes = nodes_source

            if isinstance(nodes_source, str):
                processed_nodes = nodes_source.split()
            elif isinstance(nodes_source, list):
                if nodes_source and isinstance(nodes_source[0], str):
                    requires_split = any(" " in entry for entry in nodes_source)
                    if requires_split:
                        processed_nodes = [entry.split() for entry in nodes_source]
                    else:
                        processed_nodes = nodes_source
                elif nodes_source and isinstance(nodes_source[0], tuple):
                    processed_nodes = [list(sample) for sample in nodes_source]

            if text_pair is None:
                text = processed_nodes
            else:
                text_pair = processed_nodes

            if isinstance(processed_nodes, list) and processed_nodes and isinstance(processed_nodes[0], (list, tuple)):
                xpaths = [[placeholder_xpath] * len(sample) for sample in processed_nodes]
            else:
                length = len(processed_nodes) if hasattr(processed_nodes, "__len__") else 0
                xpaths = [placeholder_xpath] * length

        def _is_valid_text_input(t):
            if isinstance(t, str):
                return True
            if isinstance(t, (list, tuple)):
                if len(t) == 0:
                    return True
                if isinstance(t[0], str):
                    return True
                if isinstance(t[0], (list, tuple)):
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
            return False

        if text_pair is not None:
            # in case text + text_pair are provided, text = questions, text_pair = nodes
            if not _is_valid_text_input(text):
                raise ValueError("text input must of type `str` (single example) or `list[str]` (batch of examples). ")
            if not isinstance(text_pair, (list, tuple)):
                raise ValueError(
                    "Nodes must be of type `list[str]` (single pretokenized example), "
                    "or `list[list[str]]` (batch of pretokenized examples)."
                )
            is_batched = isinstance(text, (list, tuple))
        else:
            # in case only text is provided => must be nodes
            if not isinstance(text, (list, tuple)):
                raise ValueError(
                    "Nodes must be of type `list[str]` (single pretokenized example), "
                    "or `list[list[str]]` (batch of pretokenized examples)."
                )
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))

        nodes = text if text_pair is None else text_pair
        assert xpaths is not None, "You must provide corresponding xpaths"
        if is_batched:
            assert len(nodes) == len(xpaths), "You must provide nodes and xpaths for an equal amount of examples"
            for nodes_example, xpaths_example in zip(nodes, xpaths):
                assert len(nodes_example) == len(xpaths_example), "You must provide as many nodes as there are xpaths"
        else:
            assert len(nodes) == len(xpaths), "You must provide as many nodes as there are xpaths"

        if is_batched:
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`:"
                    f" {len(text_pair)}."
                )
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            is_pair = bool(text_pair is not None)
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                is_pair=is_pair,
                xpaths=xpaths,
                node_labels=node_labels,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                xpaths=xpaths,
                node_labels=node_labels,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            list[TextInput],
            list[TextInputPair],
            list[PreTokenizedInput],
        ],
        is_pair: Optional[bool] = None,
        xpaths: Optional[list[list[list[int]]]] = None,
        node_labels: Optional[Union[list[int], list[list[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            xpaths=xpaths,
            node_labels=node_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> list[str]:
        batched_input = [(text, pair)] if pair else [text]
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )

        return encodings[0].tokens

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        xpaths: Optional[list[list[int]]] = None,
        node_labels: Optional[list[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
        `__call__` should be used instead.

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.
            text_pair (`list[str]` or `list[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of list of strings (words of a batch of examples).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._encode_plus(
            text=text,
            xpaths=xpaths,
            text_pair=text_pair,
            node_labels=node_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            list[TextInput],
            list[TextInputPair],
            list[PreTokenizedInput],
        ],
        is_pair: Optional[bool] = None,
        xpaths: Optional[list[list[list[int]]]] = None,
        node_labels: Optional[list[list[int]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        if not isinstance(batch_text_or_text_pairs, list):
            raise TypeError(f"batch_text_or_text_pairs has to be a list (got {type(batch_text_or_text_pairs)})")

        # Set the truncation and padding strategy and restore the initial configuration
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
        )

        if is_pair:
            processed_inputs = []
            for text, text_pair in batch_text_or_text_pairs:
                if isinstance(text, tuple):
                    text = list(text)
                if isinstance(text, str):
                    text = [text]
                if isinstance(text_pair, tuple):
                    text_pair = list(text_pair)
                if isinstance(text_pair, str):
                    text_pair = [text_pair]
                processed_inputs.append((text, text_pair))
            batch_text_or_text_pairs = processed_inputs
        else:
            processed_inputs = []
            for text in batch_text_or_text_pairs:
                if isinstance(text, tuple):
                    text = list(text)
                if isinstance(text, str):
                    text = [text]
                processed_inputs.append(text)
            batch_text_or_text_pairs = processed_inputs

        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=True,  # we set this to True as MarkupLM always expects pretokenized inputs
        )

        # Convert encoding to dict
        # `Tokens` is a tuple of (list[dict[str, list[list[int]]]] or list[dict[str, 2D-Tensor]],
        #  list[EncodingFast]) with nested dimensions corresponding to batch, overflows, sequence length
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=True
                if node_labels is not None
                else return_offsets_mapping,  # we use offsets to create the labels
                return_length=return_length,
                verbose=verbose,
            )
            for encoding in encodings
        ]

        # Convert the output to have dict[list] from list[dict] and remove the additional overflows dimension
        # From (variable) shape (batch, overflows, sequence length) to ~ (batch * overflows, sequence length)
        # (we say ~ because the number of overflow varies with the example in the batch)
        #
        # To match each overflowing sample with the original sample in the batch
        # we add an overflow_to_sample_mapping array (see below)
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0]:
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, (toks, _) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks["input_ids"])
            sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        for input_ids in sanitized_tokens["input_ids"]:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)

        # create the token-level xpaths tags and subscripts
        xpath_tags_seq = []
        xpath_subs_seq = []
        for batch_index in range(len(sanitized_tokens["input_ids"])):
            if return_overflowing_tokens:
                original_index = sanitized_tokens["overflow_to_sample_mapping"][batch_index]
            else:
                original_index = batch_index
            xpath_tags_seq_example = []
            xpath_subs_seq_example = []
            for id, sequence_id, word_id in zip(
                sanitized_tokens["input_ids"][batch_index],
                sanitized_encodings[batch_index].sequence_ids,
                sanitized_encodings[batch_index].word_ids,
            ):
                if word_id is not None:
                    if is_pair and sequence_id == 0:
                        xpath_tags_seq_example.append(self.pad_xpath_tags_seq)
                        xpath_subs_seq_example.append(self.pad_xpath_subs_seq)
                    else:
                        xpath_tags_list, xpath_subs_list = self.get_xpath_seq(xpaths[original_index][word_id])
                        xpath_tags_seq_example.extend([xpath_tags_list])
                        xpath_subs_seq_example.extend([xpath_subs_list])
                else:
                    if id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                        xpath_tags_seq_example.append(self.pad_xpath_tags_seq)
                        xpath_subs_seq_example.append(self.pad_xpath_subs_seq)
                    else:
                        raise ValueError("Id not recognized")
            xpath_tags_seq.append(xpath_tags_seq_example)
            xpath_subs_seq.append(xpath_subs_seq_example)

        sanitized_tokens["xpath_tags_seq"] = xpath_tags_seq
        sanitized_tokens["xpath_subs_seq"] = xpath_subs_seq

        # optionally, create the labels
        if node_labels is not None:
            labels = []
            for batch_index in range(len(sanitized_tokens["input_ids"])):
                if return_overflowing_tokens:
                    original_index = sanitized_tokens["overflow_to_sample_mapping"][batch_index]
                else:
                    original_index = batch_index
                labels_example = []
                for id, offset, word_id in zip(
                    sanitized_tokens["input_ids"][batch_index],
                    sanitized_tokens["offset_mapping"][batch_index],
                    sanitized_encodings[batch_index].word_ids,
                ):
                    if word_id is not None:
                        if self.only_label_first_subword:
                            if offset[0] == 0:
                                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                                labels_example.append(node_labels[original_index][word_id])
                            else:
                                labels_example.append(self.pad_token_label)
                        else:
                            labels_example.append(node_labels[original_index][word_id])
                    else:
                        labels_example.append(self.pad_token_label)
                labels.append(labels_example)

            sanitized_tokens["labels"] = labels
            # finally, remove offsets if the user didn't want them
            if not return_offsets_mapping:
                del sanitized_tokens["offset_mapping"]

        return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        xpaths: Optional[list[list[int]]] = None,
        node_labels: Optional[list[int]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        placeholder_xpath = "/document/node"

        if isinstance(text, tuple):
            text = list(text)
        if text_pair is not None and isinstance(text_pair, tuple):
            text_pair = list(text_pair)

        nodes_single = text if text_pair is None else text_pair
        processed_nodes = nodes_single

        if isinstance(nodes_single, str):
            processed_nodes = nodes_single.split()
        elif isinstance(nodes_single, list) and nodes_single and isinstance(nodes_single[0], str):
            processed_nodes = nodes_single

        if text_pair is None:
            text = processed_nodes
        else:
            text_pair = processed_nodes

        if xpaths is None:
            length = len(processed_nodes) if hasattr(processed_nodes, "__len__") else 0
            xpaths = [placeholder_xpath] * length

        # make it a batched input
        # 2 options:
        # 1) only text, in case text must be a list of str
        # 2) text + text_pair, in which case text = str and text_pair a list of str
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_xpaths = [xpaths]
        batched_node_labels = [node_labels] if node_labels is not None else None
        batched_output = self._batch_encode_plus(
            batched_input,
            is_pair=bool(text_pair is not None),
            xpaths=batched_xpaths,
            node_labels=batched_node_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        # Return tensor is None, then we can remove the leading batch axis
        # Overflowing tokens are returned as a batch of output so we keep them in this case
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output

    def _pad(
        self,
        encoded_inputs: Union[dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Args:
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
            encoded_inputs:
                Dictionary of tokenized inputs (`list[int]`) or batch of tokenized inputs (`list[list[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.
                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:
                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side:
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            padding_side = padding_side if padding_side is not None else self.padding_side
            if padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "xpath_tags_seq" in encoded_inputs:
                    encoded_inputs["xpath_tags_seq"] = (
                        encoded_inputs["xpath_tags_seq"] + [self.pad_xpath_tags_seq] * difference
                    )
                if "xpath_subs_seq" in encoded_inputs:
                    encoded_inputs["xpath_subs_seq"] = (
                        encoded_inputs["xpath_subs_seq"] + [self.pad_xpath_subs_seq] * difference
                    )
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = encoded_inputs["labels"] + [self.pad_token_label] * difference
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "xpath_tags_seq" in encoded_inputs:
                    encoded_inputs["xpath_tags_seq"] = [self.pad_xpath_tags_seq] * difference + encoded_inputs[
                        "xpath_tags_seq"
                    ]
                if "xpath_subs_seq" in encoded_inputs:
                    encoded_inputs["xpath_subs_seq"] = [self.pad_xpath_subs_seq] * difference + encoded_inputs[
                        "xpath_subs_seq"
                    ]
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = [self.pad_token_label] * difference + encoded_inputs["labels"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(padding_side))

        return encoded_inputs

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoBERTa sequence has the following format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`list[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `list[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`list[int]`):
                List of IDs.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `list[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)


MarkupLMTokenizerFast = MarkupLMTokenizer


__all__ = ["MarkupLMTokenizer", "MarkupLMTokenizerFast"]
