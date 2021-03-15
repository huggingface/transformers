# coding=utf-8
# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors.
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
""" Tokenization class for model LayoutLM."""
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union

from ... import TensorType, add_end_docstrings
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlm-base-uncased": "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/vocab.txt",
        "microsoft/layoutlm-large-uncased": "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlm-base-uncased": 512,
    "microsoft/layoutlm-large-uncased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/layoutlm-base-uncased": {"do_lower_case": True},
    "microsoft/layoutlm-large-uncased": {"do_lower_case": True},
}

TEXT_AND_BOX_ARGS_DOCSTRING = r"""
        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            bbox (:obj:`BoundingBox`, :obj:`List[BoundingBox]`, :obj:`List[List[BoundingBox]]`):
                The bounding boxes associated with the given :obj:`text` A bounding box is defined by (top, left,
                right, bottom) coordinates Floating point coordinates will be rounded to integer values as expected by
                the LayoutLM model The values are optionally normalized to a page's width and height, see parameter
                :obj:`orig_width_and_height`
            text_pair (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            bbox_pair (:obj:`BoundingBox`, :obj:`List[BoundingBox]`, :obj:`List[List[BoundingBox]]`):
                Additional bounding boxes associated with the given :obj:`text_pair` A bounding box is defined by (top,
                left, right, bottom) coordinates. Floating point coordinates will be rounded to integer values as
                expected by the LayoutLM model The values are optionally normalized to a page's width and height, see
                parameter :obj:`orig_width_and_height`
            orig_width_and_height (:obj:`Tuple[float, float]`, :obj:`List[Tuple[float, float]]`, `optional`):
                size of original document page from which bounding boxes are taken if this argument is passed, the
                bounding box coordinates are normalized accordingly in case of batch inputs, a list of size tuples may
                be passed, otherwise the same size is applied for all samples in the batch
"""


class BoundingBox(namedtuple("BoundingBox", "left top right bottom")):
    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @classmethod
    def rounded_bbox(cls, left, top, right, bottom):
        return cls(int(left), int(top), int(right), int(bottom))


BoundingBoxList = List[BoundingBox]


class LayoutLMTokenizer(BertTokenizer):
    r"""
    Constructs a LayoutLM tokenizer.

    :class:`~transformers.LayoutLMTokenizer like :class:`~transformers.BertTokenizer` runs end-to-end tokenization:
    punctuation splitting + wordpiece It additionally takes care of handling bounding boxes by repeating a bounding box
    associated with the corresponding text. Additional default boxes are defined for [SEP], [PAD] and [CLS] tokens and
    added correspondingly. Bounding box coordinates will be rounded to integer values and may optionally be normalized
    to a target width and height.

    Note that for decoding bounding boxes are not considered.

    Args:
        sep_box (:obj:`BoundingBox`, defaults to :obj:`BoundingBox(1000, 1000, 1000, 1000)`):
            the default box assigned to the [SEP] token
        pad_box (:obj:`BoundingBox`, defaults to :obj:`BoundingBox(0, 0, 0, 0)`):
            the default box assigned to the [PAD] token
        cls_box (:obj:`BoundingBox`, defaults to :obj:`BoundingBox(1, 1, 1, 1)`):
            the default box assigned to the [CLS] token
        target_height_and_width (:obj:`Tuple[int, int]`, defaults to :obj:`(1000, 1000)`):
            the target width and height of bounding boxes only has an effect when the :obj:`orig_height_and_width`
            argument is passed to the tokenization methods
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    model_input_names: List[str] = ["input_ids", "bbox", "token_type_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        sep_box: BoundingBox = BoundingBox(1000, 1000, 1000, 1000),
        pad_box: BoundingBox = BoundingBox(0, 0, 0, 0),
        cls_box: BoundingBox = BoundingBox(1, 1, 1, 1),
        target_width_and_height: Tuple[int, int] = (1000, 1000),
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            cls_box=cls_box,
            pad_box=pad_box,
            sep_box=sep_box,
            target_width_and_height=target_width_and_height,
            **kwargs,
        )
        self.cls_box = cls_box
        self.pad_box = pad_box
        self.sep_box = sep_box
        self.target_width, self.target_height = target_width_and_height

    @add_end_docstrings(TEXT_AND_BOX_ARGS_DOCSTRING, ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        bbox: Union[BoundingBox, BoundingBoxList, List[BoundingBox], List[BoundingBoxList]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        bbox_pair: Optional[Union[BoundingBox, BoundingBoxList, List[BoundingBox], List[BoundingBoxList]]] = None,
        orig_width_and_height: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences. Identical to :obj:`PreTrainedTokenizerBase.__call__` with additional handling of bounding boxes. If
        a text is split further the corresponding bounding boxes will be repeated that many times. The bounding boxes
        may optionally be normalized to the given width and height, according to the parameters of
        :obj:`target_width_and_height` and :obj:`orig_height_and_width`.
        """
        self._check_input_type(text, input_name="text")
        self._check_num_samples(text, bbox)
        if text_pair is not None:
            self._check_input_type(text_pair, input_name="text_pair")
            assert bbox_pair is not None, "When passing a `text_pair` argument, a `bbox_pair` must be provided as well"
            self._check_num_samples(text_pair, bbox_pair)

        is_batched = bool(
            (not is_split_into_words and isinstance(text, (list, tuple)))
            or (
                is_split_into_words and isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
            )
        )

        if is_batched:
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            batch_bbox_or_bbox_pairs = list(zip(bbox, bbox_pair)) if bbox_pair is not None else bbox
            if orig_width_and_height is not None:
                if isinstance(orig_width_and_height[0], (list, tuple)):
                    assert len(orig_width_and_height) == len(batch_text_or_text_pairs), (
                        "When passing a list of `orig_width_and_height` for a batch it must contain an item for each "
                        "sample"
                    )
                else:
                    orig_width_and_height = [orig_width_and_height] * len(batch_text_or_text_pairs)
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                batch_bbox_or_bbox_pairs=batch_bbox_or_bbox_pairs,
                orig_width_and_height=orig_width_and_height,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
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
                bbox=bbox,
                text_pair=text_pair,
                bbox_pair=bbox_pair,
                orig_width_and_height=orig_width_and_height,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
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

    def _check_num_samples(self, text, bbox):
        if not text:
            assert not bbox, "supplied bounding boxes for empty text"
            return
        if isinstance(text, str):
            assert isinstance(bbox, tuple) and (
                isinstance(bbox[0], (float, int))
            ), "if just one text sample is provided, you may only provide one bounding box for it"
        if isinstance(text, (list, tuple)):
            if isinstance(text[0], (list, tuple)):
                assert isinstance(
                    bbox[0], (list, tuple)
                ), "expected a batch of bounding boxes for a batch of text samples"
                text_sample_lengths = list(map(len, text[0]))
                bbox_sample_lengths = list(map(len, bbox[0]))
                assert text_sample_lengths == bbox_sample_lengths, (
                    "text and bbox argument must be of same length for each sample in a batch! Got text samples of "
                    f"length {text_sample_lengths} and bbox samples of length {bbox_sample_lengths}"
                )
            else:
                assert len(text) == len(bbox), (
                    "text and bbox argument must be of same length for pretokenized inputs, got text of length "
                    f"{len(text)} and bbox og length {len(bbox)}"
                )
                assert isinstance(bbox[0], tuple) and (
                    isinstance(bbox[0][0], (float, int))
                ), "expected a bounding box for each token in pretokenized input"

    @add_end_docstrings(
        TEXT_AND_BOX_ARGS_DOCSTRING,
        ENCODE_KWARGS_DOCSTRING,
        """
            **kwargs: Passed along to the `.tokenize()` method.
        """,
        """
        Returns:
            :obj:`List[int]`, :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`: The tokenized ids of the
            text.
        """,
    )
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        bbox: Union[BoundingBox, BoundingBoxList],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        bbox_pair: Optional[Union[BoundingBox, BoundingBoxList]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> Tuple[List[int], BoundingBoxList]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary. Adds additional bounding
        boxes for split words.
        """
        encoded_inputs = self.encode_plus(
            text,
            bbox,
            text_pair=text_pair,
            bbox_pair=bbox_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"], encoded_inputs["bbox"]

    @add_end_docstrings(TEXT_AND_BOX_ARGS_DOCSTRING, ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        bbox: Union[BoundingBox, BoundingBoxList],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        bbox_pair: Optional[Union[BoundingBox, BoundingBoxList]] = None,
        orig_width_and_height: Optional[Tuple[float, float]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        .. warning::
            This method is deprecated, ``__call__`` should be used instead.
        """

        self._check_num_samples(text, bbox)
        if text_pair is not None:
            self._check_num_samples(text_pair, bbox_pair)

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
            bbox=bbox,
            text_pair=text_pair,
            bbox_pair=bbox_pair,
            orig_width_and_height=orig_width_and_height,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
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

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        batch_bbox_or_bbox_pairs: Union[
            List[BoundingBox], List[BoundingBoxList], List[Tuple[BoundingBox, BoundingBox]]
        ],
        orig_width_and_height: Optional[List[Tuple[float, float]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        .. warning::
            This method is deprecated, ``__call__`` should be used instead.

        Args:
            batch_text_or_text_pairs (:obj:`List[str]`, :obj:`List[Tuple[str, str]]`, :obj:`List[List[str]]`, :obj:`List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also :obj:`List[List[int]]`, :obj:`List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in ``encode_plus``).
            batch_bbox_or_bbox_pairs (:obj:`List[BoundingBox]`, :obj:`List[BoundingBoxList]`, :obj:`List[Tuple[BoundingBox, BoundingBox]]`)
                Batch of sequences or pair of sequences of bounding boxes
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

        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            batch_bbox_or_bbox_pairs=batch_bbox_or_bbox_pairs,
            orig_width_and_height=orig_width_and_height,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
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

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        bbox: Union[BoundingBox, BoundingBoxList],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        bbox_pair: Optional[Union[BoundingBox, BoundingBoxList]] = None,
        orig_width_and_height: Optional[Tuple[float, float]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids_and_extend_bbox(text, bbox):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                extended_bbox = [bbox] * len(tokens)
                return self.convert_tokens_to_ids(tokens), extended_bbox
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = []
                    boxes = []
                    for t, b in zip(text, bbox):
                        new_tokens = self.tokenize(t, is_split_into_words=True, **kwargs)
                        tokens.extend(new_tokens)
                        boxes.extend([b] * len(new_tokens))
                    return self.convert_tokens_to_ids(tokens), boxes
                else:
                    return self.convert_tokens_to_ids(text), bbox
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text, bbox
            else:
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when `is_split_into_words=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                    )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        first_ids, first_bbox = get_input_ids_and_extend_bbox(text, bbox)
        second_ids, second_bbox = (
            get_input_ids_and_extend_bbox(text_pair, bbox_pair) if text_pair is not None else (None, None)
        )

        return self.prepare_for_model(
            first_ids,
            first_bbox,
            pair_ids=second_ids,
            pair_bbox=second_bbox,
            orig_width_and_height=orig_width_and_height,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        batch_bbox_or_bbox_pairs: Union[
            List[BoundingBox], List[BoundingBoxList], List[Tuple[BoundingBox, BoundingBox]]
        ],
        orig_width_and_height: Optional[List[Tuple[float, float]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        def get_input_ids_and_extend_bbox(text, bbox):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens), [bbox] * len(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = []
                    boxes = []
                    for t, b in zip(text, bbox):
                        new_tokens = self.tokenize(t, is_split_into_words=True, **kwargs)
                        tokens.extend(new_tokens)
                        boxes.extend([b] * len(new_tokens))
                    return self.convert_tokens_to_ids(tokens), boxes
                else:
                    return self.convert_tokens_to_ids(text), bbox
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text, bbox
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        input_boxes = []
        for ids_or_pair_ids, bbox_or_pair_bbox in zip(batch_text_or_text_pairs, batch_bbox_or_bbox_pairs):
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
                boxes, pair_boxes = bbox_or_pair_bbox, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
                boxes, pair_boxes = bbox_or_pair_bbox, None
            else:
                ids, pair_ids = ids_or_pair_ids
                boxes, pair_boxes = bbox_or_pair_bbox

            first_ids, first_bbox = get_input_ids_and_extend_bbox(ids, boxes)
            second_ids, second_bbox = (
                get_input_ids_and_extend_bbox(pair_ids, pair_boxes) if pair_ids is not None else (None, None)
            )
            input_ids.append((first_ids, second_ids))
            input_boxes.append((first_bbox, second_bbox))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            input_boxes,
            orig_width_and_height=orig_width_and_height,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    @add_end_docstrings(TEXT_AND_BOX_ARGS_DOCSTRING, ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        ids: List[int],
        bbox: BoundingBoxList,
        pair_ids: Optional[List[int]] = None,
        pair_bbox: Optional[BoundingBoxList] = None,
        orig_width_and_height: Optional[Tuple[float, float]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids with corresponding bounding boxes so that
        they can be used by the model. It adds special tokens and boxes, truncates sequences if overflowing while
        taking into account the special tokens and manages a moving window (with user defined stride) for overflowing
        tokens
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

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        assert len_ids == len(bbox)
        len_pair_ids = len(pair_ids) if pair else 0
        if pair:
            assert len_pair_ids == len(pair_bbox)

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        overflowing_boxes = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, bbox, pair_ids, pair_bbox, overflowing_tokens, overflowing_boxes = self.truncate_sequences(
                ids,
                bbox,
                pair_ids=pair_ids,
                pair_bbox=pair_bbox,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length
            encoded_inputs["overflowing_boxes"] = overflowing_boxes

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            bbox_sequence = self.build_bbox_input_with_special_boxes(bbox, pair_bbox)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            bbox_sequence = bbox + pair_bbox if pair else bbox
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # normalize bounding boxes and round to integer values
        if orig_width_and_height is None:
            bbox_sequence = [BoundingBox.rounded_bbox(*box) for box in bbox_sequence]
        else:
            bbox_sequence = [self._normalize_bbox(box, *orig_width_and_height) for box in bbox_sequence]

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        encoded_inputs["bbox"] = bbox_sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def build_bbox_input_with_special_boxes(
        self, boxes_0: BoundingBoxList, boxes_1: Optional[BoundingBoxList] = None
    ) -> BoundingBoxList:
        """
        Build bounding box model inputs from a sequence or a pair of sequence for sequence classification tasks by
        concatenating and adding special boxes. A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        where X, A and B are sequences of bounding boxes and [CLS] and [SEP] are the defined special boxes

        Args:
            boxes_0 (:obj:`List[BoundingBox]`):
                List of bounding boxes to which the special boxes will be added.
            boxes_1 (:obj:`List[BoundingBoxes]`, `optional`):
                Optional second list of bounding boxes for sequence pairs.

        Returns:
            :obj:`List[BoundingBoxes]`: List of input bounding boxes with the appropriate special boxes.
        """
        if boxes_1 is None:
            return [self.cls_box] + boxes_0 + [self.sep_box]
        cls = [self.cls_box]
        sep = [self.sep_box]
        return cls + boxes_0 + sep + boxes_1 + sep

    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        batch_box_pairs: List[Union[Tuple[BoundingBoxList, BoundingBox], Tuple[BoundingBoxList, None]]],
        orig_width_and_height: Optional[List[Tuple[float, float]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares sequences of input id and bounding boxes, or a pair of sequences of inputs ids and bounding boxes so
        that it can be used by the model. It adds special tokens and boxes, truncates sequences if overflowing while
        taking into account the special tokens and manages a moving window (with user defined stride) for overflowing
        tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
            batch_box_pairs: list of bounding boxes or bounding box pairs
        """

        batch_outputs = {}
        for i, ((first_ids, second_ids), (first_boxes, second_boxes)) in enumerate(
            zip(batch_ids_pairs, batch_box_pairs)
        ):
            orig_wh = orig_width_and_height[i] if orig_width_and_height else None
            outputs = self.prepare_for_model(
                first_ids,
                first_boxes,
                second_ids,
                second_boxes,
                orig_width_and_height=orig_wh,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch) Identical to
        :obj:`PreTrainedTokenizerBase._pad` but additionally handles bounding box inputs, padding with
        :obj:`self.pad_box`

        Args:
            encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`)
                            As well
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
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        input_ids = encoded_inputs[self.model_input_names[0]]
        bbox = encoded_inputs[self.model_input_names[1]]

        if len(input_ids) != len(bbox):
            raise ValueError(
                "encountered differently sized input_ids and bbox lists, "
                "please make sure that for each input token there is an associated bounding box. "
                f"length of input_ids: {len(input_ids)}, length of bbox: {len(bbox)}"
            )

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(input_ids)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(input_ids) != max_length

        if needs_to_be_padded:
            difference = max_length - len(input_ids)
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(input_ids) + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = input_ids + [self.pad_token_id] * difference
                encoded_inputs[self.model_input_names[1]] = bbox + [self.pad_box] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(input_ids)
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + input_ids
                encoded_inputs[self.model_input_names[1]] = [self.pad_box] * difference + bbox
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(input_ids)

        return encoded_inputs

    def truncate_sequences(
        self,
        ids: List[int],
        bbox: BoundingBoxList,
        pair_ids: Optional[List[int]] = None,
        pair_bbox: Optional[BoundingBoxList] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], BoundingBoxList, List[int], BoundingBoxList, List[int], BoundingBoxList]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (:obj:`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            bbox (:obj:`List[BoundingBox]`)
                Bounding boxes corresponding to
            pair_ids (:obj:`List[int]`, `optional`):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (:obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`, `optional`, defaults to :obj:`False`):
                The strategy to follow for truncation. Can be:

                * :obj:`'longest_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            stride (:obj:`int`, `optional`, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            :obj:`Tuple[List[int], List[int], List[int]]`: The truncated ``ids``, the truncated ``pair_ids`` and the
            list of overflowing tokens.
        """

        if num_tokens_to_remove <= 0:
            return ids, bbox, pair_ids, pair_bbox, [], []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        overflowing_boxes = []
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    if not overflowing_tokens:
                        window_len = min(len(ids), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(ids[-window_len:])
                    overflowing_boxes.extend(bbox[-window_len:])
                    ids = ids[:-1]
                    bbox = bbox[:-1]
                else:
                    if not overflowing_tokens:
                        window_len = min(len(pair_ids), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(pair_ids[-window_len:])
                    overflowing_boxes.extend(pair_bbox[-window_len:])
                    pair_ids = pair_ids[:-1]
                    pair_bbox = pair_bbox[:-1]
        elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                overflowing_tokens = ids[-window_len:]
                overflowing_boxes = bbox[-window_len:]
                ids = ids[:-num_tokens_to_remove]
                bbox = bbox[:-num_tokens_to_remove]
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input"
                    f"but the first sequence has a length {len(ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    f"for instance 'longest_first' or 'only_second'."
                )
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                overflowing_tokens = pair_ids[-window_len:]
                overflowing_boxes = pair_bbox[-window_len:]
                pair_ids = pair_ids[:-num_tokens_to_remove]
                pair_bbox = pair_bbox[:-num_tokens_to_remove]
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input"
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    f"for instance 'longest_first' or 'only_first'."
                )

        return ids, bbox, pair_ids, pair_bbox, overflowing_tokens, overflowing_boxes

    def _normalize_bbox(self, bbox: BoundingBox, orig_width: float, orig_height: float):
        return BoundingBox(
            int(self.target_width * (bbox[0] / orig_width)),
            int(self.target_height * (bbox[1] / orig_height)),
            int(self.target_width * (bbox[2] / orig_width)),
            int(self.target_height * (bbox[3] / orig_height)),
        )
