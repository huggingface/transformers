# tokenization_arlow.py

import copy
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import tokenizers.pre_tokenizers as pre_tokenizers_fast
from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer

# Adjust these imports to your actual project structure
from .convert_slow_tokenizer import convert_slow_tokenizer
from .integrations.ggml import convert_gguf_tokenizer
from .modeling_gguf_pytorch_utils import load_gguf_checkpoint
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from .utils import PaddingStrategy, add_end_docstrings, logging


logger = logging.get_logger(__name__)

################################################################################
# Constants / Mappings
################################################################################

TOKENIZER_FILE = "tokenizer.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
TIKTOKEN_VOCAB_FILE = "tokenizer.model"

ADDED_TOKENS_FILE = "added_tokens.json"

INIT_TOKENIZER_DOCSTRING += """
    tokenizer_object ([`tokenizers.Tokenizer`]):
        A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from.
    tokenizer_file ([`str`]):
        A path to a local JSON file representing a previously serialized
        [`tokenizers.Tokenizer`] object from 🤗 tokenizers.
"""

MODEL_TO_TRAINER_MAPPING = {
    "BPE": BpeTrainer,
    "Unigram": UnigramTrainer,
    "WordLevel": WordLevelTrainer,
    "WordPiece": WordPieceTrainer,
}

VOCAB_FILES_NAMES = {
    "tokenizer_file": TOKENIZER_FILE,
    "vocab_file": TIKTOKEN_VOCAB_FILE,
}

################################################################################
# Custom "fast" tokenizer class named ArlowTokenizer
################################################################################

@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class ArlowTokenizer(PreTrainedTokenizerBase):
    """
    ArlowTokenizer is a custom "fast" tokenizer class (wrapping the Hugging Face
    `tokenizers` library) that functions like a PreTrainedTokenizerFast but bears
    the custom architecture name "ArlowTokenizer". This allows you to do:

        tokenizer = ArlowTokenizer.from_pretrained(...)

    The implementation copies the functionality of PreTrainedTokenizerFast, including
    training from scratch, saving, loading, handling special tokens, pre/post-processing,
    etc.

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class: PreTrainedTokenizer = None

    def __init__(self, *args, **kwargs):
        tokenizer_object = kwargs.pop("tokenizer_object", None)
        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        gguf_file = kwargs.pop("gguf_file", None)
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)
        from_slow = kwargs.pop("from_slow", False)
        added_tokens_decoder = kwargs.pop("added_tokens_decoder", {})

        # Like GPT2/ByteLevel tokenizers might want a prefix space
        self.add_prefix_space = kwargs.get("add_prefix_space", False)

        if from_slow and slow_tokenizer is None and self.slow_tokenizer_class is None:
            raise ValueError(
                "Cannot instantiate this tokenizer from a slow version. If it's based on "
                "sentencepiece, make sure sentencepiece is installed."
            )

        # 1) If an actual HuggingFace `tokenizers` object was passed:
        if tokenizer_object is not None:
            fast_tokenizer = copy.deepcopy(tokenizer_object)
        # 2) If we have a .json file from the `tokenizers` library:
        elif fast_tokenizer_file is not None and not from_slow:
            fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
        # 3) If we have a slow tokenizer that needs to be converted:
        elif slow_tokenizer:
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        # 4) If we have a GGUF checkpoint:
        elif gguf_file is not None:
            gguf_param = load_gguf_checkpoint(kwargs.get("vocab_file"))
            architecture = gguf_param["config"]["model_type"]
            tokenizer_dict = gguf_param["tokenizer"]
            tokenizer_config = gguf_param["tokenizer_config"]
            fast_tokenizer, additional_kwargs = convert_gguf_tokenizer(architecture, tokenizer_dict)
            kwargs.update(tokenizer_config)
            if len(additional_kwargs) > 0:
                kwargs.update(additional_kwargs)
        # 5) If we have a slow tokenizer class that can be instantiated & converted
        elif self.slow_tokenizer_class is not None and slow_tokenizer is not False:
            slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
            fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        # 6) If all else fails, try to create from tiktoken-based approach
        elif not slow_tokenizer:
            self.vocab_file = kwargs.get("vocab_file", None)
            self.additional_special_tokens = kwargs.get("additional_special_tokens", [])
            fast_tokenizer = convert_slow_tokenizer(self, from_tiktoken=False)
            slow_tokenizer = None
        else:
            raise ValueError(
                "Couldn't instantiate the tokenizer from any of the known sources:\n"
                "1) a `tokenizers` library JSON,\n"
                "2) a slow tokenizer instance,\n"
                "3) an equivalent slow tokenizer class,\n"
                "4) a tiktoken-based approach.\n"
                "Ensure sentencepiece or tiktoken is installed if needed."
            )

        self._tokenizer = fast_tokenizer

        # If we got a slow tokenizer, update `kwargs` from that
        if slow_tokenizer is not None:
            kwargs.update(slow_tokenizer.init_kwargs)

        # For internal logic
        self._decode_use_source_tokenizer = False

        # If the tokenizer had preset truncation/padding, keep that
        _truncation = self._tokenizer.truncation
        if _truncation is not None:
            self._tokenizer.enable_truncation(**_truncation)
            kwargs.setdefault("max_length", _truncation["max_length"])
            kwargs.setdefault("truncation_side", _truncation["direction"])
            kwargs.setdefault("stride", _truncation["stride"])
            kwargs.setdefault("truncation_strategy", _truncation["strategy"])
        else:
            self._tokenizer.no_truncation()

        _padding = self._tokenizer.padding
        if _padding is not None:
            self._tokenizer.enable_padding(**_padding)
            kwargs.setdefault("pad_token", _padding["pad_token"])
            kwargs.setdefault("pad_token_type_id", _padding["pad_type_id"])
            kwargs.setdefault("padding_side", _padding["direction"])
            kwargs.setdefault("max_length", _padding["length"])
            kwargs.setdefault("pad_to_multiple_of", _padding["pad_to_multiple_of"])
        else:
            self._tokenizer.no_padding()

        # Init from our base class
        super().__init__(**kwargs)

        # Let the user decide if we want to keep special tokens separate on encode
        self._tokenizer.encode_special_tokens = self.split_special_tokens

        # Now add tokens that might be missing
        added_tokens_decoder_hash = {hash(repr(token)) for token in self.added_tokens_decoder}
        tokens_to_add = [
            token
            for index, token in sorted(added_tokens_decoder.items(), key=lambda x: x[0])
            if hash(repr(token)) not in added_tokens_decoder_hash
        ]

        encoder_known = list(self.added_tokens_encoder.keys()) + [str(token) for token in tokens_to_add]
        tokens_to_add += [
            token for token in self.all_special_tokens_extended
            if token not in encoder_known and token not in tokens_to_add
        ]

        if len(tokens_to_add) > 0:
            tokens = []
            special_tokens = self.all_special_tokens
            for token in tokens_to_add:
                is_special = (
                    (token.special or str(token) in special_tokens)
                    if isinstance(token, AddedToken)
                    else str(token) in special_tokens
                )
                if isinstance(token, str):
                    token = AddedToken(token, special=is_special)
                else:
                    token.special = is_special
                tokens.append(token)
            if tokens:
                self.add_tokens(tokens)

        # If the pre_tokenizer has an 'add_prefix_space' config, update it
        try:
            pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
            if pre_tok_state.get("add_prefix_space", self.add_prefix_space) != self.add_prefix_space:
                pre_tok_class = getattr(pre_tokenizers_fast, pre_tok_state.pop("type"))
                pre_tok_state["add_prefix_space"] = self.add_prefix_space
                self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
        except Exception:
            # Some tokenizers or custom pretokenizers can't be re-serialized
            pass

        # Set a default "model_type" name, if you want to show up as "ArlowGPT" or so
        self.model_type = "ArlowGPT"

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        Whether or not the slow tokenizer can be saved. SentencePiece-based ones can fail
        if the original "sentencepiece.model" is missing. For BPE-based, it's typically True.
        """
        return True

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        return self._tokenizer.get_added_tokens_decoder()

    def get_added_vocab(self) -> Dict[str, int]:
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        return self._tokenizer.decoder

    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[EncodingFast]]:
        """
        Internal helper to convert a single `EncodingFast` to a python dictionary,
        and handle overflows if `return_overflowing_tokens` is True.
        """
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)
            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            if return_length:
                encoding_dict["length"].append(len(e.ids))

        return encoding_dict, encodings

    def convert_tokens_to_ids(self, tokens: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)
        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)
        return self._tokenizer.add_tokens(new_tokens)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)

        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    def tokenize(
        self,
        text: str,
        pair: Optional[str] = None,
        add_special_tokens: bool = False,
        **kwargs
    ) -> List[str]:
        return self.encode_plus(
            text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs
        ).tokens()

    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        stride: int,
        pad_to_multiple_of: Optional[int],
        padding_side: Optional[str],
    ):
        """
        Sets truncation/padding on the internal `tokenizers` object before batch encoding.
        Restores the old state upon exit.
        """
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding

        # Truncation
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            if _truncation is not None:
                self._tokenizer.no_truncation()
        else:
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy.value,
                "direction": self.truncation_side,
            }
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}
            if current != target:
                self._tokenizer.enable_truncation(**target)

        # Padding
        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            if _padding is not None:
                self._tokenizer.no_padding()
        else:
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {
                "length": length,
                "direction": padding_side if padding_side is not None else self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_type_id": self.pad_token_type_id,
                "pad_to_multiple_of": pad_to_multiple_of,
            }
            if _padding != target:
                self._tokenizer.enable_padding(**target)

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
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
        split_special_tokens: bool = False,
    ) -> BatchEncoding:
        if not isinstance(batch_text_or_text_pairs, (list, tuple)):
            raise TypeError("batch_text_or_text_pairs must be a list or tuple.")

        # Adjust the internal truncation/padding config
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
        )

        # Possibly alter whether special tokens remain separate
        if self._tokenizer.encode_special_tokens != split_special_tokens:
            self._tokenizer.encode_special_tokens = split_special_tokens

        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )

        # Convert to python dict + handle overflows
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=enc,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
            )
            for enc in encodings
        ]

        # Flatten it out: batch dimension + possible overflows
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]

        if return_overflowing_tokens:
            # map each overflow back to the original batch index
            overflow_to_sample_mapping = []
            for i, (tok_data, _) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(tok_data["input_ids"])
            sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        for input_ids_list in sanitized_tokens["input_ids"]:
            self._eventual_warn_about_too_long_sequence(input_ids_list, max_length, verbose)

        return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
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
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        # Convert to single-batch for simpler usage
        if text_pair is not None:
            batch_items = [(text, text_pair)]
        else:
            batch_items = [text]

        batch_output = self._batch_encode_plus(
            batch_items,
            is_split_into_words=is_split_into_words,
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
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

        # If we are not returning multiple overflows, we can drop the extra dimension
        if return_tensors is None and not return_overflowing_tokens:
            batch_output = BatchEncoding(
                {
                    k: (v[0] if len(v) > 0 and isinstance(v[0], list) else v)
                    for k, v in batch_output.items()
                },
                batch_output.encodings,
            )

        self._eventual_warn_about_too_long_sequence(batch_output["input_ids"], max_length, verbose)
        return batch_output

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        if self.backend_tokenizer.decoder is not None:
            return self.backend_tokenizer.decoder.decode(tokens)
        return " ".join(tokens)

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        if clean_up_tokenization_spaces is None:
            clean_up_tokenization_spaces = self.clean_up_tokenization_spaces
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        return text

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save the tokenizer in both possible formats: 
          - slow (legacy) format (vocab file + merges + etc.) if supported
          - fast (tokenizer.json)
        """
        save_directory = str(save_directory)

        # If we have no slow version, we can't do legacy_format=True
        if self.slow_tokenizer_class is None and legacy_format is True:
            raise ValueError(
                "This custom tokenizer does not have a slow version. "
                "Cannot save with legacy_format=True."
            )

        save_slow = (
            (legacy_format is None or legacy_format is True)
            and self.slow_tokenizer_class is not None
            and self.can_save_slow_tokenizer
        )
        save_fast = (legacy_format is None or legacy_format is False)

        if save_slow:
            added_tokens_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
            )
            added_vocab = {
                tok: idx for tok, idx in self.added_tokens_encoder.items() if idx >= self.vocab_size
            }
            if added_vocab:
                with open(added_tokens_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

            vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
            file_names = file_names + vocab_files + (added_tokens_file,)

        if save_fast:
            tokenizer_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
            )
            self.backend_tokenizer.save(tokenizer_file)
            file_names = file_names + (tokenizer_file,)

        return file_names

    def train_new_from_iterator(
        self,
        text_iterator,
        vocab_size,
        length=None,
        new_special_tokens=None,
        special_tokens_map=None,
        **kwargs,
    ):
        """
        Train a brand-new tokenizer from an iterator of text batches, preserving
        the same "pre-tokenization" and pipeline details as this instance.

        text_iterator:
            generator or list of text batches (list of strings).
        vocab_size:
            target size of the vocabulary.
        length:
            total number of sequences in text_iterator (optional, used for progress bars).
        new_special_tokens:
            list of additional special tokens to incorporate.
        special_tokens_map:
            dictionary {old_token: new_token} to rename special tokens if desired.
        **kwargs:
            additional arguments passed to the underlying Trainer from 🤗 tokenizers.
        """
        tokenizer_json = json.loads(self._tokenizer.to_str())

        # Remove added tokens + merges + any post_processor references to old IDs
        added_tokens = tokenizer_json.pop("added_tokens")
        post_processor = tokenizer_json.pop("post_processor")

        unk_token = None

        # Clear out the existing vocab merges
        if tokenizer_json["model"]["type"] == "BPE":
            tokenizer_json["model"]["vocab"] = {}
            tokenizer_json["model"]["merges"] = []
        elif tokenizer_json["model"]["type"] == "Unigram":
            if tokenizer_json["model"]["unk_id"] is not None:
                unk_id = tokenizer_json["model"]["unk_id"]
                unk_token = tokenizer_json["model"]["vocab"][unk_id][0]
                if special_tokens_map and unk_token in special_tokens_map:
                    unk_token = special_tokens_map[unk_token]
                tokenizer_json["model"]["unk_id"] = 0
                tokenizer_json["model"]["vocab"] = [[unk_token, 0.0]]
        elif tokenizer_json["model"]["type"] in ["WordLevel", "WordPiece"]:
            tokenizer_json["model"]["vocab"] = {}
        else:
            raise ValueError(
                f"train_new_from_iterator not supported for {tokenizer_json['model']['type']} tokenizer type."
            )

        # Possibly rename the model's unknown token if `special_tokens_map` says so
        if (
            special_tokens_map
            and "unk_token" in tokenizer_json["model"]
            and tokenizer_json["model"]["unk_token"] in special_tokens_map
        ):
            tokenizer_json["model"]["unk_token"] = special_tokens_map[tokenizer_json["model"]["unk_token"]]

        # Recreate a blank tokenizer
        tokenizer = TokenizerFast.from_str(json.dumps(tokenizer_json))

        # Keep only "special" added tokens
        special_tokens = []
        for atok in added_tokens:
            special = atok.pop("special", None)
            _ = atok.pop("id", None)
            if tokenizer_json["model"]["type"] != "Unigram" and not special:
                continue
            if special_tokens_map and atok["content"] in special_tokens_map:
                atok["content"] = special_tokens_map[atok["content"]]
            special_tokens.append(AddedToken(**atok))

        if new_special_tokens is not None:
            special_tokens.extend(new_special_tokens)

        # Transfer subword prefix/suffix for BPE
        if tokenizer_json["model"]["type"] == "BPE":
            if (
                "continuing_subword_prefix" not in kwargs
                and tokenizer_json["model"]["continuing_subword_prefix"] is not None
            ):
                kwargs["continuing_subword_prefix"] = tokenizer_json["model"]["continuing_subword_prefix"]
            if (
                "end_of_word_suffix" not in kwargs
                and tokenizer_json["model"]["end_of_word_suffix"] is not None
            ):
                kwargs["end_of_word_suffix"] = tokenizer_json["model"]["end_of_word_suffix"]

        # For Unigram, define the `unk_token`
        if tokenizer_json["model"]["type"] == "Unigram" and unk_token is not None:
            kwargs["unk_token"] = unk_token

        # Possibly set ByteLevel initial alphabet
        if tokenizer_json["pre_tokenizer"] is not None:
            if (
                tokenizer_json["pre_tokenizer"]["type"] == "ByteLevel"
                or (
                    tokenizer_json["pre_tokenizer"]["type"] == "Sequence"
                    and "pretokenizers" in tokenizer_json["pre_tokenizer"]
                    and any(pt["type"] == "ByteLevel" for pt in tokenizer_json["pre_tokenizer"]["pretokenizers"])
                )
            ):
                kwargs["initial_alphabet"] = pre_tokenizers_fast.ByteLevel.alphabet()

        trainer_class = MODEL_TO_TRAINER_MAPPING[tokenizer_json["model"]["type"]]
        trainer = trainer_class(vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
        tokenizer.train_from_iterator(text_iterator, length=length, trainer=trainer)

        # Reattach the post-processor with correct IDs
        if post_processor is not None:
            trained_tokenizer_json = json.loads(tokenizer.to_str())

            if "special_tokens" in post_processor:
                for key in post_processor["special_tokens"]:
                    tokens = post_processor["special_tokens"][key]["tokens"]
                    if special_tokens_map:
                        tokens = [special_tokens_map.get(t, t) for t in tokens]
                    post_processor["special_tokens"][key]["tokens"] = tokens
                    for tok in tokens:
                        if tokenizer.token_to_id(tok) is None:
                            raise ValueError(f"Post-processor token ({tok}) not in vocab!")
                    post_processor["special_tokens"][key]["ids"] = [tokenizer.token_to_id(tok) for tok in tokens]

            for sp_token in ["cls", "sep"]:
                if sp_token in post_processor:
                    token, _ = post_processor[sp_token]
                    if special_tokens_map and token in special_tokens_map:
                        token = special_tokens_map[token]
                    token_id = tokenizer.token_to_id(token)
                    if token_id is None:
                        raise ValueError(
                            "Attempted to set a token in the post-processor that doesn't exist in the mapping."
                        )
                    post_processor[sp_token] = [token, token_id]

            trained_tokenizer_json["post_processor"] = post_processor
            tokenizer = TokenizerFast.from_str(json.dumps(trained_tokenizer_json))

        # Build the final ArlowTokenizer
        kwargs = self.init_kwargs.copy()

        # Transfer existing special tokens from the current tokenizer
        special_tokens_list = SpecialTokensMixin.SPECIAL_TOKENS_ATTRIBUTES.copy()
        special_tokens_list.remove("additional_special_tokens")

        for token_name in special_tokens_list:
            if getattr(self, token_name) is not None:
                stoken_val = getattr(self, token_name)
                if special_tokens_map and stoken_val in special_tokens_map:
                    stoken_val = special_tokens_map[stoken_val]

                stoken_full = self._special_tokens_map.get(token_name, None)
                if isinstance(stoken_full, AddedToken):
                    kwargs[token_name] = AddedToken(
                        stoken_val,
                        single_word=stoken_full.single_word,
                        lstrip=stoken_full.lstrip,
                        rstrip=stoken_full.rstrip,
                        normalized=stoken_full.normalized,
                        special=True,
                    )
                else:
                    kwargs[token_name] = stoken_val

        additional_special_tokens = self.additional_special_tokens
        if new_special_tokens is not None:
            additional_special_tokens.extend(new_special_tokens)
        if len(additional_special_tokens) > 0:
            kwargs["additional_special_tokens"] = additional_special_tokens

        # Create a new instance of ArlowTokenizer with the newly trained tokenizer
        return self.__class__(tokenizer_object=tokenizer, **kwargs)
