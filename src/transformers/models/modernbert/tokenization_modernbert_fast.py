from typing import List, Optional, Tuple

from tokenizers import processors

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}


class ModernBertTokenizerFast(PreTrainedTokenizerFast):
    """
    A flexible tokenizer that can operate in both encoder and decoder (causal) modes based on byte-level
    Byte-Pair-Encoding. When is_causal=True, it behaves like a causal decoder tokenizer that always adds BOS tokens.
    Otherwise, it behaves like a standard PreTrainedTokenizerFast.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. The rest of
    the code is heavily taken from the [`GPTNeoXTokenizerFast`] class.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to the tokenizer file.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token.
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether to add a space to the beginning of the input.
        is_causal (`bool`, *optional*, defaults to `False`):
            Whether to operate in causal mode with forced BOS tokens.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="[UNK]",
        bos_token="[CLS]",
        eos_token="[SEP]",
        pad_token=None,
        add_eos_token=False,
        add_prefix_space=False,
        is_causal=False,
        **kwargs,
    ):
        self.is_causal = is_causal
        if is_causal:
            # Force add_bos_token to True for causal mode
            kwargs.pop("add_bos_token", None)
            add_bos_token = True
        else:
            add_bos_token = kwargs.pop("add_bos_token", False)

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        if is_causal:
            self._add_bos_token = True
            self._add_eos_token = add_eos_token

            # Define causal-mode specific properties and methods
            def add_bos_token_getter(self):
                """
                `bool`: Whether or not the beginning of sequence token is added. Always True in causal mode.
                """
                return True

            def add_bos_token_setter(self, value):
                """Ignore attempts to change add_bos_token in causal mode."""
                self._add_bos_token = True
                self.update_post_processor()

            def add_eos_token_getter(self):
                """
                `bool`: Whether or not the end of sequence token is added.
                """
                return self._add_eos_token

            def add_eos_token_setter(self, value):
                self._add_eos_token = value
                self.update_post_processor()

            def update_post_processor(self):
                """
                Updates the underlying post processor with the current `bos_token` and `eos_token`.
                """
                bos = self.bos_token
                bos_token_id = self.bos_token_id
                if bos is None:
                    raise ValueError("add_bos_token = True but bos_token = None")

                eos = self.eos_token
                eos_token_id = self.eos_token_id
                if eos is None and self._add_eos_token:
                    raise ValueError("add_eos_token = True but eos_token = None")

                single = f"{bos}:0 $A:0{(' '+eos+':0') if self._add_eos_token else ''}"
                pair = f"{single} {bos}:1 $B:1{(' '+eos+':1') if self._add_eos_token else ''}"

                special_tokens = [(bos, bos_token_id)]
                if self._add_eos_token:
                    special_tokens.append((eos, eos_token_id))

                self._tokenizer.post_processor = processors.TemplateProcessing(
                    single=single, pair=pair, special_tokens=special_tokens
                )

            def get_special_tokens_mask(
                self,
                token_ids_0: List[int],
                token_ids_1: Optional[List[int]] = None,
                already_has_special_tokens: bool = False,
            ) -> List[int]:
                """
                Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
                special tokens using the tokenizer `prepare_for_model` method.

                Args:
                    token_ids_0 (`List[int]`):
                        List of IDs.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                    already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                        Whether or not the token list is already formatted with special tokens for the model.

                Returns:
                    `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
                """
                if already_has_special_tokens:
                    return super().get_special_tokens_mask(
                        token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                    )

                # BOS token is always added in causal mode
                bos_token_id = [1]
                eos_token_id = [1] if self._add_eos_token else []

                if token_ids_1 is None:
                    return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id

                return (
                    bos_token_id
                    + ([0] * len(token_ids_0))
                    + eos_token_id
                    + bos_token_id
                    + ([0] * len(token_ids_1))
                    + eos_token_id
                )

            def build_inputs_with_special_tokens(
                self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
            ) -> List[int]:
                """
                Build model inputs from a sequence or a pair of sequence by concatenating and adding special tokens.

                Args:
                    token_ids_0 (`List[int]`):
                        List of IDs to which the special tokens will be added.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.

                Returns:
                    `List[int]`: List of input IDs with the appropriate special tokens.
                """
                # BOS token is always added in causal mode
                bos_token_id = [self.bos_token_id]
                eos_token_id = [self.eos_token_id] if self._add_eos_token else []

                output = bos_token_id + token_ids_0 + eos_token_id

                if token_ids_1 is not None:
                    output = output + bos_token_id + token_ids_1 + eos_token_id

                return output

            def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
                """
                Save the vocabulary and special tokens file to a directory.

                Args:
                    save_directory (`str`):
                        The directory in which to save the vocabulary.
                    filename_prefix (`str`, *optional*):
                        An optional prefix to add to the named of the saved files.

                Returns:
                    `Tuple[str]`: Paths to the files saved.
                """
                files = self._tokenizer.model.save(save_directory, name=filename_prefix)
                return tuple(files)

            # Assign the property getters and setters
            type(self).add_bos_token = property(add_bos_token_getter, add_bos_token_setter)
            type(self).add_eos_token = property(add_eos_token_getter, add_eos_token_setter)

            # Assign the methods
            self.update_post_processor = update_post_processor.__get__(self)
            self.get_special_tokens_mask = get_special_tokens_mask.__get__(self)
            self.build_inputs_with_special_tokens = build_inputs_with_special_tokens.__get__(self)
            self.save_vocabulary = save_vocabulary.__get__(self)

            # Initialize the post processor
            self.update_post_processor()


__all__ = ["ModernBertTokenizerFast"]
