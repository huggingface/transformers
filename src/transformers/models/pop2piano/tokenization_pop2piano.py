# coding=utf-8
# Copyright 2023 The Pop2Piano Authors and The HuggingFace Inc. team.
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
"""Tokenization class for Pop2Piano."""

import json
import os
from typing import Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
from ...utils import TensorType, is_pretty_midi_available, logging, requires_backends, to_numpy
from ...utils.import_utils import requires


if is_pretty_midi_available():
    import pretty_midi

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab": "vocab.json",
}


def token_time_to_note(number, cutoff_time_idx, current_idx):
    current_idx += number
    if cutoff_time_idx is not None:
        current_idx = min(current_idx, cutoff_time_idx)

    return current_idx


def token_note_to_note(number, current_velocity, default_velocity, note_onsets_ready, current_idx, notes):
    if note_onsets_ready[number] is not None:
        # offset with onset
        onset_idx = note_onsets_ready[number]
        if onset_idx < current_idx:
            # Time shift after previous note_on
            offset_idx = current_idx
            notes.append([onset_idx, offset_idx, number, default_velocity])
            onsets_ready = None if current_velocity == 0 else current_idx
            note_onsets_ready[number] = onsets_ready
    else:
        note_onsets_ready[number] = current_idx
    return notes


@requires(backends=("pretty_midi", "torch"))
class Pop2PianoTokenizer(PreTrainedTokenizer):
    """
    Constructs a Pop2Piano tokenizer. This tokenizer does not require training.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab (`str`):
            Path to the vocab file which contains the vocabulary.
        default_velocity (`int`, *optional*, defaults to 77):
            Determines the default velocity to be used while creating midi Notes.
        num_bars (`int`, *optional*, defaults to 2):
            Determines cutoff_time_idx in for each token.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"-1"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 1):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 0):
             A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 2):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    """

    model_input_names = ["token_ids", "attention_mask"]
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab,
        default_velocity=77,
        num_bars=2,
        unk_token="-1",
        eos_token="1",
        pad_token="0",
        bos_token="2",
        **kwargs,
    ):
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token

        self.default_velocity = default_velocity
        self.num_bars = num_bars

        # Load the vocab
        with open(vocab, "rb") as file:
            self.encoder = json.load(file)

        # create mappings for encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
            bos_token=bos_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return len(self.encoder)

    def get_vocab(self):
        """Returns the vocabulary of the tokenizer."""
        return dict(self.encoder, **self.added_tokens_encoder)

    def _convert_id_to_token(self, token_id: int) -> list:
        """
        Decodes the token ids generated by the transformer into notes.

        Args:
            token_id (`int`):
                This denotes the ids generated by the transformers to be converted to Midi tokens.

        Returns:
            `List`: A list consists of token_type (`str`) and value (`int`).
        """

        token_type_value = self.decoder.get(token_id, f"{self.unk_token}_TOKEN_TIME")
        token_type_value = token_type_value.split("_")
        token_type, value = "_".join(token_type_value[1:]), int(token_type_value[0])

        return [token_type, value]

    def _convert_token_to_id(self, token, token_type="TOKEN_TIME") -> int:
        """
        Encodes the Midi tokens to transformer generated token ids.

        Args:
            token (`int`):
                This denotes the token value.
            token_type (`str`):
                This denotes the type of the token. There are four types of midi tokens such as "TOKEN_TIME",
                "TOKEN_VELOCITY", "TOKEN_NOTE" and "TOKEN_SPECIAL".

        Returns:
            `int`: returns the id of the token.
        """
        return self.encoder.get(f"{token}_{token_type}", int(self.unk_token))

    def relative_batch_tokens_ids_to_notes(
        self,
        tokens: np.ndarray,
        beat_offset_idx: int,
        bars_per_batch: int,
        cutoff_time_idx: int,
    ):
        """
        Converts relative tokens to notes which are then used to generate pretty midi object.

        Args:
            tokens (`numpy.ndarray`):
                Tokens to be converted to notes.
            beat_offset_idx (`int`):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`):
                Denotes the cutoff time index for each note in generated Midi.
        """

        notes = None

        for index in range(len(tokens)):
            _tokens = tokens[index]
            _start_idx = beat_offset_idx + index * bars_per_batch * 4
            _cutoff_time_idx = cutoff_time_idx + _start_idx
            _notes = self.relative_tokens_ids_to_notes(
                _tokens,
                start_idx=_start_idx,
                cutoff_time_idx=_cutoff_time_idx,
            )

            if len(_notes) == 0:
                pass
            elif notes is None:
                notes = _notes
            else:
                notes = np.concatenate((notes, _notes), axis=0)

        if notes is None:
            return []
        return notes

    def relative_batch_tokens_ids_to_midi(
        self,
        tokens: np.ndarray,
        beatstep: np.ndarray,
        beat_offset_idx: int = 0,
        bars_per_batch: int = 2,
        cutoff_time_idx: int = 12,
    ):
        """
        Converts tokens to Midi. This method calls `relative_batch_tokens_ids_to_notes` method to convert batch tokens
        to notes then uses `notes_to_midi` method to convert them to Midi.

        Args:
            tokens (`numpy.ndarray`):
                Denotes tokens which alongside beatstep will be converted to Midi.
            beatstep (`np.ndarray`):
                We get beatstep from feature extractor which is also used to get Midi.
            beat_offset_idx (`int`, *optional*, defaults to 0):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`, *optional*, defaults to 2):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`, *optional*, defaults to 12):
                Denotes the cutoff time index for each note in generated Midi.
        """
        beat_offset_idx = 0 if beat_offset_idx is None else beat_offset_idx
        notes = self.relative_batch_tokens_ids_to_notes(
            tokens=tokens,
            beat_offset_idx=beat_offset_idx,
            bars_per_batch=bars_per_batch,
            cutoff_time_idx=cutoff_time_idx,
        )
        midi = self.notes_to_midi(notes, beatstep, offset_sec=beatstep[beat_offset_idx])
        return midi

    # Taken from the original code
    # Please see https://github.com/sweetcocoa/pop2piano/blob/fac11e8dcfc73487513f4588e8d0c22a22f2fdc5/midi_tokenizer.py#L257
    def relative_tokens_ids_to_notes(
        self, tokens: np.ndarray, start_idx: float, cutoff_time_idx: Optional[float] = None
    ):
        """
        Converts relative tokens to notes which will then be used to create Pretty Midi objects.

        Args:
            tokens (`numpy.ndarray`):
                Relative Tokens which will be converted to notes.
            start_idx (`float`):
                A parameter which denotes the starting index.
            cutoff_time_idx (`float`, *optional*):
                A parameter used while converting tokens to notes.
        """
        words = [self._convert_id_to_token(token) for token in tokens]

        current_idx = start_idx
        current_velocity = 0
        note_onsets_ready = [None for i in range(sum([k.endswith("NOTE") for k in self.encoder]) + 1)]
        notes = []
        for token_type, number in words:
            if token_type == "TOKEN_SPECIAL":
                if number == 1:
                    break
            elif token_type == "TOKEN_TIME":
                current_idx = token_time_to_note(
                    number=number, cutoff_time_idx=cutoff_time_idx, current_idx=current_idx
                )
            elif token_type == "TOKEN_VELOCITY":
                current_velocity = number

            elif token_type == "TOKEN_NOTE":
                notes = token_note_to_note(
                    number=number,
                    current_velocity=current_velocity,
                    default_velocity=self.default_velocity,
                    note_onsets_ready=note_onsets_ready,
                    current_idx=current_idx,
                    notes=notes,
                )
            else:
                raise ValueError("Token type not understood!")

        for pitch, note_onset in enumerate(note_onsets_ready):
            # force offset if no offset for each pitch
            if note_onset is not None:
                if cutoff_time_idx is None:
                    cutoff = note_onset + 1
                else:
                    cutoff = max(cutoff_time_idx, note_onset + 1)

                offset_idx = max(current_idx, cutoff)
                notes.append([note_onset, offset_idx, pitch, self.default_velocity])

        if len(notes) == 0:
            return []
        else:
            notes = np.array(notes)
            note_order = notes[:, 0] * 128 + notes[:, 1]
            notes = notes[note_order.argsort()]
            return notes

    def notes_to_midi(self, notes: np.ndarray, beatstep: np.ndarray, offset_sec: int = 0.0):
        """
        Converts notes to Midi.

        Args:
            notes (`numpy.ndarray`):
                This is used to create Pretty Midi objects.
            beatstep (`numpy.ndarray`):
                This is the extrapolated beatstep that we get from feature extractor.
            offset_sec (`int`, *optional*, defaults to 0.0):
                This represents the offset seconds which is used while creating each Pretty Midi Note.
        """

        requires_backends(self, ["pretty_midi"])

        new_pm = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
        new_inst = pretty_midi.Instrument(program=0)
        new_notes = []

        for onset_idx, offset_idx, pitch, velocity in notes:
            new_note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=beatstep[onset_idx] - offset_sec,
                end=beatstep[offset_idx] - offset_sec,
            )
            new_notes.append(new_note)
        new_inst.notes = new_notes
        new_pm.instruments.append(new_inst)
        new_pm.remove_invalid_notes()
        return new_pm

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        """
        Saves the tokenizer's vocabulary dictionary to the provided save_directory.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
            filename_prefix (`Optional[str]`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # Save the encoder.
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"]
        )
        with open(out_vocab_file, "w") as file:
            file.write(json.dumps(self.encoder))

        return (out_vocab_file,)

    def encode_plus(
        self,
        notes: Union[np.ndarray, list[pretty_midi.Note]],
        truncation_strategy: Optional[TruncationStrategy] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        This is the `encode_plus` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer
        generated token ids. It only works on a single batch, to process multiple batches please use
        `batch_encode_plus` or `__call__` method.

        Args:
            notes (`numpy.ndarray` of shape `[sequence_length, 4]` or `list` of `pretty_midi.Note` objects):
                This represents the midi notes. If `notes` is a `numpy.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`], *optional*):
                Indicates the truncation strategy that is going to be used during truncation.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).

        Returns:
            `BatchEncoding` containing the tokens ids.
        """

        requires_backends(self, ["pretty_midi"])

        # check if notes is a pretty_midi object or not, if yes then extract the attributes and put them into a numpy
        # array.
        if isinstance(notes[0], pretty_midi.Note):
            notes = np.array(
                [[each_note.start, each_note.end, each_note.pitch, each_note.velocity] for each_note in notes]
            ).reshape(-1, 4)

        # to round up all the values to the closest int values.
        notes = np.round(notes).astype(np.int32)
        max_time_idx = notes[:, :2].max()

        times = [[] for i in range(max_time_idx + 1)]
        for onset, offset, pitch, velocity in notes:
            times[onset].append([pitch, velocity])
            times[offset].append([pitch, 0])

        tokens = []
        current_velocity = 0
        for i, time in enumerate(times):
            if len(time) == 0:
                continue
            tokens.append(self._convert_token_to_id(i, "TOKEN_TIME"))
            for pitch, velocity in time:
                velocity = int(velocity > 0)
                if current_velocity != velocity:
                    current_velocity = velocity
                    tokens.append(self._convert_token_to_id(velocity, "TOKEN_VELOCITY"))
                tokens.append(self._convert_token_to_id(pitch, "TOKEN_NOTE"))

        total_len = len(tokens)

        # truncation
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            tokens, _, _ = self.truncate_sequences(
                ids=tokens,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                **kwargs,
            )

        return BatchEncoding({"token_ids": tokens})

    def batch_encode_plus(
        self,
        notes: Union[np.ndarray, list[pretty_midi.Note]],
        truncation_strategy: Optional[TruncationStrategy] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        This is the `batch_encode_plus` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer
        generated token ids. It works on multiple batches by calling `encode_plus` multiple times in a loop.

        Args:
            notes (`numpy.ndarray` of shape `[batch_size, sequence_length, 4]` or `list` of `pretty_midi.Note` objects):
                This represents the midi notes. If `notes` is a `numpy.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`], *optional*):
                Indicates the truncation strategy that is going to be used during truncation.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).

        Returns:
            `BatchEncoding` containing the tokens ids.
        """

        encoded_batch_token_ids = []
        for i in range(len(notes)):
            encoded_batch_token_ids.append(
                self.encode_plus(
                    notes[i],
                    truncation_strategy=truncation_strategy,
                    max_length=max_length,
                    **kwargs,
                )["token_ids"]
            )

        return BatchEncoding({"token_ids": encoded_batch_token_ids})

    def __call__(
        self,
        notes: Union[
            np.ndarray,
            list[pretty_midi.Note],
            list[list[pretty_midi.Note]],
        ],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        This is the `__call__` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer generated
        token ids.

        Args:
            notes (`numpy.ndarray` of shape `[batch_size, max_sequence_length, 4]` or `list` of `pretty_midi.Note` objects):
                This represents the midi notes.

                If `notes` is a `numpy.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `False`):
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
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.

        Returns:
            `BatchEncoding` containing the token_ids.
        """

        # check if it is batched or not
        # it is batched if its a list containing a list of `pretty_midi.Notes` where the outer list contains all the
        # batches and the inner list contains all Notes for a single batch. Otherwise if np.ndarray is passed it will be
        # considered batched if it has shape of `[batch_size, sequence_length, 4]` or ndim=3.
        is_batched = notes.ndim == 3 if isinstance(notes, np.ndarray) else isinstance(notes[0], list)

        # get the truncation and padding strategy
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        if is_batched:
            # If the user has not explicitly mentioned `return_attention_mask` as False, we change it to True
            return_attention_mask = True if return_attention_mask is None else return_attention_mask
            token_ids = self.batch_encode_plus(
                notes=notes,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                **kwargs,
            )
        else:
            token_ids = self.encode_plus(
                notes=notes,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                **kwargs,
            )

        # since we already have truncated sequnences we are just left to do padding
        token_ids = self.pad(
            token_ids,
            padding=padding_strategy,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return token_ids

    def batch_decode(
        self,
        token_ids,
        feature_extractor_output: BatchFeature,
        return_midi: bool = True,
    ):
        r"""
        This is the `batch_decode` method for `Pop2PianoTokenizer`. It converts the token_ids generated by the
        transformer to midi_notes and returns them.

        Args:
            token_ids (`Union[np.ndarray, torch.Tensor, tf.Tensor]`):
                Output token_ids of `Pop2PianoConditionalGeneration` model.
            feature_extractor_output (`BatchFeature`):
                Denotes the output of `Pop2PianoFeatureExtractor.__call__`. It must contain `"beatstep"` and
                `"extrapolated_beatstep"`. Also `"attention_mask_beatsteps"` and
                `"attention_mask_extrapolated_beatstep"`
                 should be present if they were returned by the feature extractor.
            return_midi (`bool`, *optional*, defaults to `True`):
                Whether to return midi object or not.
        Returns:
            If `return_midi` is True:
                - `BatchEncoding` containing both `notes` and `pretty_midi.pretty_midi.PrettyMIDI` objects.
            If `return_midi` is False:
                - `BatchEncoding` containing `notes`.
        """

        # check if they have attention_masks(attention_mask, attention_mask_beatsteps, attention_mask_extrapolated_beatstep) or not
        attention_masks_present = bool(
            hasattr(feature_extractor_output, "attention_mask")
            and hasattr(feature_extractor_output, "attention_mask_beatsteps")
            and hasattr(feature_extractor_output, "attention_mask_extrapolated_beatstep")
        )

        # if we are processing batched inputs then we must need attention_masks
        if not attention_masks_present and feature_extractor_output["beatsteps"].shape[0] > 1:
            raise ValueError(
                "attention_mask, attention_mask_beatsteps and attention_mask_extrapolated_beatstep must be present "
                "for batched inputs! But one of them were not present."
            )

        # check for length mismatch between inputs_embeds, beatsteps and extrapolated_beatstep
        if attention_masks_present:
            # since we know about the number of examples in token_ids from attention_mask
            if (
                sum(feature_extractor_output["attention_mask"][:, 0] == 0)
                != feature_extractor_output["beatsteps"].shape[0]
                or feature_extractor_output["beatsteps"].shape[0]
                != feature_extractor_output["extrapolated_beatstep"].shape[0]
            ):
                raise ValueError(
                    "Length mistamtch between token_ids, beatsteps and extrapolated_beatstep! Found "
                    f"token_ids length - {token_ids.shape[0]}, beatsteps shape - {feature_extractor_output['beatsteps'].shape[0]} "
                    f"and extrapolated_beatsteps shape - {feature_extractor_output['extrapolated_beatstep'].shape[0]}"
                )
            if feature_extractor_output["attention_mask"].shape[0] != token_ids.shape[0]:
                raise ValueError(
                    f"Found attention_mask of length - {feature_extractor_output['attention_mask'].shape[0]} but token_ids of length - {token_ids.shape[0]}"
                )
        else:
            # if there is no attention mask present then it's surely a single example
            if (
                feature_extractor_output["beatsteps"].shape[0] != 1
                or feature_extractor_output["extrapolated_beatstep"].shape[0] != 1
            ):
                raise ValueError(
                    "Length mistamtch of beatsteps and extrapolated_beatstep! Since attention_mask is not present the number of examples must be 1, "
                    f"But found beatsteps length - {feature_extractor_output['beatsteps'].shape[0]}, extrapolated_beatsteps length - {feature_extractor_output['extrapolated_beatstep'].shape[0]}."
                )

        if attention_masks_present:
            # check for zeros(since token_ids are separated by zero arrays)
            batch_idx = np.where(feature_extractor_output["attention_mask"][:, 0] == 0)[0]
        else:
            batch_idx = [token_ids.shape[0]]

        notes_list = []
        pretty_midi_objects_list = []
        start_idx = 0
        for index, end_idx in enumerate(batch_idx):
            each_tokens_ids = token_ids[start_idx:end_idx]
            # check where the whole example ended by searching for eos_token_id and getting the upper bound
            each_tokens_ids = each_tokens_ids[:, : np.max(np.where(each_tokens_ids == int(self.eos_token))[1]) + 1]
            beatsteps = feature_extractor_output["beatsteps"][index]
            extrapolated_beatstep = feature_extractor_output["extrapolated_beatstep"][index]

            # if attention mask is present then mask out real array/tensor
            if attention_masks_present:
                attention_mask_beatsteps = feature_extractor_output["attention_mask_beatsteps"][index]
                attention_mask_extrapolated_beatstep = feature_extractor_output[
                    "attention_mask_extrapolated_beatstep"
                ][index]
                beatsteps = beatsteps[: np.max(np.where(attention_mask_beatsteps == 1)[0]) + 1]
                extrapolated_beatstep = extrapolated_beatstep[
                    : np.max(np.where(attention_mask_extrapolated_beatstep == 1)[0]) + 1
                ]

            each_tokens_ids = to_numpy(each_tokens_ids)
            beatsteps = to_numpy(beatsteps)
            extrapolated_beatstep = to_numpy(extrapolated_beatstep)

            pretty_midi_object = self.relative_batch_tokens_ids_to_midi(
                tokens=each_tokens_ids,
                beatstep=extrapolated_beatstep,
                bars_per_batch=self.num_bars,
                cutoff_time_idx=(self.num_bars + 1) * 4,
            )

            for note in pretty_midi_object.instruments[0].notes:
                note.start += beatsteps[0]
                note.end += beatsteps[0]
                notes_list.append(note)

            pretty_midi_objects_list.append(pretty_midi_object)
            start_idx += end_idx + 1  # 1 represents the zero array

        if return_midi:
            return BatchEncoding({"notes": notes_list, "pretty_midi_objects": pretty_midi_objects_list})

        return BatchEncoding({"notes": notes_list})


__all__ = ["Pop2PianoTokenizer"]
