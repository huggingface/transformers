import time
import warnings
from abc import ABC
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import add_start_docstrings, logging


logger = logging.get_logger(__name__)


STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
            make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional stopping criteria specific kwargs.

    Return:
        `torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`), where `True` indicates we stop generation
            for a particular row, `True` indicates we should continue.

"""


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    """

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
        max_position_embeddings (`int`, *optional*):
            The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
    """

    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)


class MaxNewTokensCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        start_length (`int`):
            The number of initial tokens.
        max_new_tokens (`int`):
            The maximum number of tokens to generate.
    """

    def __init__(self, start_length: int, max_new_tokens: int):
        warnings.warn(
            "The class `MaxNewTokensCriteria` is deprecated. "
            f"Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` "
            "with `max_length = start_length + max_new_tokens` instead.",
            FutureWarning,
        )
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = input_ids.shape[-1] >= self.max_length
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.

    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = time.time() - self.initial_timestamp > self.max_time
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)


class StopStringCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever specific string sequences are encountered. It preprocesses
    the strings together with the tokenizer vocab to find positions where tokens can validly complete the stop strings.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The model's associated tokenizer (necessary to extract vocab and tokenize the termination sequences)
        stop_strings (`Union[str, List[str]]`):
            A list of strings that should end generation. If a string is passed, it will be treated like a
            list with a single element.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_strings: Union[str, List[str]]):
        if isinstance(stop_strings, str):
            stop_strings = [stop_strings]

        self.vocab = tokenizer.get_vocab()
        self.stop_strings: List[str] = stop_strings
        self.token_valid_positions, self.token_end_overlaps = self._get_matching_positions()

        self.max_valid_positions = {
            stop_string: max([len(val) for val in self.token_valid_positions[stop_string].values()])
            for stop_string in stop_strings
        }
        self.max_valid_end_lens = {
            stop_string: max([len(val) for val in self.token_end_overlaps[stop_string].values()])
            for stop_string in stop_strings
        }
        self.embedding_vecs = self._create_embedding_vecs()

    def _get_matching_positions(self) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]]]:
        """This function preprocesses stop strings and the tokenizer vocabulary to determine where tokens can
        validly appear in the stop strings. For each stop string, it returns a dictionary mapping tokens to a list of
        valid positions, as well as a dictionary mapping tokens to a list of possible overlap lengths at the
        end of the stop string."""
        tok_list = list(self.vocab.keys())
        reversed_filtered_tok_list = [token[::-1].replace("▁", " ").replace("Ġ", " ") for token in tok_list]
        token_valid_positions = {}
        token_end_overlaps = {}
        for stop_string in self.stop_strings:
            reversed_stop_string = stop_string[::-1]
            token_valid_positions[stop_string] = {}
            token_end_overlaps[stop_string] = {}
            for token, reversed_filtered_token in zip(tok_list, reversed_filtered_tok_list):
                matching_positions = []
                possible_end_lengths = []
                for i in range(1 - len(token), len(stop_string)):
                    tok = reversed_filtered_token
                    stop = reversed_stop_string
                    if i < 0:
                        tok = tok[-i:]
                        i = 0
                    stop = stop[i : i + len(tok)]
                    if len(tok) > len(stop):
                        tok = tok[: len(stop)]
                    if tok == stop:
                        if i == 0:
                            possible_end_lengths.append(len(tok))
                        else:
                            matching_positions.append(i)
                if matching_positions:
                    token_valid_positions[stop_string][token] = matching_positions
                if possible_end_lengths:
                    token_end_overlaps[stop_string][token] = possible_end_lengths
        return token_valid_positions, token_end_overlaps

    def _create_embedding_vecs(self) -> Dict[str, torch.tensor]:
        """
        This function builds an embedding matrix for each stop string, consisting of possible valid positions
        and possible end lengths for each token, and the total length of the token string. When tokens have
        fewer valid positions or end lengths than the maximum, we pad the vectors with -1.
        """
        # TODO Matt: Merge the embeddings across all stop strings to save space and reduce gather calls?
        vocab = self.vocab
        embedding_vecs = {}
        for stop_string in self.stop_strings:
            positions = self.token_valid_positions[stop_string]
            end_lens = self.token_end_overlaps[stop_string]
            max_valid_positions = self.max_valid_positions[stop_string]
            max_valid_end_lens = self.max_valid_end_lens[stop_string]
            vec_size = max_valid_positions + max_valid_end_lens + 1
            # Since this is lots of very small assignments of lists, we build it with numpy rather
            # than torch for speed + simplicity, then convert to torch at the end
            gather_vec = np.full((len(self.vocab), vec_size), dtype=np.int32, fill_value=-1)
            for token, valid_positions in positions.items():
                token_idx = vocab[token]
                gather_vec[token_idx, : len(valid_positions)] = valid_positions
            for token, possible_end_lens in end_lens.items():
                token_idx = vocab[token]
                gather_vec[
                    token_idx, max_valid_positions : max_valid_positions + len(possible_end_lens)
                ] = possible_end_lens
            for token, token_idx in vocab.items():
                gather_vec[token_idx, -1] = len(token)
            embedding_vecs[stop_string] = torch.tensor(gather_vec, dtype=torch.int32)
        return embedding_vecs

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # The maximum length we need to consider is 1 token per character. Note that input_ids can also be
        # *shorter* than the global max, and the code below should be ready for that
        maximum_token_len = max([len(stop_string) for stop_string in self.stop_strings])
        input_ids = input_ids[:, -maximum_token_len:]

        # Flip input_ids because we're only matching strings at the end of the generated sequence
        flipped_ids = torch.flip(input_ids, (1,))

        string_matches = []
        for stop_string in self.stop_strings:
            # We need the length of the stop string to know how many characters our token sequence should have
            target_len = len(stop_string)

            # Size of the vector of positions a single token can match
            max_valid_positions = self.max_valid_positions[stop_string]

            # Size of the vector of overlap sizes a single token can have with the end of the string
            max_valid_end_lens = self.max_valid_end_lens[stop_string]

            # The embedding vec contains the valid positions, end_lengths and total lengths for each token
            embedding_vec = self.embedding_vecs[stop_string].to(flipped_ids.device)
            embedded = F.embedding(flipped_ids, embedding_vec)
            # Now we split the embedding vector. valid_positions is the positions in the stop string the token can fit
            valid_positions = embedded[:, 1:, :max_valid_positions]
            # end_lengths is the number of characters from the string, counting from the end, that the token
            # contains. It can have multiple values if the same token can overlap different end lengths
            end_lengths = embedded[:, :1, max_valid_positions : max_valid_positions + max_valid_end_lens]
            # Lengths is the total length of each token. Unlike the others, it always has a single value
            lengths = embedded[:, 1:, -1:]

            # Concatenate lengths onto each possible end_lengths value
            lengths = lengths.expand((-1, -1, end_lengths.shape[-1]))
            lengths_with_ends = torch.cat([end_lengths, lengths], dim=1)

            # cumsum() to get the number of matched characters in the stop string after each token
            cumsum = lengths_with_ends.cumsum(dim=1)  # B x maximum_token_len x max_valid_end_lens

            # The calculation above assumes that all tokens are in valid positions. Now we mask the ones that are not.
            # First, tokens match the start of the string if they have a positive value in the end_lengths vector
            initial_match = end_lengths > 0

            # Tokens continue the string if the cumsum() so far is one of the valid positions for that token
            # Note that we're actually tracking one cumsum() for for each possible end_length
            later_match = torch.any(cumsum[:, :-1, None] == valid_positions[:, :, :, None], axis=2)

            # The match vector is a boolean vector that indicates which positions have valid tokens
            match = torch.cat([initial_match, later_match], dim=1)

            # Once a single position does not match, all positions following that position are masked
            mask = (~match).cumsum(dim=1, dtype=torch.int32)
            mask = mask == 0

            # The string is matched if we reached a cumsum equal to or greater than the length of the string
            # before hitting the mask
            string_matches.append(torch.amax(cumsum * mask, dim=(1, 2)) >= target_len)

        # Now we concatenate the match booleans across all strings and check if any are True
        string_matches = torch.cat(string_matches, dim=0)
        return torch.any(string_matches).item()


class EosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, eos_token_id: Union[int, List[int]]):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = torch.tensor(eos_token_id)

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = torch.isin(input_ids[:, -1], self.eos_token_id.to(input_ids.device))
        return is_done


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = torch.full((input_ids.shape[0],), False, device=input_ids.device)
        for criteria in self:
            is_done = is_done | criteria(input_ids, scores, **kwargs)
        return is_done

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria
