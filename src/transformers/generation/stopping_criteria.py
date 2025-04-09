import time
import warnings
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from ..pytorch_utils import isin_mps_friendly
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import add_start_docstrings, logging


logger = logging.get_logger(__name__)
# We maintain a module-level cache of the embedding vectors for the stop string criterion
# because they are slow to compute
STOP_STRING_EMBEDDING_CACHE = OrderedDict()


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
    This class can be used to stop generation whenever specific string sequences are generated. It preprocesses
    the strings together with the tokenizer vocab to find positions where tokens can validly complete the stop strings.

    Generation is stopped as soon as a token is generated that completes any of the stop strings.
    We want to catch any instance in which the stop string would be present in the decoded output, which means
    we must also catch cases with "overhangs" off one or both ends. To make this more concrete, for the stop string
    "stop", any of the following token sequences would trigger the match:

    - ["st", "op"]
    - ["stop"]
    - ["st", "opera"]
    - ["sto", "pper"]
    - ["las", "topper"]
    - ["s", "to", "pped"]

    Note that a match will only be triggered if the stop string is at the end of the generated sequence. In other
    words, these sequences will not trigger a match:

    - ["stop", "at"]
    - ["st", "op", "at"]
    - ["st", "opera", "tion"]

    The reason these are not a match is that the stop string does not overlap with the final token. If you can remove
    one or more tokens from the end of the sequence without destroying the stop string, then this criterion will not
    match that stop string. This is by design; because this check is run after each token is generated, we can't miss a
    valid stop string if one is generated, but we don't want to halt generation just because the stop string exists
    somewhere in the past input_ids.

    How is the match actually performed, though? We do it in quite a confusing way, because we want the entire match
    process to be compilable with Torch or XLA, which means we cannot use standard string methods. However, it is possible,
    with some work, to do string matching with pure tensor operations. We'll begin by describing the algorithm we use
    with standard string operations, and then at the end we'll explain how this is converted to pure tensor operations.

    The key to the algorithm is an observation: Because the stop string must overlap with the end of the token sequence, we can start at
    the end of the sequence and work backwards. Specifically, we check that there is an overlap between the start of
    the final token and the end of the stop_string, or to put it another way, stop_string[-i:] == token[:i] for
    some i > 0. If you look at the positive examples above, you'll see the last token in all of them fulfills this
    property:

    - ["st", "op"] (overlap is "op", overlap length == 2)
    - ["stop"]  (overlap is "stop", overlap length == 4)
    - ["st", "opera"]  (overlap is "op", overlap length == 2)
    - ["sto", "pper"]  (overlap is "p", overlap length == 1)
    - ["las", "topper"]  (overlap is "top", overlap length == 3)
    - ["s", "to", "pped"]  (overlap is "p", overlap length == 1)

    It's impossible to construct a matching sequence that does not have this property (feel free to verify this
    yourself). However, although this overlap between the start of the final token and the end of the stop string is
    necessary for a match, it is not sufficient. We also need to check that the rest of the token sequence is
    consistent with the stop string.

    How do we do that? Let's use ["s", "to", "pped"] as an example. We know that the final token, "pped", has an
    overlap of 1 with the stop string, "stop". We then go back to the previous token, "to". Since we have already
    matched 1 character from the stop string, the remainder to check is "sto". We check that the next token "to"
    matches the end of the remainder, which it does. We have now matched 3 characters from the stop string, and the
    remainder to match is "s". We go back to the previous token again, which is also "s". This is a match, and so
    we have matched the entire stop string.

    How does it work when the tokens run off the start of the stop string, though? Let's consider the example of
    ["las", "topper"]. The final token, "topper", has an overlap of 3 with the stop string, "stop". Therefore,
    the remaining stop string to match is "s". We go back to the previous token, "las". Because the remainder to
    match is just "s", with length 1, we consider only the final 1 character from the token, which is "s". This
    matches the stop string, and so the entire string is matched.

    How do we compute these matches with tensor operations, though? Simply: we efficiently precompute the necessary
    information for all tokens! For every token, we compute:
    - Its overlap with the end of the stop string, if any
    - The positions inside the stop string where the token matches, including matches that run off the start.
    - The total length of the token

    For example, for the token "pped", we would compute an end overlap of 1, no internal matching positions,
    and a length of 4. For the token "to", we would compute no end overlap, a single internal matching position
    of 1 (counting from the end), and a length of 2. For the token "s", we would compute no end overlap,
    a single internal matching position of 3 (again counting from the end) and a length of 1.

    As long as we have this information, we can execute the algorithm above without any string comparison
    operations. We simply perform the following steps:
    - Check if the final token has an end-overlap with the start string
    - Continue backwards, keeping track of how much of the stop string we've matched so far
    - At each point, check if the next token has the current position as one of its valid positions
    - Continue until either a match fails, or we completely match the whole stop string

    Again, consider ["s", "to", "pped"] as an example. "pped" has an end overlap of 1, so we can begin a match.
    We have matched 1 character so far, so we check that the next token "to", has 1 as a valid position (again,
    counting from the end). It does, so we add the length of "to" to our position tracker. We have now matched
    3 characters, so we check that the next token "s" has 3 as a valid position. It does, so we add its length
    to the position tracker. The position tracker is now 4, which is the length of the stop string. We have matched the
    entire stop string.

    In the second case, ["las", "topper"], "topper" has an end overlap of 3, so we can begin a match. We have
    matched 3 characters so far, so we check that the next token "las" has 3 as a valid position. It does, because we
    allow tokens to match positions that run off the start of the stop string. We add its length to the position
    tracker. The position tracker is now 6, which is greater than the length of the stop string! Don't panic, though -
    this also counts as a match of the stop string. We have matched the entire stop string.


    Args:
        tokenizer (`PreTrainedTokenizer`):
            The model's associated tokenizer (necessary to extract vocab and tokenize the termination sequences)
        stop_strings (`Union[str, List[str]]`):
            A list of strings that should end generation. If a string is passed, it will be treated like a
            list with a single element.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    >>> model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    >>> inputs = tokenizer("The biggest states in the USA by land area:", return_tensors="pt")

    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    The biggest states in the USA by land area:
    - Alaska
    - Texas
    - California

    >>> # Passing one or more stop strings will halt generation after those strings are emitted
    >>> # Note that generating with stop strings requires you to pass the tokenizer too
    >>> gen_out = model.generate(**inputs, stop_strings=["Texas"], tokenizer=tokenizer)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    The biggest states in the USA by land area:
    - Alaska
    - Texas
    ```
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_strings: Union[str, List[str]]):
        if isinstance(stop_strings, str):
            stop_strings = [stop_strings]
        self.stop_strings: Tuple[str, ...] = tuple(stop_strings)
        vocab = tokenizer.get_vocab()
        token_list, token_indices = tuple(vocab.keys()), tuple(vocab.values())
        self.embedding_vec, self.max_valid_positions, self.max_valid_end_lens = self.clean_and_embed_tokens_with_cache(
            token_list, token_indices, tokenizer
        )

        self.maximum_token_len = max([len(stop_string) for stop_string in self.stop_strings])
        self.num_stop_strings = len(self.stop_strings)
        self.target_lens = torch.tensor([len(stop_string) for stop_string in stop_strings], dtype=torch.int32)

    def clean_and_embed_tokens_with_cache(self, token_list, token_indices, tokenizer):
        # We don't use the tokenizer in the cache key, because I don't trust it to have well-behaved equality
        if (token_list, token_indices, self.stop_strings) in STOP_STRING_EMBEDDING_CACHE:
            embedding_vec, max_valid_positions, max_valid_end_lens = STOP_STRING_EMBEDDING_CACHE[
                (token_list, token_indices, self.stop_strings)
            ]
            STOP_STRING_EMBEDDING_CACHE.move_to_end((token_list, token_indices, self.stop_strings))
        else:
            clean_token_list, clean_token_indices = self.clean_tokenizer_vocab(tokenizer)
            embedding_vec, max_valid_positions, max_valid_end_lens = self._stop_string_create_embedding_vec(
                clean_token_list, clean_token_indices, self.stop_strings
            )
            STOP_STRING_EMBEDDING_CACHE[(token_list, token_indices, self.stop_strings)] = (
                embedding_vec,
                max_valid_positions,
                max_valid_end_lens,
            )
            if len(STOP_STRING_EMBEDDING_CACHE) > 8:
                STOP_STRING_EMBEDDING_CACHE.popitem(last=False)  # Pop from the start, the least recently used item
        return embedding_vec, max_valid_positions, max_valid_end_lens

    @staticmethod
    def clean_tokenizer_vocab(tokenizer, static_prefix="abcdef"):
        """
        This method turns a tokenizer vocab into a "clean" vocab where each token represents the actual string
        it will yield, without any special prefixes like "##" or "Ġ". This is trickier than it looks - the method
        tokenizer.convert_tokens_to_string() does not always return the correct string because of issues with prefix
        space addition/removal. To work around this, we add a static prefix to the start of the token, then remove
        it (and any prefix that may have been introduced with it) after calling convert_tokens_to_string().
        """
        vocab = tokenizer.get_vocab()
        clean_token_list = []
        clean_token_indices = []
        sentence_base = tokenizer(static_prefix, add_special_tokens=False)["input_ids"]
        tokens_base = [tokenizer._convert_id_to_token(tok) for tok in sentence_base]
        for token, token_idx in vocab.items():
            token_string = tokenizer.convert_tokens_to_string(tokens_base + [token])
            token_string = token_string[token_string.index(static_prefix) + len(static_prefix) :]
            clean_token_list.append(token_string)
            clean_token_indices.append(token_idx)
        return tuple(clean_token_list), tuple(clean_token_indices)

    @staticmethod
    def _stop_string_get_matching_positions(
        token_list, token_indices, stop_strings
    ) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]]]:
        """This function preprocesses stop strings and the tokenizer vocabulary to determine where tokens can
        validly appear in the stop strings. For each token, it computes a list of positions in the stop string where the
        token appears, as well as a list of the possible "end overlaps" for that token - that is, the number of characters
        from the end of the stop string that overlap with the start of the token, which can have more than one value.

        The reason for computing these may seem a bit cryptic - please see the docstring for StopStringCriteria for a full
        explanation of what these values are for!"""

        token_valid_positions = {}
        token_end_overlaps = {}
        for stop_string in stop_strings:
            reversed_stop_string = stop_string[::-1]
            token_valid_positions[stop_string] = {}
            token_end_overlaps[stop_string] = {}
            for token, tok_idx in zip(token_list, token_indices):
                reversed_token = token[::-1]
                matching_positions = []
                possible_end_lengths = []
                for i in range(1 - len(token), len(stop_string)):
                    if i < 0:
                        tok = reversed_token[-i:]
                        i = 0
                    else:
                        tok = reversed_token
                    stop = reversed_stop_string[i : i + len(tok)]
                    if tok.startswith(stop):
                        if i == 0:
                            possible_end_lengths.append(min(len(tok), len(stop)))
                        else:
                            matching_positions.append(i)

                if matching_positions:
                    token_valid_positions[stop_string][tok_idx] = matching_positions
                if possible_end_lengths:
                    token_end_overlaps[stop_string][tok_idx] = possible_end_lengths
        return token_valid_positions, token_end_overlaps

    @staticmethod
    def _stop_string_create_embedding_vec(token_list, token_indices, stop_strings) -> Dict[str, torch.tensor]:
        """This function precomputes everything needed for the run-time checks in StopStringCriteria, and packs
        them into an embedding tensor that can be accessed with pure tensor operations. For the specifics of the values
        that are precomputed and what they are used for, please refer to the StopStringCriteria docstring!"""
        token_valid_positions, token_end_overlaps = StopStringCriteria._stop_string_get_matching_positions(
            token_list, token_indices, stop_strings
        )
        all_valid_positions = [len(val) for positions in token_valid_positions.values() for val in positions.values()]
        # In some cases, tokens may have no valid internal positions (such as single-character stop strings), so
        # we need a fallback to handle this case
        max_valid_positions = max(all_valid_positions) if all_valid_positions else 1
        # There should always be at least one valid end_len, however, so no fallback needed here
        valid_end_lens = [len(val) for positions in token_end_overlaps.values() for val in positions.values()]
        if not valid_end_lens:
            raise ValueError(
                "Stop string preprocessing was unable to identify tokens matching one or more of the "
                "supplied stop string(s). This is most often caused by the stop "
                "strings containing unusual characters that are not in the tokenizer vocabulary."
            )
        max_valid_end_lens = max(valid_end_lens)
        vec_size = len(stop_strings) * (max_valid_positions + max_valid_end_lens) + 1
        # We use +2 instead of +1 so we can have a dummy entry at the end. We will clamp all token values
        # over the max to this, ensuring they do not contribute to stop string matching.
        gather_vec = np.full((max(token_indices) + 2, vec_size), dtype=np.int32, fill_value=-1)

        for i, stop_string in enumerate(stop_strings):
            positions = token_valid_positions[stop_string]
            end_lens = token_end_overlaps[stop_string]

            # Since this is lots of very small assignments of lists, we build it with numpy rather
            # than torch for speed + simplicity, then convert to torch at the end
            for token_idx, valid_positions in positions.items():
                gather_vec[token_idx, max_valid_positions * i : max_valid_positions * i + len(valid_positions)] = (
                    valid_positions
                )
            for token_idx, possible_end_lens in end_lens.items():
                gather_vec[
                    token_idx,
                    max_valid_positions * len(stop_strings) + max_valid_end_lens * i : max_valid_positions
                    * len(stop_strings)
                    + max_valid_end_lens * i
                    + len(possible_end_lens),
                ] = possible_end_lens
            for token, token_idx in zip(token_list, token_indices):
                gather_vec[token_idx, -1] = len(token)

        gather_vec = torch.tensor(gather_vec, dtype=torch.int32)

        return gather_vec, max_valid_positions, max_valid_end_lens

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.Tensor:
        self.embedding_vec = self.embedding_vec.to(input_ids.device)
        self.target_lens = self.target_lens.to(input_ids.device)
        # The maximum length we need to consider is 1 token per character. Note that input_ids can also be
        # *shorter* than the global max, and the code below should be ready for that
        input_ids = input_ids[:, -self.maximum_token_len :]

        # Flip input_ids because we're only matching strings at the end of the generated sequence
        flipped_ids = torch.flip(input_ids, (1,))

        # Clip out-of-vocab values to the dummy value at the end of the embedding vector
        flipped_ids = torch.clamp(flipped_ids, max=self.embedding_vec.size(0) - 1)

        # Size of the vector of positions a single token can match
        max_valid_positions = self.max_valid_positions

        # The embedding vec contains the valid positions, end_lengths and total lengths for each token
        embedded = F.embedding(flipped_ids, self.embedding_vec)

        # Now we split the embedding vector. valid_positions is the positions in the stop string the token can fit
        valid_positions = embedded[:, 1:, : max_valid_positions * self.num_stop_strings].unflatten(
            -1, (self.num_stop_strings, -1)
        )
        # end_lengths is the number of characters from the string, counting from the end, that the token
        # contains. It can have multiple values if the same token can overlap different end lengths
        end_lengths = embedded[:, :1, max_valid_positions * self.num_stop_strings : -1].unflatten(
            -1, (self.num_stop_strings, -1)
        )
        # Lengths is the total length of each token. Unlike the others, it always has a single value
        lengths = embedded[:, 1:, None, -1:]  # Insert a dummy dimension for stop_strings even though lengths are const

        # Concatenate lengths onto each possible end_lengths value
        lengths = lengths.expand((-1, -1, end_lengths.shape[-2], end_lengths.shape[-1]))
        lengths_with_ends = torch.cat([end_lengths, lengths], dim=1)

        # cumsum() to get the number of matched characters in the stop string after each token
        cumsum = lengths_with_ends.cumsum(dim=1)  # B x maximum_token_len x num_stop_strings x max_valid_end_lens

        # The calculation above assumes that all tokens are in valid positions. Now we mask the ones that are not.
        # First, tokens match the start of the string if they have a positive value in the end_lengths vector
        initial_match = end_lengths > 0

        # Tokens continue the string if the cumsum() so far is one of the valid positions for that token
        # Note that we're actually tracking one cumsum() for for each possible end_length
        later_match = torch.any(cumsum[:, :-1, :, None] == valid_positions[:, :, :, :, None], axis=-2)

        # The match vector is a boolean vector that indicates which positions have valid tokens
        match = torch.cat([initial_match, later_match], dim=1)

        # Once a single position does not match, all positions following that position are masked
        mask = (~match).cumsum(dim=1, dtype=torch.int32)
        mask = mask == 0

        # The string is matched if we reached a cumsum equal to or greater than the length of the string
        # before hitting the mask
        string_matches = torch.amax(cumsum * mask, dim=(1, -1)) >= self.target_lens[None, :]

        # We return a per-sample vector that is True if any stop string is matched for that sample
        return torch.any(string_matches, dim=-1)


class EosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, eos_token_id: Union[int, List[int], torch.Tensor]):
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)
        self.eos_token_id = eos_token_id

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        self.eos_token_id = self.eos_token_id.to(input_ids.device)
        is_done = isin_mps_friendly(input_ids[:, -1], self.eos_token_id)
        return is_done


class ConfidenceCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever assistant model's confidence in its prediction for the current token is lower than the threshold
        `model.generation_config.assistant_confidence_threshold` even if the number of speculative tokens (defined by `num_assistant_tokens`) is not yet reached.

    Args:
        assistant_confidence_threshold (`float`):
            The value of the threshold.
    """

    def __init__(self, assistant_confidence_threshold):
        self.assistant_confidence_threshold = assistant_confidence_threshold

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        probs = scores[-1].softmax(-1)
        p = probs[0, input_ids[0, -1]].item()
        if p < self.assistant_confidence_threshold:
            return True
        return False


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)
        for criteria in self:
            is_done = is_done | criteria(input_ids, scores, **kwargs)
        return is_done

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
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
