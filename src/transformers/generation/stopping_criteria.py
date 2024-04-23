import time
import warnings
from abc import ABC
from copy import deepcopy
from typing import List, Optional, Union

import torch

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
        if input_ids.device.type == "mps":
            # https://github.com/pytorch/pytorch/issues/77764#issuecomment-2067838075
            is_done = (
                input_ids[:, -1]
                .tile(self.eos_token_id.shape[0], 1)
                .eq(self.eos_token_id.unsqueeze(1).to(input_ids.device))
                .sum(dim=0)
                .bool()
                .squeeze()
            )
        else:
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
