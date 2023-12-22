import math
import torch
import logging
from transformers.generation.logits_process import LogitsProcessor, LOGITS_PROCESSOR_INPUTS_DOCSTRING
from transformers.utils import add_start_docstrings

logger = logging.getLogger(__name__)


class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint):
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_stacks = None

    def filter_logits(self, logits, device):
        # resolve each stack to a tensor of True/False for each token
        # indicating acceptance
        # acceptance = self.grammar_acceptor.filter_vocab(self.stacks, device)
        acceptance = self.grammar_constraint.batch_filter_vocab(self.batch_stacks, device)
        logger.debug(acceptance)
        # Logits to -inf where False
        logits[~acceptance] = -math.inf

    def process_logits(self, input_ids, scores, parse_start_index=None):
        """
        :param input_ids:
        :param scores:
        :param parse_start_index: default None, which means generate from scratch. Set to 0 to parse all input_ids
        :return:
        """
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_stacks is None:
            self.batch_stacks = [self.grammar_constraint.init_stacks() for _ in range(len(input_ids))]

        # if self.last_size is not set (which would be the case when processing the first token).
        # In this case, do nothing.
        if self.last_size is None:
            prefix_to_parse = [
                single_input_ids[parse_start_index:] if parse_start_index is not None else []
                for single_input_ids in input_ids
            ]
            # self.grammar_acceptor.accept_token_ids(prefix_to_parse, self.stacks)
            self.batch_stacks = [
                self.grammar_constraint.accept_token_ids(prefix, stack)
                for prefix, stack in zip(prefix_to_parse, self.batch_stacks)
            ]
        #  if the length of the current input IDs (input_ids[0]) is exactly one more than self.last_size.
        #  This is expected in a scenario where inputs are processed incrementally, one token at a time.
        elif len(input_ids[0]) == self.last_size + 1:
            # self.stacks = self.grammar_acceptor.accept_token_id(input_ids[0][-1], self.stacks)
            self.batch_stacks = [
                self.grammar_constraint.accept_token_id(single_input_ids[-1], stack)
                for single_input_ids, stack in zip(input_ids, self.batch_stacks)
            ]
        #  ensure that the input size is consistent with the expected incremental processing
        #  (i.e., one token at a time).
        else:
            # here we check if the input_ids are one token longer than the last time we processed
            # but we don't check if input_ids are actually valid.
            # Imagine a scenario where we generate 10 tokens, then we replace the 10 generated tokens with 10 new tokens.
            # In this case, the input_ids will be consistent with the last_size, but the input_ids are not valid.
            # However, should we really check if the input_ids are valid here?
            # If we do, then we need to reparse the whole input_ids at each call, which is not efficient.
            # Maybe we should just trust the user to provide valid input_ids?
            # The conclusion is that, we assume the input_ids are valid, and our generation will be correct.
            # If the input_ids are not valid, then the generation result will be wrong and we don't take responsibility for that.
            raise RuntimeError(
                "Input ID's length is inconsistent with the current state of "
                "the GrammarConstrainedLogitsProcessor. If you want to process "
                "another input sequence, please instantiate a new "
                "GrammarConstrainedLogitsProcessor."
            )

        self.filter_logits(scores, scores.device)

        self.last_size = len(input_ids[0])
        return scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.process_logits(input_ids, scores)

