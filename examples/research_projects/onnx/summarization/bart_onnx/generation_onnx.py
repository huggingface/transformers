import copy
import itertools
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from transformers import BartConfig
from transformers.generation import GenerationMixin


def _convert_past_list_to_tuple(past_key_values):
    """
    In Bart model, the type of past_key_values is tuple(tuple(torch.FloatTensor)) which is not
    TorchScript-compatible. To support this, we have to convert it during the export process.
    This function will convert past values from a list to tuple(tuple(torch.FloatTensor)) for
    the inner decoder.

    According to the definition of past_key_values, each inner tuple(torch.FloatTensor) has 4 tensors,
    so we convert every 4 elements in the list as a tuple(torch.FloatTensor).
    """
    count_of_each_inner_tuple = 4
    results = ()
    temp_result = ()
    count_n = len(past_key_values) // count_of_each_inner_tuple
    for idx in range(count_n):
        real_idx = idx * count_of_each_inner_tuple
        temp_result = tuple(past_key_values[real_idx : real_idx + count_of_each_inner_tuple])
        results += ((temp_result),)

    return results


class EncoderForONNX(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )


class DecoderForONNX(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids, encoder_state, attention_mask, past=None):
        all_results = None
        if past is not None:
            all_results = _convert_past_list_to_tuple(past)
            input_ids = input_ids[:, -1:]

        last_hidden_state, past_key_values = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_state,
            encoder_attention_mask=attention_mask,
            past_key_values=all_results,
            return_dict=False,
        )

        past_values = []
        for past in past_key_values:
            past_values = past_values + list(past)
        return last_hidden_state, past_values


def _create_traced_encoder(encoder, input_ids, attention_mask):
    encoder_c = copy.deepcopy(encoder)
    encoder_for_onnx = EncoderForONNX(encoder_c)

    return torch.jit.trace(encoder_for_onnx, (input_ids, attention_mask))


def _create_traced_decoder(decoder, input_ids, encoder_state, attention_mask, past=None):
    decoder_c = copy.deepcopy(decoder)
    decoder_for_onnx = DecoderForONNX(decoder_c)
    past_values = list(itertools.chain.from_iterable(past or ()))

    # Do this twice so we got 2 different decoders for further work.
    if past_values:
        return torch.jit.trace(decoder_for_onnx, (input_ids, encoder_state, attention_mask, past_values))
    else:
        return torch.jit.trace(decoder_for_onnx, (input_ids, encoder_state, attention_mask))


class BartConfigTS(BartConfig, torch.nn.Module):
    """
    BartConfigTS is a TorchScript-compatible transformers.models.bart.configuration_bart.BartConfig.
    TorchScript only supports sub-classes of torch.nn.Module.
    """

    def __init__(self, config):
        BartConfig.__init__(self, config)
        torch.nn.Module.__init__(self)


class MinLengthLogitsProcessorTS(torch.nn.Module):
    r"""
    :class:`transformers.LogitsProcessor` enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        super().__init__()

        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def forward(self, input_ids, scores) -> torch.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores


class BARTGenerator(torch.nn.Module, GenerationMixin):
    def __init__(self, model):
        super().__init__()
        self.config = BartConfigTS(model.config)
        self.config.force_bos_token_to_be_generated = False
        self._trace_modules(model)
        self.logits_processor = MinLengthLogitsProcessorTS(self.config.min_length, self.config.eos_token_id)
        self.final_logits_weight = model.model.shared.weight
        self.final_logits_bias = model.final_logits_bias
        self.decoder_layers = model.config.decoder_layers

    def _trace_modules(self, model):
        input_ids = torch.tensor(
            [
                [
                    19,
                    669,
                    18,
                    420,
                    8,
                    664,
                    57,
                    42,
                    8,
                    664,
                    21,
                    3028,
                    195,
                    4445,
                    331,
                    1293,
                    34,
                    21,
                    10,
                    6174,
                    1100,
                    6,
                    69,
                    104,
                    42,
                    32,
                    2621,
                    1638,
                    144,
                    4,
                    6174,
                    558,
                    108,
                    4419,
                    1091,
                    28,
                    4,
                    1668,
                    9,
                    1509,
                    1621,
                    279,
                    35,
                    867,
                    2734,
                    85,
                    11,
                    2216,
                    2734,
                    85,
                    203,
                    2244,
                    7,
                    6,
                    15,
                    8102,
                    7,
                    57,
                    8629,
                    5,
                    model.config.eos_token_id,
                ]
            ],
            device=model.device,
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [[True] * input_ids.shape[-1]],
            device=model.device,
            dtype=torch.bool,
        )
        self.encoder = _create_traced_encoder(model.get_encoder(), input_ids, attention_mask)
        encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask, return_dict=True)
        decoder = model.model.decoder
        decoder_outputs = decoder(input_ids, attention_mask, encoder_outputs["last_hidden_state"], None, None, None)
        self.decoder_no_past = _create_traced_decoder(
            model.model.decoder, input_ids, encoder_outputs["last_hidden_state"], attention_mask
        )
        self.decoder_with_past = _create_traced_decoder(
            model.model.decoder, input_ids, encoder_outputs["last_hidden_state"], attention_mask, decoder_outputs[1]
        )

    def _encoder_forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask)[0]

    @staticmethod
    def _init_sequence_length_for_generation(
        input_ids: torch.LongTensor, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        unfinished_sequences = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device) + 1
        sequence_lengths = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device) + max_length

        cur_len = input_ids.shape[-1]
        return sequence_lengths, unfinished_sequences, cur_len

    def _decoder_forward(self, input_ids, encoder_output, attention_mask, past: List[torch.Tensor]):
        # Update here to use different decoder for different values of past.
        if past is None or len(past) == 0:
            decoder_output, past = self.decoder_no_past(
                input_ids=input_ids, encoder_state=encoder_output, attention_mask=attention_mask
            )
        else:
            decoder_output, past = self.decoder_with_past(
                input_ids=input_ids, encoder_state=encoder_output, attention_mask=attention_mask, past=past
            )

        lm_logits = F.linear(decoder_output, self.final_logits_weight, bias=self.final_logits_bias)

        return lm_logits, past

    def greedy_search(
        self, input_ids, encoder_output, attention_mask, max_length, pad_token_id: int, eos_token_id: int
    ):
        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        past: List[torch.Tensor] = []
        while cur_len < max_length:
            logits, past = self._decoder_forward(input_ids, encoder_output, attention_mask, past)
            next_token_logits = logits[:, -1, :]

            # pre-process distribution
            scores = self.logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(scores, dim=-1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # increase cur_len
            cur_len = cur_len + 1

        return input_ids

    def _prepare_decoder_input_ids_for_generation(
        self,
        input_ids: torch.LongTensor,
        decoder_start_token_id,
        bos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
            * decoder_start_token_id
        )
        return decoder_input_ids

    def forward(self, input_ids, attention_mask, max_length, decoder_start_token_id):
        pad_token_id = self.config.pad_token_id
        bos_token_id = self.config.bos_token_id
        eos_token_id = self.config.eos_token_id

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            # Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.
            pad_token_id = eos_token_id

        encoder_output = self._encoder_forward(input_ids, attention_mask)

        input_ids = self._prepare_decoder_input_ids_for_generation(
            input_ids,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,
        )

        return self.greedy_search(
            input_ids,
            encoder_output,
            attention_mask,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )


# TorchScript compatible BeamSearchScorer
class BeamSearchScorerTS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_length: int = 200
        self.num_beams: int = 3
        self.batch_size: int = 1
        self.length_penalty: float = 1.0
        self.do_early_stopping: bool = True
        self.num_beam_hyps_to_keep: int = 1
        self.num_beam_groups: int = 1
        self.group_size: int = self.num_beams // self.num_beam_groups
        self._done = torch.zeros(self.batch_size, dtype=torch.bool)
        self._beam_hyps_count = torch.zeros(self.batch_size, dtype=torch.long)
        self._beam_hyps_worst_scores = torch.zeros(self.batch_size) + 1e9
        self._beam_hyps_max_length: int = self.max_length - 1
        self._beam_hyps: List[torch.Tensor] = [torch.zeros(2)]  # placeholder for TorchScript compatibility
        self._beam_scores: List[torch.Tensor] = [torch.zeros(2)]  # placeholder for TorchScript compatibility

    def is_done(self) -> torch.Tensor:
        return self._done.all()

    def init(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        device: torch.device,
        length_penalty: float = 1.0,
        do_early_stopping: bool = False,
        num_beam_hyps_to_keep: int = 1,
        num_beam_groups: int = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        # NOTE: TorchScript does not support List of Modules
        #       Rewritten BeamHypotheses with tensors and list of tensors.
        self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._beam_hyps_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._beam_hyps_worst_scores = torch.zeros(batch_size, device=device) + 1e9
        self._beam_hyps = []
        self._beam_scores = []

        self._beam_hyps_max_length = max_length - 1  # ignoring bos_token

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    def hypo_len(self, hypo_idx: int):
        """
        Number of hypotheses in the list.
        """
        return self._beam_hyps_count[hypo_idx]

    def hypo_add(self, hyp: torch.Tensor, sum_logprobs: float, hypo_idx: int):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        hyps_count = self.hypo_len(hypo_idx)
        if hyps_count < self.num_beams or score > self._beam_hyps_worst_scores[hypo_idx]:
            # NOTE: work around difference of torch.sum(empty_tensor) == 0, while error in onnx.
            # Bug: https://msdata.visualstudio.com/Vienna/_workitems/edit/1486599
            beam_idx = (
                torch.sum(self._beam_hyps_count[:hypo_idx]) if hypo_idx != 0 else torch.tensor(0, dtype=torch.long)
            )
            self._beam_scores.insert(beam_idx, torch.tensor([score]))
            self._beam_hyps.insert(beam_idx, hyp)
            if hyps_count + 1 > self.num_beams:
                sorted_next_scores, sorted_indices = torch.topk(
                    torch.cat(self._beam_scores)[beam_idx : beam_idx + hyps_count + 1], hyps_count + 1, largest=False
                )
                del self._beam_hyps[int((sorted_indices[0] + beam_idx))]
                del self._beam_scores[int((sorted_indices[0] + beam_idx))]
                self._beam_hyps_worst_scores[hypo_idx] = sorted_next_scores[1]
            else:
                self._beam_hyps_worst_scores[hypo_idx] = min(score, self._beam_hyps_worst_scores[hypo_idx])
                self._beam_hyps_count[hypo_idx] = hyps_count + 1

    def hypo_is_done(self, hypo_idx: int, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        if self.hypo_len(hypo_idx) < self.num_beams:
            return False
        elif self.do_early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self._beam_hyps_worst_scores[hypo_idx].item() >= cur_score
            return ret

    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps_count)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx in range(batch_size):
            if self._done[batch_idx]:
                assert (
                    self.hypo_len(batch_idx) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    self.hypo_add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        batch_idx,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or self.hypo_is_done(
                batch_idx,
                next_scores[batch_idx].max().item(),
                cur_len,
            )

        return next_beam_scores.view(-1), next_beam_tokens.view(-1), next_beam_indices.view(-1)

    def finalize(
        self,
        input_ids: torch.Tensor,
        final_beam_scores: torch.Tensor,
        final_beam_tokens: torch.Tensor,
        final_beam_indices: torch.Tensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(self._beam_hyps_count)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                self.hypo_add(final_tokens, final_score, batch_idx)

        # select the best hypotheses
        # NOTE: torch.Tensor.new_zeros() is not scriptable
        sent_lengths = torch.zeros(batch_size * self.num_beam_hyps_to_keep, dtype=torch.long)
        best = []
        best_scores = torch.zeros(
            batch_size * self.num_beam_hyps_to_keep, device=input_ids.device, dtype=torch.float32
        )
        # retrieve best hypotheses
        for i in range(batch_size):
            # NOTE: lambda is not scriptable
            batch_hypo_start = torch.sum(self._beam_hyps_count[:i]) if i > 0 else torch.tensor(0, dtype=torch.long)
            batch_hypo_end = torch.sum(self._beam_hyps_count[: i + 1])
            beam_scores = torch.cat(self._beam_scores)[batch_hypo_start:batch_hypo_end]
            sorted_next_scores, sorted_indices = torch.topk(beam_scores, len(beam_scores), largest=True)
            for j in range(self.num_beam_hyps_to_keep):
                best_score = beam_scores[sorted_indices[j]]
                best_hyp = self._beam_hyps[batch_hypo_start + sorted_indices[j]]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max() + 1, self.max_length)
        decoded = torch.zeros(batch_size * self.num_beam_hyps_to_keep, sent_max_len, dtype=torch.long)
        # shorter batches are padded if needed
        if sent_lengths.min() != sent_lengths.max():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id

        return decoded, best_scores


class BARTBeamSearchGenerator(BARTGenerator):
    def __init__(self, model):
        super().__init__(model)
        self.beam_scorer = BeamSearchScorerTS()
        self.device = model.device

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_state: torch.Tensor,
        expand_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        attention_mask = attention_mask.index_select(0, expanded_return_idx)

        last_hidden_state = last_hidden_state.index_select(0, expanded_return_idx.to(last_hidden_state.device))
        return input_ids, attention_mask, last_hidden_state

    def adjust_logits_during_generation(self, logits, cur_len: int, max_length: int):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            logits = self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            logits = self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id: int):
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        mask = torch.full_like(scores, 1, dtype=torch.bool)
        mask[:, token_id] = False
        return scores.masked_fill(mask, -float("inf"))

    def _reorder_cache(self, past: List[torch.Tensor], beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        reordered_decoder_past = []
        for state in past:
            reordered_decoder_past.append(state.index_select(0, beam_idx))
        return reordered_decoder_past

    def beam_search(
        self, input_ids, encoder_output, attention_mask, num_beams, max_length, pad_token_id: int, eos_token_id: int
    ):
        batch_size = self.beam_scorer.batch_size

        num_beams = self.beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        next_tokens = torch.zeros((batch_size, num_beams), dtype=torch.long, device=input_ids.device)
        next_indices = torch.zeros((batch_size, num_beams), dtype=torch.long, device=input_ids.device)

        past: List[torch.Tensor] = []
        while cur_len < max_length:
            logits, past = self._decoder_forward(input_ids, encoder_output, attention_mask, past)
            next_token_logits = logits[:, -1, :]

            # adjust tokens for Bart, *e.g.*
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            # pre-process distribution
            next_token_scores = self.logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            beam_scores, beam_next_tokens, beam_idx = self.beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            cur_len = cur_len + 1

            if len(past) > 0:
                past = self._reorder_cache(past, beam_idx)

            if self.beam_scorer.is_done():
                break

        sequences, sequence_scores = self.beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        return sequences

    def forward(self, input_ids, attention_mask, num_beams, max_length, decoder_start_token_id):
        pad_token_id = self.config.pad_token_id
        bos_token_id = self.config.bos_token_id
        eos_token_id = self.config.eos_token_id

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            # logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        encoder_output = self._encoder_forward(input_ids, attention_mask)

        input_ids = self._prepare_decoder_input_ids_for_generation(
            input_ids,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,
        )

        batch_size = input_ids.shape[0]

        length_penalty = self.config.length_penalty
        num_return_sequences = self.config.num_return_sequences
        early_stopping = True

        self.beam_scorer.init(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )

        input_ids, attention_mask, encoder_output = self._expand_inputs_for_generation(
            input_ids,
            attention_mask,
            encoder_output,
            expand_size=num_beams,
        )

        return self.beam_search(
            input_ids=input_ids,
            encoder_output=encoder_output,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
