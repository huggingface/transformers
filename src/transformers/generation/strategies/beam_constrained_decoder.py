import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from typing import List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

from .utils import BeamSearchDecoderOnlyOutput, BeamSearchEncoderDecoderOutput, BeamSampleOutput
from .beam_utils import BeamHypotheses, BeamScorer
from .utils import GenerationStrategy

from ..configuration_utils import GenerationConfig
from ..logits_process import LogitsProcessorList
from ..stopping_criteria import StoppingCriteriaList, validate_stopping_criteria

if TYPE_CHECKING:
    from ...modeling_utils import PreTrainedModel, GenerationMixin
    from ..streamers import BaseStreamer


class ConstrainedDecoder(GenerationStrategy):
    """
    Implements constrained decoding method.
    """

    def __init__(self, config: GenerationConfig, check_config: bool = True):
        super().__init__(config, check_config=check_config)
        self.scorer = None

    @classmethod
    def param_check(cls, config: GenerationConfig):
        if config.constraints is None and config.force_words_ids is None:
            raise ValueError(
                f"{cls.__class__.__name__} requires a constraint or a force_words_ids to be set."
            )

    def __call__(self,
                 model: Union["PreTrainedModel", "GenerationMixin"],
                 input_ids: torch.LongTensor,
                 assistant_model: Optional["PreTrainedModel"] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 typical_p: Optional[float] = None,
                 penalty_alpha: Optional[float] = None,
                 do_sample: bool = False,
                 logits_processor: Optional[LogitsProcessorList] = None,
                 logits_warper: Optional[LogitsProcessorList] = None,
                 max_length: Optional[int] = None,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[Union[int, List[int]]] = None,
                 output_attentions: Optional[bool] = None,
                 output_hidden_states: Optional[bool] = None,
                 output_scores: Optional[bool] = None,
                 return_dict_in_generate: Optional[bool] = None,
                 synced_gpus: bool = False,
                 streamer: Optional["BaseStreamer"] = None,
                 sequential: Optional[bool] = None,
                 **model_kwargs,
                 ) -> Union[BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **constrained beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.constrained_beam_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation, while satisfying a list of positive constraints. For more information, the
                documentation of [`ConstrainedBeamSearchScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers.generation import ConstrainedBeamSearchScorer        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...      PhrasalConstraint,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> constraint_str = "Sie"
        >>> constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # slice to remove eos token
        >>> constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]


        >>> # instantiate beam scorer
        >>> beam_scorer = ConstrainedBeamSearchScorer(
        ...     batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.constrained_beam_search(
        ...     input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt sind Sie?']
        ```"""
        if self.check_config:
            if self.config.num_return_sequences > self.config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if self.config.num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")

            if self.config.do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if self.config.num_beam_groups is not None and self.config.num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if self.config.constraints is not None:
                final_constraints = self.config.constraints

            if self.config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {self.config.force_words_ids}."
                    )

                if (
                        not isinstance(self.config.force_words_ids, list)
                        or len(self.config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in self.config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                                any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                                for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            batch_size = input_ids.shape[0]
            if self.scorer is None:
                self.scorer = ConstrainedBeamSearchScorer(
                    constraints=final_constraints,
                    batch_size=batch_size,
                    num_beams=self.config.num_beams,
                    device=input_ids.device,
                    length_penalty=self.config.length_penalty,
                    do_early_stopping=self.config.early_stopping,
                    num_beam_hyps_to_keep=self.config.num_return_sequences,
                    max_length=self.config.max_length,
                )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=self.config.num_beams,
                is_encoder_decoder=model.config.is_encoder_decoder,
                **model_kwargs,
            )
        # 13. run generation
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(self.scorer._beam_hyps)
        num_beams = self.scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            scores_for_all_vocab = next_token_scores.clone()

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = self.scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                scores_for_all_vocab,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if self.scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = self.scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if model.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


class Constraint(ABC):
    r"""Abstract base class for all constraints that can be applied during generation.
    It must define how the constraint can be satisfied.

    All classes that inherit Constraint must follow the requirement that

    ```py
    completed = False
    while not completed:
        _, completed = constraint.update(constraint.advance())
    ```

    will always terminate (halt).
    """

    def __init__(self):
        # test for the above condition
        self.test()

    def test(self):
        """
        Tests whether this constraint has been properly defined.
        """
        counter = 0
        completed = False
        while not completed:
            if counter == 1:
                self.reset()
            advance = self.advance()
            if not self.does_advance(advance):
                raise Exception(
                    "Custom Constraint is not defined correctly. self.does_advance(self.advance()) must be true."
                )

            stepped, completed, reset = self.update(advance)
            counter += 1

            if counter > 10000:
                raise Exception("update() does not fulfill the constraint.")

        if self.remaining() != 0:
            raise Exception("Custom Constraint is not defined correctly.")

    @abstractmethod
    def advance(self):
        """
        When called, returns the token that would take this constraint one step closer to being fulfilled.

        Return:
            token_ids(`torch.tensor`): Must be a tensor of a list of indexable tokens, not some integer.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def does_advance(self, token_id: int):
        """
        Reads in a token and returns whether it creates progress.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def update(self, token_id: int):
        """
        Reads in a token and returns booleans that indicate the progress made by it. This function will update the
        state of this object unlikes `does_advance(self, token_id: int)`.

        This isn't to test whether a certain token will advance the progress; it's to update its state as if it has
        been generated. This becomes important if token_id != desired token (refer to else statement in
        PhrasalConstraint)

        Args:
            token_id(`int`):
                The id of a newly generated token in the beam search.
        Return:
            stepped(`bool`):
                Whether this constraint has become one step closer to being fulfuilled.
            completed(`bool`):
                Whether this constraint has been completely fulfilled by this token being generated.
            reset (`bool`):
                Whether this constraint has reset its progress by this token being generated.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def reset(self):
        """
        Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
        a constraint is abrupted by an unwanted token.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def remaining(self):
        """
        Returns the number of remaining steps of `advance()` in order to complete this constraint.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def copy(self, stateful=False):
        """
        Creates a new instance of this constraint.

        Args:
            stateful(`bool`): Whether to not only copy the constraint for new instance, but also its state.

        Return:
            constraint(`Constraint`): The same constraint as the one being called from.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class PhrasalConstraint(Constraint):
    r"""
    [`Constraint`] enforcing that an ordered sequence of tokens is included in the output.

    Args:
        token_ids (`List[int]`):
            The id of the token that must be generated by the output.
    """

    def __init__(self, token_ids: List[int]):
        super(Constraint, self).__init__()

        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise ValueError(f"`token_ids` has to be a non-empty list, but is {token_ids}.")
        if any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids):
            raise ValueError(f"Each list in `token_ids` has to be a list of positive integers, but is {token_ids}.")

        self.token_ids = token_ids

        self.seqlen = len(self.token_ids)
        self.fulfilled_idx = -1  # the index of the currently fulfilled step
        self.completed = False

    def advance(self):
        if self.completed:
            return None
        return self.token_ids[self.fulfilled_idx + 1]

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        if self.completed:
            return False

        return token_id == self.token_ids[self.fulfilled_idx + 1]

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.fulfilled_idx += 1
            stepped = True
            if self.fulfilled_idx == (self.seqlen - 1):
                completed = True
            self.completed = completed
        else:
            # failed to make progress.
            reset = True
            self.reset()
        return stepped, completed, reset

    def reset(self):
        self.completed = False
        self.fulfilled_idx = 0

    def remaining(self):
        return self.seqlen - (self.fulfilled_idx + 1)

    def copy(self, stateful=False):
        new_constraint = PhrasalConstraint(self.token_ids)

        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.fulfilled_idx = self.fulfilled_idx
            new_constraint.completed = self.completed

        return new_constraint


class DisjunctiveTrie:
    def __init__(self, nested_token_ids: List[List[int]], no_subsets=True):
        r"""
        A helper class that builds a trie with the words represented in `nested_token_ids`.
        """
        self.max_height = max([len(one) for one in nested_token_ids])

        root = {}
        for token_ids in nested_token_ids:
            level = root
            for tidx, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {}

                level = level[token_id]

        if no_subsets and self.has_subsets(root, nested_token_ids):
            raise ValueError(
                "Each list in `nested_token_ids` can't be a complete subset of another list, but is"
                f" {nested_token_ids}."
            )

        self.trie = root

    def next_tokens(self, current_seq):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        start = self.trie

        for current_token in current_seq:
            start = start[current_token]

        next_tokens = list(start.keys())

        return next_tokens

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)

        return len(next_tokens) == 0

    def count_leaves(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        else:
            return sum([self.count_leaves(nn) for nn in next_nodes])

    def has_subsets(self, trie, nested_token_ids):
        """
        Returns whether # of leaves == # of words. Otherwise some word is a subset of another.
        """
        leaf_count = self.count_leaves(trie)
        return len(nested_token_ids) != leaf_count


class DisjunctiveConstraint(Constraint):
    r"""
    A special [`Constraint`] that is fulfilled by fulfilling just one of several constraints.

    Args:
        nested_token_ids (`List[List[int]]`): a list of words, where each word is a list of ids. This constraint
        is fulfilled by generating just one from the list of words.
    """

    def __init__(self, nested_token_ids: List[List[int]]):
        super(Constraint, self).__init__()

        if not isinstance(nested_token_ids, list) or len(nested_token_ids) == 0:
            raise ValueError(f"`nested_token_ids` has to be a non-empty list, but is {nested_token_ids}.")
        if any(not isinstance(token_ids, list) for token_ids in nested_token_ids):
            raise ValueError(f"`nested_token_ids` has to be a list of lists, but is {nested_token_ids}.")
        if any(
            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
            for token_ids in nested_token_ids
        ):
            raise ValueError(
                f"Each list in `nested_token_ids` has to be a list of positive integers, but is {nested_token_ids}."
            )

        self.trie = DisjunctiveTrie(nested_token_ids)
        self.token_ids = nested_token_ids

        self.seqlen = self.trie.max_height
        self.current_seq = []
        self.completed = False

    def advance(self):
        token_list = self.trie.next_tokens(self.current_seq)

        if len(token_list) == 0:
            return None
        else:
            return token_list

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        next_tokens = self.trie.next_tokens(self.current_seq)

        return token_id in next_tokens

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.current_seq.append(token_id)
            stepped = True
        else:
            reset = True
            self.reset()

        completed = self.trie.reached_leaf(self.current_seq)
        self.completed = completed

        return stepped, completed, reset

    def reset(self):
        self.completed = False
        self.current_seq = []

    def remaining(self):
        if self.completed:
            # since this can be completed without reaching max height
            return 0
        else:
            return self.seqlen - len(self.current_seq)

    def copy(self, stateful=False):
        new_constraint = DisjunctiveConstraint(self.token_ids)

        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.current_seq = self.current_seq
            new_constraint.completed = self.completed

        return new_constraint


class ConstraintListState:
    r"""
    A class for beam scorers to track its progress through a list of constraints.

    Args:
        constraints (`List[Constraint]`):
            A list of [`Constraint`] objects that must be fulfilled by the beam scorer.
    """

    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints

        # max # of steps required to fulfill a given constraint
        self.max_seqlen = max([c.seqlen for c in constraints])
        self.n_constraints = len(constraints)
        self.completed = False

        self.init_state()

    def init_state(self):
        self.complete_constraints = []
        self.inprogress_constraint = None
        self.pending_constraints = [constraint.copy(stateful=False) for constraint in self.constraints]

    def get_bank(self):
        add = 0
        if self.inprogress_constraint:
            # extra points for having a constraint mid-fulfilled
            add += self.max_seqlen - self.inprogress_constraint.remaining()

        return (len(self.complete_constraints) * self.max_seqlen) + add

    def advance(self):
        """The list of tokens to generate such that we can make progress.
        By "list" we don't mean the list of token that will fully fulfill a constraint.

        Given constraints `c_i = {t_ij | j == # of tokens}`, If we're not in the middle of progressing through a
        specific constraint `c_i`, we return:

        `[t_k1 for k in indices of unfulfilled constraints]`

        If we are in the middle of a constraint, then we return:
            `[t_ij]`, where `i` is the index of the inprogress constraint, `j` is the next step for the constraint.

        Though we don't care which constraint is fulfilled first, if we are in the progress of fulfilling a constraint,
        that's the only one we'll return.
        """
        token_list = []
        if self.inprogress_constraint is None:
            for constraint in self.pending_constraints:  # "pending" == "unfulfilled yet"
                advance = constraint.advance()
                if isinstance(advance, int):
                    token_list.append(advance)
                elif isinstance(advance, list):
                    token_list.extend(advance)
        else:
            advance = self.inprogress_constraint.advance()
            if isinstance(advance, int):
                token_list.append(advance)
            elif isinstance(advance, list):
                token_list.extend(advance)

        if len(token_list) == 0:
            return None
        else:
            return token_list

    def reset(self, token_ids: Optional[List[int]]):
        """
        token_ids: the tokens generated thus far to reset the state of the progress through constraints.
        """
        self.init_state()

        if token_ids is not None:
            for token in token_ids:
                # completes or steps **one** constraint
                complete, stepped = self.add(token)

                # the entire list of constraints are fulfilled
                if self.completed:
                    break

    def add(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` should be an `int`, but is `{token_id}`.")

        complete, stepped = False, False

        if self.completed:
            complete = True
            stepped = False
            return complete, stepped

        if self.inprogress_constraint is not None:
            # In the middle of fulfilling a constraint. If the `token_id` *does* makes an incremental progress to current
            # job, simply update the state

            stepped, complete, reset = self.inprogress_constraint.update(token_id)
            if reset:
                # 1. If the next token breaks the progress, then we must restart.
                #     e.g. constraint = "I love pies" and sequence so far is "I love" but `token_id` == "books".

                #     But that doesn't mean we self.init_state(), since we only reset the state for this particular
                #     constraint, not the full list of constraints.

                self.pending_constraints.append(self.inprogress_constraint.copy(stateful=False))
                self.inprogress_constraint = None

            if complete:
                # 2. If the next token completes the constraint, move it to completed list, set
                #     inprogress to None. If there are no pending constraints either, then this full list of constraints
                #     is complete.

                self.complete_constraints.append(self.inprogress_constraint)
                self.inprogress_constraint = None

                if len(self.pending_constraints) == 0:
                    # we're done!
                    self.completed = True

        else:
            # Not in the middle of fulfilling a constraint. So does this `token_id` helps us step towards any of our list
            # of constraints?

            for cidx, pending_constraint in enumerate(self.pending_constraints):
                if pending_constraint.does_advance(token_id):
                    stepped, complete, reset = pending_constraint.update(token_id)

                    if not stepped:
                        raise Exception(
                            "`constraint.update(token_id)` is not yielding incremental progress, "
                            "even though `constraint.does_advance(token_id)` is true."
                        )

                    if complete:
                        self.complete_constraints.append(pending_constraint)
                        self.inprogress_constraint = None

                    if not complete and stepped:
                        self.inprogress_constraint = pending_constraint

                    if complete or stepped:
                        # If we made any progress at all, then it's at least not a "pending constraint".

                        self.pending_constraints = (
                            self.pending_constraints[:cidx] + self.pending_constraints[cidx + 1 :]
                        )

                        if len(self.pending_constraints) == 0 and self.inprogress_constraint is None:
                            # If there's no longer any pending after this and no inprogress either, then we must be
                            # complete.

                            self.completed = True

                        break  # prevent accidentally stepping through multiple constraints with just one token.

        return complete, stepped

    def copy(self, stateful=True):
        new_state = ConstraintListState(self.constraints)  # we actually never though self.constraints objects
        # throughout this process. So it's at initialization state.

        if stateful:
            new_state.complete_constraints = [
                constraint.copy(stateful=True) for constraint in self.complete_constraints
            ]
            if self.inprogress_constraint is not None:
                new_state.inprogress_constraint = self.inprogress_constraint.copy(stateful=True)
            new_state.pending_constraints = [constraint.copy() for constraint in self.pending_constraints]

        return new_state


class ConstrainedBeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing constrained beam search decoding.


    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        constraints (`List[Constraint]`):
            A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
            output. For more information, the documentation of [`Constraint`] should be read.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        constraints: List[Constraint],
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self.constraints = constraints

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

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

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def make_constraint_states(self, n):
        return [ConstraintListState([constraint.copy() for constraint in self.constraints]) for _ in range(n)]

    def check_completes_constraints(self, sequence):
        new_state = self.make_constraint_states(1)[0]
        new_state.reset(sequence)
        return new_state.completed

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        scores_for_all_vocab: torch.FloatTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
                Current scores of the top `2 * num_beams` non-finished beam hypotheses.
            next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
            next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
            scores_for_all_vocab (`torch.FloatTensor` of shape `(batch_size * num_beams, sequence_length)`):
                The scores of all tokens in the vocabulary for each of the beam hypotheses.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

        Return:
            `UserDict`: A dictionary composed of the fields as defined above:

                - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of
                  all
                non-finished beams.

                - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be
                  added
                to the non-finished beam_hypotheses.
                - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices
                indicating to which beam the next tokens shall be added.
        """

        cur_len = input_ids.shape[-1] + 1  # add up to the length which the next_scores is calculated on
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device

        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence.
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue

                    completes_constraint = self.check_completes_constraints(input_ids[batch_beam_idx].cpu().tolist())
                    if completes_constraint:
                        beam_hyp.add(
                            input_ids[batch_beam_idx].clone(),
                            next_score.item(),
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

            new_scores, new_tokens, new_indices = self.step_sentence_constraint(
                batch_idx,
                input_ids,
                scores_for_all_vocab,
                next_beam_scores[batch_idx],
                next_beam_tokens[batch_idx],
                next_beam_indices[batch_idx],
            )

            next_beam_scores[batch_idx] = new_scores
            next_beam_tokens[batch_idx] = new_tokens
            next_beam_indices[batch_idx] = new_indices

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def step_sentence_constraint(
        self,
        batch_idx: int,
        input_ids: torch.LongTensor,
        vocab_scores: torch.FloatTensor,
        sent_beam_scores: torch.FloatTensor,
        sent_beam_tokens: torch.LongTensor,
        sent_beam_indices: torch.LongTensor,
        push_progress: bool = False,
    ):
        # sent_beam_tokens are the next {num_beams} number of tokens that are under consideration for this beam
        # (candidate next tokens)

        # 1. Adding "advance_tokens"
        #     using ConstraintStateList.advance(), we propose new tokens to be added into this "candidate list" that will
        #     advance us in fulfilling the constraints.

        # 2. Selecting best candidates such that we end up with highest probable candidates
        #     that fulfill our constraints.

        orig_len = sent_beam_indices.size(0)
        device = sent_beam_indices.device

        # initialize states
        topk_contraint_states = self.make_constraint_states(orig_len)
        advance_constraint_states = self.make_constraint_states(orig_len)

        sidx, eidx = batch_idx * orig_len, (batch_idx + 1) * orig_len
        this_batch_input_ids = input_ids[sidx:eidx]
        this_batch_token_scores = vocab_scores[sidx:eidx]
        full_hypotheses = torch.cat((input_ids[sent_beam_indices], sent_beam_tokens.unsqueeze(-1)), dim=-1)

        # need to make new hypothesis that advance the constraints
        track_new = {
            "new_seqs": full_hypotheses.tolist(),
            "new_states": [],
            "new_indices": [],
            "new_tokens": [],
            "new_scores": [],
        }
        for seq_idx, pre_seq in enumerate(this_batch_input_ids):
            # pre_seq = ith sequence generated before this step.

            # input_ids -> (topk) generic beam search best model next tokens
            #           -> (advance) constraints forcing the next token
            # either way, we need to sort them into "banks" later, so store a "ConstraintListState" for all types of
            # hypotheses.

            topk_state = topk_contraint_states[seq_idx]
            topk_state.reset(full_hypotheses[seq_idx].cpu().tolist())

            advance_state = advance_constraint_states[seq_idx]
            advance_state.reset(pre_seq.cpu().tolist())

            if not advance_state.completed:
                advance_tokens = torch.LongTensor(advance_state.advance()).to(device)
                for advance_token in advance_tokens:
                    # since adding each `advance_token` leads to a different hypothesis, create new state instance.
                    new_state = advance_state.copy(stateful=True)
                    new_state.add(advance_token.cpu().tolist())

                    advance_seq = torch.cat((pre_seq, advance_token.unsqueeze(0)), -1).cpu().tolist()
                    if advance_seq not in track_new["new_seqs"]:
                        # prevent duplicates, which are basically bound to happen in this process.
                        track_new["new_seqs"].append(advance_seq)
                        track_new["new_indices"].append(sidx + seq_idx)  # idx -> global idx across all the batches
                        track_new["new_tokens"].append(advance_token)
                        track_new["new_scores"].append(this_batch_token_scores[seq_idx].take(advance_token))
                        track_new["new_states"].append(new_state)
            elif push_progress:
                # Basically, `sent_beam_indices` often chooses very little among `input_ids` the generated sequences that
                # actually fulfill our constraints. For example, let constraints == ["loves pies"] and

                #     pre_seq_1 = "The child loves pies and" pre_seq_2 = "The child plays in the playground and"

                # Without this step, if `sent_beam_indices` is something like [1,1], then
                #     1. `pre_seq_1` won't be added to the list of (topk) hypothesis since it's not in the indices and
                #     2.  it won't be added to the list of (advance) hypothesis since it's completed already. (this is
                #         the else part of `if constraints_completed[seq_idx]`)
                #     3. it ends up simply getting removed from consideration.

                # #3 might be fine and actually desired, since it's likely that it's a low-probability output anyways,
                # especially if it's not in the list of `sent_beam_indices`. But this often leads to lengthened beam
                # search times, since completed sequences keep getting removed after all this effort for constrained
                # generation.

                # Here, we basically take `pre_seq_1` and to "push" it into the considered list of hypotheses, by simply
                # appending the next likely token in the vocabulary and adding it to the list of hypotheses.

                new_score, new_token = torch.max(this_batch_token_scores[seq_idx], 0)  # some next probable token
                advance_seq = torch.cat((pre_seq, new_token.unsqueeze(0)), -1)

                advance_state = advance_constraint_states[seq_idx]

                advance_seq = advance_seq.cpu().tolist()

                advance_state.reset(advance_seq)
                if advance_seq not in track_new["new_seqs"]:
                    # but still don't want to have duplicates
                    track_new["new_seqs"].append(advance_seq)
                    track_new["new_indices"].append(seq_idx)
                    track_new["new_tokens"].append(new_token)
                    track_new["new_scores"].append(new_score)
                    track_new["new_states"].append(advance_state)

        if len(track_new["new_indices"]) > 0:
            new_indices = torch.tensor(track_new["new_indices"]).to(device)
            new_tokens = torch.stack(track_new["new_tokens"]).to(device)
            new_scores = torch.stack(track_new["new_scores"]).to(device)

            all_states = topk_contraint_states + track_new["new_states"]
            all_tokens = torch.cat((sent_beam_tokens, new_tokens), -1)
            all_scores = torch.cat((sent_beam_scores, new_scores), -1)
            all_banks = torch.tensor([one.get_bank() for one in all_states]).to(device)

            zipped = all_banks * 100 + all_scores
            indices = zipped.sort(descending=True).indices
            sorted_banks = all_banks[indices]

            # Then we end up with {sorted among bank C}, {sorted among bank C-1}, ..., {sorted among bank 0}

            counter = -1
            cur_bank = sorted_banks[0]
            increments = []
            for bank in sorted_banks:
                if bank == cur_bank:
                    counter += 1
                else:
                    counter = 0
                    cur_bank = bank
                increments.append(counter)
            rearrangers = torch.tensor(np.argsort(increments, kind="mergesort"))

            indices = indices[rearrangers][:orig_len]

            sent_beam_scores = all_scores[indices]
            sent_beam_tokens = all_tokens[indices]
            sent_beam_indices = torch.cat((sent_beam_indices, new_indices))[indices]

        return sent_beam_scores, sent_beam_tokens, sent_beam_indices

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams

            ids_collect = []
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]

                completes_constraint = self.check_completes_constraints(final_tokens.cpu().tolist())
                if completes_constraint:
                    beam_hyp.add(final_tokens, final_score)
                    ids_collect.append(beam_id)

            # due to overly complex constraints or other factors, sometimes we can't gaurantee a successful
            # generation. In these cases we simply return the highest scoring outputs.
            if len(ids_collect) < self.num_beam_hyps_to_keep:
                for beam_id in range(self.num_beams):
                    if beam_id not in ids_collect:
                        batch_beam_idx = batch_idx * self.num_beams + beam_id
                        final_score = final_beam_scores[batch_beam_idx].item()
                        final_tokens = input_ids[batch_beam_idx]
                        beam_hyp.add(final_tokens, final_score)
                    if len(ids_collect) >= self.num_beam_hyps_to_keep:
                        break

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1

        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )
