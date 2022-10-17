"""Token Classification (NER) pipeline that performs truncation on inputs and
reconstitutes across overflowing segments.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union, overload

import torch
from transformers.file_utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
)
from transformers.models.bert.tokenization_bert import BasicTokenizer
from transformers.pipelines.base import (
    PIPELINE_INIT_ARGS,
    ChunkPipeline,
    ModelOutput,
    PipelineChunkIterator,
)
from transformers.pipelines.token_classification import (
    AggregationStrategy,
    TokenClassificationArgumentHandler,
    TokenClassificationPipeline,
)
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER, BatchEncoding
from transformers.utils import logging

if is_tf_available():
    from transformers.models.auto.modeling_tf_auto import (
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    )

if is_torch_available():
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    )

logger = logging.get_logger(__name__)


class PipelineChunkWrapIterator(PipelineChunkIterator):
    """An iterator that ensures `infer` is called on the entire batch.
    This does not iterate over the individual items as other PipelineIterators
    do, but wraps the input in a separate list to treat the whole pipeline input
    as a single preprocessor input"""

    def __init__(self, loader, infer, params, loader_batch_size=None):
        if isinstance(loader, str):
            loader = [[loader]]
        elif isinstance(loader, list) and all(isinstance(i, str) for i in loader):
            loader = [loader]
        super().__init__(loader, infer, params)

    def __len__(self):
        return len(self.loader[0])


class OverflowTokenClassificationArgumentHandler(TokenClassificationArgumentHandler):
    """
    Handles pipeline call arguments for token classification as well as
    additional arguments for batch truncation
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):  # type: ignore
        # Potential bug in upstream, the wrapped inputs aren't used anywhere
        inputs, _ = super().__call__(inputs, **kwargs)
        return inputs, None


class ReconstitutionStrategy(ExplicitEnum):
    """All the valid reconstitution strategies for
    OverflowTokenClassificationPipeline"""

    FIRST = "first"
    MAX_SCORE = "max_score"
    ENTITIES = "entities"
    FIRST_ENTITIES = "first_entities"


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        reconstitution_strategy (`str`, *optional*, defaults to `"entities"`):
            The strategy to select the tag for tokens with multiple predictions
            included in the stride portion of overflowing text.

                - "first" : Will simply take the tag from the first segment
                - "entities" : Will pick any entity tag over a non-entity
                  tag, for example, "B-PER" over "O". "I-" tags which follow a
                  preceding tag of the same type will always take precedence over
                  "B-" tags. If there are multiple entity tags of any other pattern,
                  the tag with the greater score will be used. If scores are equal,
                  the tag from the first segment is used
                - "first_entities": Will use the `ENTITIES` strategy
                  except that in the case of multiple entity tags, the tag from
                  the first segment is used and the scores are not considered
                - "max_score" : Will select whichever tag has the highest prediction
                  score, regardless of what that tag was
    """,
)
class OverflowTokenClassificationPipeline(ChunkPipeline, TokenClassificationPipeline):
    """Extension to the transformers token classification pipeline that
    applies truncation with overflow and padding to batch inputs"""

    entity_type: Type = dict
    reconstitute_ignore_tags = ["O"]

    def __init__(
        self,
        args_parser=None,
        max_seq_len: int = 0,
        stride: int = 10,
        *args,
        **kwargs,
    ):
        # Would be nice to call TokenClassificationPipeline superclass here
        #  instead of duplicating
        super().__init__(*args, **kwargs)
        self.check_model_type(
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
        )

        # FIXME I think this is a bug in the upstream parent class.
        #  _basic_tokenizer does nothing and accessing self.tokenizer.<attr>
        #  if this is None raises an error
        if self.tokenizer is None:
            # BasicTokenizer is missing some attributes that are needed, including
            #  the __call__ method and a model_max_len value
            # self.tokenizer = BasicTokenizer(do_lower_case=False)
            raise ValueError("An explicit tokenizer is required to run this pipeline")
        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        if args_parser is None:
            args_parser = TokenClassificationArgumentHandler()
        self._args_parser = args_parser
        # Determine the correct max length to use from the different max lengths
        model_max_len = self.tokenizer.model_max_length
        if model_max_len == VERY_LARGE_INTEGER:
            model_max_len = 0
        if max_seq_len:
            if model_max_len:
                self.max_length = min(max_seq_len, self.tokenizer.model_max_length)
            else:
                self.max_length = max_seq_len
        else:
            if model_max_len:
                self.max_length = model_max_len
            else:
                self.max_length = 0
        self.truncation = self.max_length > 0
        # Prevent error when calling tokenizer method
        if self.truncation and stride >= self.max_length:
            raise ValueError(
                f"Stride '{stride}' was not less than detected max length "
                f"'{self.max_length}'"
            )
        self.stride = stride

    def _sanitize_parameters(  # type: ignore
        self,
        ignore_labels=None,
        grouped_entities: Optional[bool] = None,
        ignore_subwords: Optional[bool] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        reconstitution_strategy: Optional[ReconstitutionStrategy] = None,
        offset_mapping: Optional[List[Tuple[int, int]]] = None,
    ):
        # TODO Reinclude entity aggregation and ignored labels
        if any(
            arg is not None
            for arg in [aggregation_strategy, grouped_entities, ignore_labels]
        ):
            message = (
                "OverflowTokenClassificationPipeline does not perform entity "
                "aggregation nor ignore labels."
                "The given aggregation strategy will not be used."
            )
            warnings.warn(message)
        preprocess_parameters, _, postprocess_params = super()._sanitize_parameters(
            ignore_subwords=ignore_subwords,
            offset_mapping=offset_mapping,
        )
        if reconstitution_strategy is not None:
            if isinstance(reconstitution_strategy, str):
                try:
                    reconstitution_strategy = ReconstitutionStrategy[  # type: ignore
                        reconstitution_strategy.upper()
                    ]
                except KeyError as err:
                    raise ValueError("Invalid reconstitution_strategy") from err
            postprocess_params["reconstitution_strategy"] = reconstitution_strategy
        return preprocess_parameters, {}, postprocess_params

    def preprocess(  # type: ignore
        self, sentences: Union[str, List[str]], offset_mapping=None
    ) -> BatchEncoding:
        max_length = self.max_length if self.truncation else None
        model_inputs = self.tokenizer(  # type: ignore
            sentences,
            max_length=max_length,
            truncation=self.truncation,
            stride=self.stride,
            padding=True,
            return_overflowing_tokens=self.truncation,
            return_tensors=self.framework,
            return_special_tokens_mask=True,
            return_attention_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,  # type: ignore
        )
        # TODO I'm pretty sure the offset mapping has a potential for failure as
        #  the shape might not be known until after preprocessing
        if offset_mapping:
            model_inputs["offset_mapping"] = offset_mapping

        def _reshape(
            tensor: Optional[torch.Tensor], index: int
        ) -> Optional[torch.Tensor]:
            """Transform a tensor with i elements to a 1-element tensor containing
            only the element located at index i.
            This function handles cases where the tensor was not present, which
            occurs if e.g. no truncation was applied"""
            if tensor is None:
                return None
            return tensor[index].unsqueeze(0)

        # We need sentences to be a list for zipping with outputs
        if isinstance(sentences, str):
            sentences = [sentences]

        # No overflow returned if not truncating, so this is just a range
        if self.truncation:
            overflow_to_sample_mapping = model_inputs["overflow_to_sample_mapping"]
        else:
            overflow_to_sample_mapping = range(len(sentences))

        for i, overflow_tensor in enumerate(overflow_to_sample_mapping):
            overflow_mapping = int(overflow_tensor)
            sentence = sentences[overflow_mapping]
            return_dict = {
                "overflow_to_sample_mapping": overflow_mapping,
                "sentence": sentence,
                "is_last": i == len(overflow_to_sample_mapping) - 1,
            }

            for tensor_key in [
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "special_tokens_mask",
            ]:
                return_dict[tensor_key] = _reshape(model_inputs[tensor_key], i)

            # Offset mapping must be handled differently as it can be overridden
            return_offset_mapping = offset_mapping
            try:
                offset_mappings: Optional[Any] = model_inputs["offset_mapping"]
                if offset_mappings is not None and isinstance(
                    offset_mappings[i], torch.Tensor
                ):
                    return_offset_mapping = _reshape(offset_mappings, i)
            except IndexError:
                # The offset mapping was not the same shape as the resultant output
                # Could raise a ValueError here as the NER arg parser does
                warnings.warn(f"Could not find offset mapping entry at index {i}")
            return_dict["offset_mapping"] = return_offset_mapping
            yield return_dict

    def _forward(self, model_inputs):
        # Forward
        overflow_to_sample_mapping = model_inputs.pop(
            "overflow_to_sample_mapping", None
        )
        is_last = model_inputs.pop("is_last")
        model_outputs = super()._forward(model_inputs)

        return {
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            "is_last": is_last,
            **model_outputs,
        }

    # TODO This signature violates the superclass signature due to model_outputs,
    #  but that should be a change introduced by the ChunkPipeline class
    def postprocess(  # type: ignore
        self,
        model_outputs: List[ModelOutput],
        reconstitution_strategy=ReconstitutionStrategy.ENTITIES,
        **kwargs,
    ) -> List[List[Any]]:
        """Batch postprocessing method. This will call the
        TokenClassificationPipeline postprocess method on each output from the
        model"""
        results: List[List[Dict]] = []
        for output in model_outputs:
            original_index = output["overflow_to_sample_mapping"]
            processed = super().postprocess(output, ignore_labels=[])
            # If this item is not yet in the results list, add it
            # Otherwise, reconstitute with previous item
            try:
                previous_processed = results[original_index]
                reconstituted = self.reconstitute_segments(
                    previous_processed, processed, reconstitution_strategy
                )
                results[original_index] = reconstituted
            except IndexError:
                results.append(processed)
        return results

    def reconstitute_segments(
        self,
        first_segment: List[Dict],
        second_segment: List[Dict],
        reconstitution_strategy: ReconstitutionStrategy,
    ) -> List[Dict]:
        """Join the entities across two segments that were truncated with
        overflowing tokens with a stride of overlapping tokens.
        This uses values within the processed entities to identify the stride for
        the given segments, rather than the instance variables used for configuration.
        The reconstitution strategy set in the postprocessing params is used to
        configure which token is picked across the two segments.

        Returns:
            Single list of tokens for the pair of segments

        Example:
            max_length = 4, stride = 2, strategy = entities
            first_segment = [("my", O), ("name", O), ("is", O), ("john", O)]
            second_segment = [("is", O), ("john", B-PER), ("doe", I-PER)]
            comparison =
                | my | name | is | john |  X  |
                | X  | X    | is | john | doe |
            reconstituted = [
                ("my", O), ("name", O), ("is", O), ("john", B-PER), ("doe", I-PER)
            ]
        """
        start_indices = [token["start"] for token in first_segment]
        end_indices = [token["end"] for token in second_segment]
        shift_start_index = start_indices.index(second_segment[0]["start"])
        shift_end_index = end_indices.index(first_segment[-1]["end"]) + 1
        if shift_start_index == 0:
            logger.warning(
                "Unable to retrieve previous entity tag to overlapping stride "
                "segments. This could indicate too small a max length or too "
                "large a stride"
            )
            previous_entity = ""
        else:
            previous_entity = first_segment[shift_start_index - 1]["entity"]
        combined_tokens = self.combine_overlapping_tokens(
            first_segment[shift_start_index:],
            second_segment[:shift_end_index],
            reconstitution_strategy,
            previous_entity,
        )
        reconstituted = (
            first_segment[:shift_start_index]
            + combined_tokens
            + second_segment[shift_end_index:]
        )
        # Reset index values in-place
        start_index_shift = reconstituted[0]["index"]
        for i, token in enumerate(reconstituted):
            token["index"] = i + start_index_shift
        return reconstituted

    def combine_overlapping_tokens(
        self,
        first_tokens: List[Dict],
        second_tokens: List[Dict],
        reconstitution_strategy: ReconstitutionStrategy,
        previous_entity: str = "",
    ) -> List[Dict]:
        """Perform the comparison and reconstitution across a pair of
        overlapping tokens.
        By overlapping, we mean two sequences of tokens which have two predictions
        for each token, due to their inclusion in the stride text section used
        when truncating.

        Token reconstitution is performed differently depending on the
        reconstitution strategy given as a postprocessing parameter."""
        # TODO Include some consideration for grouping entity tokens
        combined_tokens: List[Dict] = []
        if reconstitution_strategy == ReconstitutionStrategy.FIRST:
            return first_tokens
        elif reconstitution_strategy in {
            ReconstitutionStrategy.ENTITIES,
            ReconstitutionStrategy.FIRST_ENTITIES,
        }:
            for first, second in zip(first_tokens, second_tokens):
                _, prev_tag = self.get_tag(previous_entity)
                first_ent = first["entity"]
                second_ent = second["entity"]
                first_bi, first_tag = self.get_tag(first_ent)
                second_bi, second_tag = self.get_tag(second_ent)
                ignore_first = first_ent in self.reconstitute_ignore_tags
                ignore_second = second_ent in self.reconstitute_ignore_tags
                if first_tag == prev_tag and first_bi == "I":
                    token_to_append = first
                elif second_tag == prev_tag and second_bi == "I":
                    token_to_append = second
                elif not ignore_first and ignore_second:
                    token_to_append = first
                elif ignore_first and not ignore_second:
                    token_to_append = second
                else:
                    if reconstitution_strategy == ReconstitutionStrategy.ENTITIES:
                        first_score = first["score"]
                        second_score = second["score"]
                        token_to_append = (
                            first if first_score >= second_score else second
                        )
                    else:
                        token_to_append = first
                combined_tokens.append(token_to_append)
                previous_entity = token_to_append["entity"]
        elif reconstitution_strategy == ReconstitutionStrategy.MAX_SCORE:
            for first, second in zip(first_tokens, second_tokens):
                first_score = first["score"]
                second_score = second["score"]
                token_to_append = first if first_score >= second_score else second
                combined_tokens.append(token_to_append)
        else:
            raise ValueError("Invalid reconstitution_strategy")
        return combined_tokens

    @overload  # type: ignore
    def __call__(self, inputs: str, **kwargs) -> List[Any]:
        ...

    @overload  # type: ignore
    def __call__(self, inputs: List[str], **kwargs) -> List[List[Any]]:
        ...

    def __call__(  # type: ignore
        self, inputs: Union[str, List[str]], **kwargs
    ) -> Union[List[Any], List[List[Any]]]:
        result = super().__call__(inputs, **kwargs)
        unpacked_result = self._unpack_result(result, inputs)
        return unpacked_result

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params,
        forward_params,
        postprocess_params,
    ):
        # TODO Should this use the PipelineWrapChunkIterator here? It's where
        #  this wrapping belongs, but is more code to duplicate
        if isinstance(inputs, str):
            inputs = [[inputs]]
        elif isinstance(inputs, list) and all(isinstance(i, str) for i in inputs):
            inputs = [inputs]
        return super().get_iterator(
            inputs,
            num_workers=num_workers,
            batch_size=batch_size,
            preprocess_params=preprocess_params,
            forward_params=forward_params,
            postprocess_params=postprocess_params,
        )

    @overload
    def _unpack_result(self, output: List, inputs: str) -> List[Any]:
        ...

    @overload
    def _unpack_result(self, output: List, inputs: List[str]) -> List[List[Any]]:
        ...

    def _unpack_result(
        self, output: List, inputs: Union[str, List[str]]
    ) -> Union[List[Any], List[List[Any]]]:
        """Unpack the result based on the shape of the inputs that were passed.
        This handles cases where the iterator has packed the results into an
        additional list and ensures that the result is as the user expected.
        Single input sentences are unpacked into a List of EntityType object,"""
        return_single_result = isinstance(inputs, str)
        input_len = 1 if return_single_result else len(inputs)
        if not output:
            return output
        while len(output) != input_len or not all(
            isinstance(item, self.entity_type) for item in output[0]
        ):
            output = output[0]
            if not output:
                return output
        return output[0] if return_single_result else output
