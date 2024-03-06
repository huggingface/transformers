import types
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from ..models.bert.tokenization_bert import BasicTokenizer
from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
)
from .base import ArgumentHandler, ChunkPipeline, Dataset, build_pipeline_init_args


if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES


class TokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        if inputs is not None and isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            inputs = list(inputs)
            batch_size = len(inputs)
        elif isinstance(inputs, str):
            inputs = [inputs]
            batch_size = 1
        elif Dataset is not None and isinstance(inputs, Dataset) or isinstance(inputs, types.GeneratorType):
            return inputs, None
        else:
            raise ValueError("At least one input is required.")

        offset_mapping = kwargs.get("offset_mapping")
        if offset_mapping:
            if isinstance(offset_mapping, list) and isinstance(offset_mapping[0], tuple):
                offset_mapping = [offset_mapping]
            if len(offset_mapping) != batch_size:
                raise ValueError("offset_mapping should have the same batch size as the input")
        return inputs, offset_mapping


class AggregationStrategy(ExplicitEnum):
    """All the valid aggregation strategies for TokenClassificationPipeline"""

    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True),
    r"""
        ignore_labels (`List[str]`, defaults to `["O"]`):
            A list of labels to ignore.
        grouped_entities (`bool`, *optional*, defaults to `False`):
            DEPRECATED, use `aggregation_strategy` instead. Whether or not to group the tokens corresponding to the
            same entity together in the predictions or not.
        stride (`int`, *optional*):
            If stride is provided, the pipeline is applied on all the text. The text is split into chunks of size
            model_max_length. Works only with fast tokenizers and `aggregation_strategy` different from `NONE`. The
            value of this argument defines the number of overlapping tokens between chunks. In other words, the model
            will shift forward by `tokenizer.model_max_length - stride` tokens each step.
        aggregation_strategy (`str`, *optional*, defaults to `"none"`):
            The strategy to fuse (or not) tokens based on the model prediction.

                - "none" : Will simply not do any aggregation and simply return raw results from the model
                - "simple" : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C,
                  I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{"word": ABC, "entity": "TAG"}, {"word": "D",
                  "entity": "TAG2"}, {"word": "E", "entity": "TAG2"}] Notice that two consecutive B tags will end up as
                  different entities. On word based languages, we might end up splitting words undesirably : Imagine
                  Microsoft being tagged as [{"word": "Micro", "entity": "ENTERPRISE"}, {"word": "soft", "entity":
                  "NAME"}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages
                  that support that meaning, which is basically tokens separated by a space). These mitigations will
                  only work on real words, "New york" might still be tagged with two different entities.
                - "first" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot
                  end up with different tags. Words will simply use the tag of the first token of the word when there
                  is ambiguity.
                - "average" : (works only on word based models) Will use the `SIMPLE` strategy except that words,
                  cannot end up with different tags. scores will be averaged first across tokens, and then the maximum
                  label is applied.
                - "max" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot
                  end up with different tags. Word entity will simply be the token with the maximum score.""",
)
class TokenClassificationPipeline(ChunkPipeline):
    """
    Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition
    examples](../task_summary#named-entity-recognition) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> token_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
    >>> sentence = "Je m'appelle jean-baptiste et je vis à montréal"
    >>> tokens = token_classifier(sentence)
    >>> tokens
    [{'entity_group': 'PER', 'score': 0.9931, 'word': 'jean-baptiste', 'start': 12, 'end': 26}, {'entity_group': 'LOC', 'score': 0.998, 'word': 'montréal', 'start': 38, 'end': 47}]

    >>> token = tokens[0]
    >>> # Start and end provide an easy way to highlight words in the original text.
    >>> sentence[token["start"] : token["end"]]
    ' jean-baptiste'

    >>> # Some models use the same idea to do part of speech.
    >>> syntaxer = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")
    >>> syntaxer("My name is Sarah and I live in London")
    [{'entity_group': 'PRON', 'score': 0.999, 'word': 'my', 'start': 0, 'end': 2}, {'entity_group': 'NOUN', 'score': 0.997, 'word': 'name', 'start': 3, 'end': 7}, {'entity_group': 'AUX', 'score': 0.994, 'word': 'is', 'start': 8, 'end': 10}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'sarah', 'start': 11, 'end': 16}, {'entity_group': 'CCONJ', 'score': 0.999, 'word': 'and', 'start': 17, 'end': 20}, {'entity_group': 'PRON', 'score': 0.999, 'word': 'i', 'start': 21, 'end': 22}, {'entity_group': 'VERB', 'score': 0.998, 'word': 'live', 'start': 23, 'end': 27}, {'entity_group': 'ADP', 'score': 0.999, 'word': 'in', 'start': 28, 'end': 30}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'london', 'start': 31, 'end': 37}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This token recognition pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=token-classification).
    """

    default_input_names = "sequences"

    def __init__(self, args_parser=TokenClassificationArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self._args_parser = args_parser

    def _sanitize_parameters(
        self,
        ignore_labels=None,
        grouped_entities: Optional[bool] = None,
        ignore_subwords: Optional[bool] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        offset_mapping: Optional[List[Tuple[int, int]]] = None,
        stride: Optional[int] = None,
    ):
        preprocess_params = {}
        if offset_mapping is not None:
            preprocess_params["offset_mapping"] = offset_mapping

        postprocess_params = {}
        if grouped_entities is not None or ignore_subwords is not None:
            if grouped_entities and ignore_subwords:
                aggregation_strategy = AggregationStrategy.FIRST
            elif grouped_entities and not ignore_subwords:
                aggregation_strategy = AggregationStrategy.SIMPLE
            else:
                aggregation_strategy = AggregationStrategy.NONE

            if grouped_entities is not None:
                warnings.warn(
                    "`grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to"
                    f' `aggregation_strategy="{aggregation_strategy}"` instead.'
                )
            if ignore_subwords is not None:
                warnings.warn(
                    "`ignore_subwords` is deprecated and will be removed in version v5.0.0, defaulted to"
                    f' `aggregation_strategy="{aggregation_strategy}"` instead.'
                )

        if aggregation_strategy is not None:
            if isinstance(aggregation_strategy, str):
                aggregation_strategy = AggregationStrategy[aggregation_strategy.upper()]
            if (
                aggregation_strategy
                in {AggregationStrategy.FIRST, AggregationStrategy.MAX, AggregationStrategy.AVERAGE}
                and not self.tokenizer.is_fast
            ):
                raise ValueError(
                    "Slow tokenizers cannot handle subwords. Please set the `aggregation_strategy` option"
                    ' to `"simple"` or use a fast tokenizer.'
                )
            postprocess_params["aggregation_strategy"] = aggregation_strategy
        if ignore_labels is not None:
            postprocess_params["ignore_labels"] = ignore_labels
        if stride is not None:
            if stride >= self.tokenizer.model_max_length:
                raise ValueError(
                    "`stride` must be less than `tokenizer.model_max_length` (or even lower if the tokenizer adds special tokens)"
                )
            if aggregation_strategy == AggregationStrategy.NONE:
                raise ValueError(
                    "`stride` was provided to process all the text but `aggregation_strategy="
                    f'"{aggregation_strategy}"`, please select another one instead.'
                )
            else:
                if self.tokenizer.is_fast:
                    tokenizer_params = {
                        "return_overflowing_tokens": True,
                        "padding": True,
                        "stride": stride,
                    }
                    preprocess_params["tokenizer_params"] = tokenizer_params
                else:
                    raise ValueError(
                        "`stride` was provided to process all the text but you're using a slow tokenizer."
                        " Please use a fast tokenizer."
                    )
        return preprocess_params, {}, postprocess_params

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (`str` or `List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of `dict`: Each result comes as a list of dictionaries (one for each token in the
            corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy) with
            the following keys:

            - **word** (`str`) -- The token/word classified. This is obtained by decoding the selected tokens. If you
              want to have the exact string in the original sentence, use `start` and `end`.
            - **score** (`float`) -- The corresponding probability for `entity`.
            - **entity** (`str`) -- The entity predicted for that token/word (it is named *entity_group* when
              *aggregation_strategy* is not `"none"`.
            - **index** (`int`, only present when `aggregation_strategy="none"`) -- The index of the corresponding
              token in the sentence.
            - **start** (`int`, *optional*) -- The index of the start of the corresponding entity in the sentence. Only
              exists if the offsets are available within the tokenizer
            - **end** (`int`, *optional*) -- The index of the end of the corresponding entity in the sentence. Only
              exists if the offsets are available within the tokenizer
        """

        _inputs, offset_mapping = self._args_parser(inputs, **kwargs)
        if offset_mapping:
            kwargs["offset_mapping"] = offset_mapping

        return super().__call__(inputs, **kwargs)

    def preprocess(self, sentence, offset_mapping=None, **preprocess_params):
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
            **tokenizer_params,
        )
        inputs.pop("overflow_to_sample_mapping", None)
        num_chunks = len(inputs["input_ids"])

        for i in range(num_chunks):
            if self.framework == "tf":
                model_inputs = {k: tf.expand_dims(v[i], 0) for k, v in inputs.items()}
            else:
                model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            if offset_mapping is not None:
                model_inputs["offset_mapping"] = offset_mapping
            model_inputs["sentence"] = sentence if i == 0 else None
            model_inputs["is_last"] = i == num_chunks - 1

            yield model_inputs

    def _forward(self, model_inputs):
        # Forward
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")
        if self.framework == "tf":
            logits = self.model(**model_inputs)[0]
        else:
            output = self.model(**model_inputs)
            logits = output["logits"] if isinstance(output, dict) else output[0]

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "is_last": is_last,
            **model_inputs,
        }

    def postprocess(self, all_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ["O"]
        all_entities = []
        for model_outputs in all_outputs:
            logits = model_outputs["logits"][0].numpy()
            sentence = all_outputs[0]["sentence"]
            input_ids = model_outputs["input_ids"][0]
            offset_mapping = (
                model_outputs["offset_mapping"][0] if model_outputs["offset_mapping"] is not None else None
            )
            special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()

            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

            if self.framework == "tf":
                input_ids = input_ids.numpy()
                offset_mapping = offset_mapping.numpy() if offset_mapping is not None else None

            pre_entities = self.gather_pre_entities(
                sentence, input_ids, scores, offset_mapping, special_tokens_mask, aggregation_strategy
            )
            grouped_entities = self.aggregate(pre_entities, aggregation_strategy)
            # Filter anything that is in self.ignore_labels
            entities = [
                entity
                for entity in grouped_entities
                if entity.get("entity", None) not in ignore_labels
                and entity.get("entity_group", None) not in ignore_labels
            ]
            all_entities.extend(entities)
        num_chunks = len(all_outputs)
        if num_chunks > 1:
            all_entities = self.aggregate_overlapping_entities(all_entities)
        return all_entities

    def aggregate_overlapping_entities(self, entities):
        if len(entities) == 0:
            return entities
        entities = sorted(entities, key=lambda x: x["start"])
        aggregated_entities = []
        previous_entity = entities[0]
        for entity in entities:
            if previous_entity["start"] <= entity["start"] < previous_entity["end"]:
                current_length = entity["end"] - entity["start"]
                previous_length = previous_entity["end"] - previous_entity["start"]
                if current_length > previous_length:
                    previous_entity = entity
                elif current_length == previous_length and entity["score"] > previous_entity["score"]:
                    previous_entity = entity
            else:
                aggregated_entities.append(previous_entity)
                previous_entity = entity
        aggregated_entities.append(previous_entity)
        return aggregated_entities

    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
        aggregation_strategy: AggregationStrategy,
    ) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                if not isinstance(start_ind, int):
                    if self.framework == "pt":
                        start_ind = start_ind.item()
                        end_ind = end_ind.item()
                word_ref = sentence[start_ind:end_ind]
                if getattr(self.tokenizer, "_tokenizer", None) and getattr(
                    self.tokenizer._tokenizer.model, "continuing_subword_prefix", None
                ):
                    # This is a BPE, word aware tokenizer, there is a correct way
                    # to fuse tokens
                    is_subword = len(word) != len(word_ref)
                else:
                    # This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". Non word aware models cannot do better than this unfortunately.
                    if aggregation_strategy in {
                        AggregationStrategy.FIRST,
                        AggregationStrategy.AVERAGE,
                        AggregationStrategy.MAX,
                    }:
                        warnings.warn(
                            "Tokenizer does not support real words, using fallback heuristic",
                            UserWarning,
                        )
                    is_subword = start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1]

                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.model.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def aggregate_words(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.

        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        if aggregation_strategy in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }:
            raise ValueError("NONE and SIMPLE strategies are invalid for word aggregation")

        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
                word_group = [entity]
        # Last item
        if word_group is not None:
            word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-", 1)[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to I- for continuation.
            bi = "I"
            tag = entity_name
        return bi, tag

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        return entity_groups


NerPipeline = TokenClassificationPipeline
