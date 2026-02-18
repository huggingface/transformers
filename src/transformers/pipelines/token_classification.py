import types
import warnings
from typing import Any, Dict, List, Tuple, overload

import numpy as np

from ..models.bert.tokenization_bert_legacy import BasicTokenizer
from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_torch_available,
)
from .base import ArgumentHandler, ChunkPipeline, Dataset, build_pipeline_init_args


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES


class TokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, inputs: str | list[str], **kwargs):
        is_split_into_words = kwargs.get("is_split_into_words", False)
        delimiter = kwargs.get("delimiter")

        if inputs is not None and isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            inputs = list(inputs)
            batch_size = len(inputs)
        elif isinstance(inputs, str):
            inputs = [inputs]
            batch_size = 1
        elif Dataset is not None and isinstance(inputs, Dataset) or isinstance(inputs, types.GeneratorType):
            return inputs, is_split_into_words, None, delimiter
        else:
            raise ValueError("At least one input is required.")

        offset_mapping = kwargs.get("offset_mapping")
        if offset_mapping:
            if isinstance(offset_mapping, list) and isinstance(offset_mapping[0], tuple):
                offset_mapping = [offset_mapping]
            if len(offset_mapping) != batch_size:
                raise ValueError("offset_mapping should have the same batch size as the input")
        return inputs, is_split_into_words, offset_mapping, delimiter


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
        ignore_labels (`list[str]`, defaults to `["O"]`):
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
                  end up with different tags. Word entity will simply be the token with the maximum score.
        use_fast_grouping (`bool`, *optional*, defaults to `False`):
            Enable the optimised BIO-grouping path. When `True` the pipeline skips `gather_pre_entities` +
            `aggregate` and instead runs a single numpy-vectorised pass over token probabilities.
            **Constraints**: only supported with fast tokenizers that produce offset mappings; always applies
            the AVERAGE aggregation strategy; `aggregation_strategy` must be `"none"` (or omitted) when
            this flag is set.""",
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

    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    def __init__(self, args_parser=TokenClassificationArgumentHandler(), use_fast_grouping: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.check_model_type(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self._args_parser = args_parser
        self.use_fast_grouping = use_fast_grouping

        # Pre-compute label lookup tables used by the fast grouping path.
        self._o_label_ids: set = set()
        self._id2bio: np.ndarray | None = None
        self._id2tag: np.ndarray | None = None
        if self.use_fast_grouping:
            self._init_label_maps()

    def _init_label_maps(self) -> None:
        """
        Build label-index look-up tables consumed by :meth:`group_entities`.

        Must be (re-)called whenever ``model.config.id2label`` changes.
        Populates:
          - ``_id2bio``  – numpy array mapping label index → BIO prefix ("B", "I" or "O")
          - ``_id2tag``  – numpy array mapping label index → entity type string
          - ``_o_label_ids`` – set of label indices that represent the Outside class
        """
        id2label = self.model.config.id2label
        num_labels = len(id2label)

        self._id2bio = np.empty(num_labels, dtype=object)
        self._id2tag = np.empty(num_labels, dtype=object)
        o_candidates = []
        for i, label in id2label.items():
            if label.startswith("B-"):
                self._id2bio[i] = "B"
                self._id2tag[i] = label[2:]
            elif label.startswith("I-"):
                self._id2bio[i] = "I"
                self._id2tag[i] = label[2:]
            else:
                # Everything that is neither B- nor I- is treated as Outside
                self._id2bio[i] = "O"
                self._id2tag[i] = label
                o_candidates.append(i)

        # Keep ALL O-like label indices so that PAD, SEP, etc. are also ignored
        self._o_label_ids = set(o_candidates) if o_candidates else {0}

    @staticmethod
    def _strip_bio_prefix(tag: str) -> str:
        """Return the entity type with any leading B-/I- prefix removed."""
        return tag.split("-", 1)[-1] if "-" in tag else tag

    def _sanitize_parameters(
        self,
        ignore_labels=None,
        aggregation_strategy: AggregationStrategy | None = None,
        offset_mapping: list[tuple[int, int]] | None = None,
        is_split_into_words: bool = False,
        stride: int | None = None,
        delimiter: str | None = None,
    ):
        preprocess_params = {}
        preprocess_params["is_split_into_words"] = is_split_into_words

        if is_split_into_words:
            preprocess_params["delimiter"] = " " if delimiter is None else delimiter

        if offset_mapping is not None:
            preprocess_params["offset_mapping"] = offset_mapping

        postprocess_params = {}
        if aggregation_strategy is not None:
            if isinstance(aggregation_strategy, str):
                aggregation_strategy = AggregationStrategy[aggregation_strategy.upper()]
            if getattr(self, "use_fast_grouping", False) and aggregation_strategy != AggregationStrategy.NONE:
                raise ValueError(
                    "`use_fast_grouping=True` always applies the AVERAGE aggregation strategy internally. "
                    "Do not pass `aggregation_strategy` when `use_fast_grouping` is enabled, or set "
                    '`aggregation_strategy="none"` explicitly.'
                )
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

    @overload
    def __call__(self, inputs: str, **kwargs: Any) -> list[dict[str, str]]: ...

    @overload
    def __call__(self, inputs: list[str], **kwargs: Any) -> list[list[dict[str, str]]]: ...

    def __call__(self, inputs: str | list[str], **kwargs: Any) -> list[dict[str, str]] | list[list[dict[str, str]]]:
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (`str` or `List[str]`):
                One or several texts (or one list of texts) for token classification. Can be pre-tokenized when
                `is_split_into_words=True`.

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

        _inputs, is_split_into_words, offset_mapping, delimiter = self._args_parser(inputs, **kwargs)
        kwargs["is_split_into_words"] = is_split_into_words
        kwargs["delimiter"] = delimiter
        if is_split_into_words and not all(isinstance(input, list) for input in inputs):
            return super().__call__([inputs], **kwargs)
        if offset_mapping:
            kwargs["offset_mapping"] = offset_mapping

        return super().__call__(inputs, **kwargs)

    def preprocess(self, sentence, offset_mapping=None, **preprocess_params):
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        truncation = self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0

        word_to_chars_map = None
        is_split_into_words = preprocess_params["is_split_into_words"]
        if is_split_into_words:
            delimiter = preprocess_params["delimiter"]
            if not isinstance(sentence, list):
                raise ValueError("When `is_split_into_words=True`, `sentence` must be a list of tokens.")
            words = sentence
            sentence = delimiter.join(words)  # Recreate the sentence string for later display and slicing
            # This map will allow to convert back word => char indices
            word_to_chars_map = []
            delimiter_len = len(delimiter)
            char_offset = 0
            for word in words:
                word_to_chars_map.append((char_offset, char_offset + len(word)))
                char_offset += len(word) + delimiter_len

            # We use `words` as the actual input for the tokenizer
            text_to_tokenize = words
            tokenizer_params["is_split_into_words"] = True
        else:
            if not isinstance(sentence, str):
                raise ValueError("When `is_split_into_words=False`, `sentence` must be an untokenized string.")
            text_to_tokenize = sentence

        inputs = self.tokenizer(
            text_to_tokenize,
            return_tensors="pt",
            truncation=truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
            **tokenizer_params,
        )

        if is_split_into_words and not self.tokenizer.is_fast:
            raise ValueError("is_split_into_words=True is only supported with fast tokenizers.")

        inputs.pop("overflow_to_sample_mapping", None)
        num_chunks = len(inputs["input_ids"])

        for i in range(num_chunks):
            model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            if offset_mapping is not None:
                model_inputs["offset_mapping"] = offset_mapping

            model_inputs["sentence"] = sentence if i == 0 else None
            model_inputs["is_last"] = i == num_chunks - 1
            if word_to_chars_map is not None:
                model_inputs["word_ids"] = inputs.word_ids(i)
                model_inputs["word_to_chars_map"] = word_to_chars_map

            yield model_inputs

    def _forward(self, model_inputs):
        # Forward
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")
        word_ids = model_inputs.pop("word_ids", None)
        word_to_chars_map = model_inputs.pop("word_to_chars_map", None)

        output = self.model(**model_inputs)
        logits = output["logits"] if isinstance(output, dict) else output[0]

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "is_last": is_last,
            "word_ids": word_ids,
            "word_to_chars_map": word_to_chars_map,
            **model_inputs,
        }

    def postprocess(self, all_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ["O"]
        all_entities = []

        # Get map from the first output, it's the same for all chunks
        word_to_chars_map = all_outputs[0].get("word_to_chars_map")

        for model_outputs in all_outputs:
            if model_outputs["logits"][0].dtype in (torch.bfloat16, torch.float16):
                logits = model_outputs["logits"][0].to(torch.float32).numpy()
            else:
                logits = model_outputs["logits"][0].numpy()

            sentence = all_outputs[0]["sentence"]
            input_ids = model_outputs["input_ids"][0]
            offset_mapping = (
                model_outputs["offset_mapping"][0].numpy()
                if model_outputs["offset_mapping"] is not None
                else None
            )
            special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()
            word_ids = model_outputs.get("word_ids")
            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

            # Fast path requires: flag set, offsets available, label maps initialised,
            # and a fast tokenizer (so offset_mapping is a real numpy array).
            use_fast = (
                self.use_fast_grouping
                and offset_mapping is not None
                and self._id2bio is not None
                and self._id2tag is not None
                and self.tokenizer.is_fast
            )

            if use_fast:
                # Vectorised special-token filter using numpy boolean mask
                keep_mask = ~special_tokens_mask.astype(bool)
                kept_input_ids = input_ids.numpy()[keep_mask]
                kept_scores = scores[keep_mask]                     # (N, num_labels)
                kept_offsets_arr = offset_mapping[keep_mask]        # (N, 2)

                if len(kept_scores) > 0:
                    # Resolve token strings from IDs in one batch call where possible
                    token_strings = [
                        self.tokenizer.convert_ids_to_tokens(int(tid)) for tid in kept_input_ids
                    ]
                    kept_offsets = [(int(s), int(e)) for s, e in kept_offsets_arr]

                    # Use tokenizer word_ids for subword detection when available
                    # (works for WordPiece, BPE, SentencePiece, etc.)
                    if word_ids is not None:
                        kept_word_ids = np.array(word_ids, dtype=object)[keep_mask]
                        # A token is a subword if it shares its word_id with the
                        # preceding non-special token
                        is_subword_list = [False]
                        for k in range(1, len(kept_word_ids)):
                            is_subword_list.append(
                                kept_word_ids[k] is not None
                                and kept_word_ids[k] == kept_word_ids[k - 1]
                            )
                    else:
                        # Fallback for tokenizers that expose ## prefix
                        is_subword_list = [t.startswith("##") for t in token_strings]

                    label_ids_arr = kept_scores.argmax(axis=1)
                    label_scores_arr = kept_scores[np.arange(len(kept_scores)), label_ids_arr]

                    fast_entities = self.group_entities(
                        token_strings,
                        label_ids_arr,
                        label_scores_arr,
                        kept_scores,
                        kept_offsets,
                        sentence,
                        threshold=0.0,
                        ignore_labels=ignore_labels,
                        is_subword=is_subword_list,
                    )
                    # Normalise keys to match legacy pipeline output schema
                    grouped_entities = [
                        {
                            "entity_group": ent["label"],
                            "score": ent["score"],
                            "word": ent["text"],
                            "start": ent["start"],
                            "end": ent["end"],
                        }
                        for ent in fast_entities
                    ]
                else:
                    grouped_entities = []
            else:
                pre_entities = self.gather_pre_entities(
                    sentence,
                    input_ids,
                    scores,
                    offset_mapping,
                    special_tokens_mask,
                    aggregation_strategy,
                    word_ids=word_ids,
                    word_to_chars_map=word_to_chars_map,
                )
                grouped_entities = self.aggregate(pre_entities, aggregation_strategy)

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
                if (
                    current_length > previous_length
                    or current_length == previous_length
                    and entity["score"] > previous_entity["score"]
                ):
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
        offset_mapping: list[tuple[int, int]] | None,
        special_tokens_mask: np.ndarray,
        aggregation_strategy: AggregationStrategy,
        word_ids: list[int | None] | None = None,
        word_to_chars_map: list[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]

                # If the input is pre-tokenized, we need to rescale the offsets to the absolute sentence.
                if word_ids is not None and word_to_chars_map is not None:
                    word_index = word_ids[idx]
                    if word_index is not None:
                        start_char, _ = word_to_chars_map[word_index]
                        start_ind += start_char
                        end_ind += start_char

                if not isinstance(start_ind, int):
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

    def aggregate(self, pre_entities: list[dict], aggregation_strategy: AggregationStrategy) -> list[dict]:
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
        return self.group_entities_deprecated(entities)

    def aggregate_word(self, entities: list[dict], aggregation_strategy: AggregationStrategy) -> dict:
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

    def aggregate_words(self, entities: list[dict], aggregation_strategy: AggregationStrategy) -> list[dict]:
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

    def group_sub_entities(self, entities: list[dict]) -> dict:
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

    def get_tag(self, entity_name: str) -> tuple[str, str]:
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

    def group_entities_deprecated(self, entities: list[dict]) -> list[dict]:
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

    def group_entities(
        self,
        tokens: List[str],
        label_ids: np.ndarray,
        scores: np.ndarray,
        chunk_probs: np.ndarray,
        offsets: List[Tuple[int, int]],
        chunk_text: str,
        threshold: float,
        ignore_labels: List[str] | None = None,
        is_subword: List[bool] | None = None,
    ) -> List[Dict]:
        """
        Optimised BIO-grouping pass used when ``use_fast_grouping=True``.

        Applies the AVERAGE aggregation strategy: for multi-token words, per-label
        probabilities are averaged across sub-tokens before the winning label is chosen.

        Args:
            tokens: Token strings for the non-special tokens in the chunk.
            label_ids: Argmax label index per token, shape ``(N,)``.
            scores: Per-token max probability, shape ``(N,)``.
            chunk_probs: Full probability matrix, shape ``(N, num_labels)``.
            offsets: ``(start_char, end_char)`` pairs aligned with *tokens*.
            chunk_text: The original sentence string (used for span extraction).
            threshold: Tokens whose winning probability is below this value are
                treated as Outside.
            ignore_labels: Label strings to suppress (defaults to ``["O"]``).
            is_subword: Boolean mask indicating which tokens are sub-tokens of the
                preceding word. When ``None``, the ``##`` prefix heuristic is used
                as a fallback for WordPiece models.

        Returns:
            List of entity dicts with keys ``start``, ``end``, ``text``,
            ``label``, ``score``.
        """
        if not tokens:
            return []

        if ignore_labels is None:
            ignore_labels = ["O"]
        ignore_label_set = set(ignore_labels)
        ignore_label_ids = {
            idx
            for idx, lbl in self.model.config.id2label.items()
            if lbl in ignore_label_set or self._strip_bio_prefix(lbl) in ignore_label_set
        }
        # Merge with the O-label set from _init_label_maps
        all_skip_ids = self._o_label_ids | ignore_label_ids

        # Resolve subword membership
        if is_subword is not None:
            is_subword_arr = np.asarray(is_subword, dtype=bool)
        else:
            # Fallback: WordPiece ## prefix
            is_subword_arr = np.fromiter(
                (t.startswith("##") for t in tokens), count=len(tokens), dtype=bool
            )

        # A word starts at every non-subword token
        word_starts = np.where(~is_subword_arr)[0]
        if len(word_starts) == 0:
            return []

        # Build half-open [start, end) ranges for each word
        word_ends = np.empty(len(word_starts), dtype=int)
        word_ends[:-1] = word_starts[1:]
        word_ends[-1] = len(tokens)
        word_ranges = list(zip(word_starts.tolist(), word_ends.tolist()))

        entities: List[Dict] = []

        # BIO state machine
        curr_ent_tag: str | None = None
        curr_ent_start_char: int = -1
        curr_ent_end_char: int = -1
        curr_ent_scores: List[float] = []

        def flush_entity() -> None:
            nonlocal curr_ent_tag, curr_ent_start_char, curr_ent_end_char, curr_ent_scores
            if curr_ent_tag is not None:
                entities.append(
                    {
                        "start": curr_ent_start_char,
                        "end": curr_ent_end_char,
                        "text": chunk_text[curr_ent_start_char:curr_ent_end_char],
                        "label": self._strip_bio_prefix(curr_ent_tag),
                        "score": float(np.mean(curr_ent_scores)),
                    }
                )
                curr_ent_tag = None
                curr_ent_scores = []

        for start, end in word_ranges:
            if end - start == 1:
                # Single-token word — no averaging needed
                w_label_id = int(label_ids[start])
                avg_score = float(scores[start])
            else:
                # Multi-token word — average probabilities across sub-tokens
                avg_probs = np.mean(chunk_probs[start:end], axis=0)
                w_label_id = int(np.argmax(avg_probs))
                avg_score = float(avg_probs[w_label_id])

            # Skip Outside labels and low-confidence predictions
            if avg_score < threshold or w_label_id in all_skip_ids:
                flush_entity()
                continue

            bi = self._id2bio[w_label_id]
            tag = self._id2tag[w_label_id]

            if curr_ent_tag == tag and bi == "I":
                # Continue the current entity span
                curr_ent_end_char = offsets[end - 1][1]
                curr_ent_scores.append(avg_score)
            else:
                # Start a new entity (B-tag or tag change)
                flush_entity()
                curr_ent_tag = tag
                curr_ent_start_char = offsets[start][0]
                curr_ent_end_char = offsets[end - 1][1]
                curr_ent_scores = [avg_score]

        flush_entity()
        return entities


NerPipeline = TokenClassificationPipeline
