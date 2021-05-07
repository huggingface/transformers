from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from ..file_utils import AggregationStrategy, add_end_docstrings, is_tf_available, is_torch_available
from ..modelcard import ModelCard
from ..models.bert.tokenization_bert import BasicTokenizer
from ..tokenization_utils import PreTrainedTokenizer
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline


if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel

if is_tf_available():

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


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
        else:
            raise ValueError("At least one input is required.")

        offset_mapping = kwargs.get("offset_mapping")
        if offset_mapping:
            if isinstance(offset_mapping, list) and isinstance(offset_mapping[0], tuple):
                offset_mapping = [offset_mapping]
            if len(offset_mapping) != batch_size:
                raise ValueError("offset_mapping should have the same batch size as the input")
        return inputs, offset_mapping


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        ignore_labels (:obj:`List[str]`, defaults to :obj:`["O"]`):
            A list of labels to ignore.
        grouped_entities (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to group the tokens corresponding to the same entity together in the predictions or not.
    """,
)
class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using any :obj:`ModelForTokenClassification`. See the `named entity recognition
    examples <../task_summary.html#named-entity-recognition>`__ for more information.

    This token recognition pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location
    or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=token-classification>`__.
    """

    default_input_names = "sequences"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = TokenClassificationArgumentHandler(),
        device: int = -1,
        binary_output: bool = False,
        ignore_labels=["O"],
        task: str = "",
        grouped_entities: bool = False,
        aggregation_strategy: Union[str, AggregationStrategy] = AggregationStrategy.FIRST,
        ignore_subwords: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            device=device,
            binary_output=binary_output,
            task=task,
        )

        self.check_model_type(
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self._args_parser = args_parser
        self.ignore_labels = ignore_labels
        self.grouped_entities = grouped_entities
        if isinstance(aggregation_strategy, str):
            self.aggregation_strategy = AggregationStrategy[aggregation_strategy.upper()]
        else:
            self.aggregation_strategy = aggregation_strategy
        self.ignore_subwords = ignore_subwords

        if self.ignore_subwords and not self.tokenizer.is_fast:
            raise ValueError(
                "Slow tokenizers cannot ignore subwords. Please set the `ignore_subwords` option"
                "to `False` or use a fast tokenizer."
            )

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with
            :obj:`grouped_entities=True`) with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `grouped_entities` is set to True.
            - **index** (:obj:`int`, only present when ``self.grouped_entities=False``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
            - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
        """

        _inputs, offset_mappings = self._args_parser(inputs, **kwargs)

        answers = []

        for i, sentence in enumerate(_inputs):

            # Manage correct placement of the tensors
            with self.device_placement():

                tokens = self.tokenizer(
                    sentence,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    truncation=True,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=self.tokenizer.is_fast,
                )
                if self.tokenizer.is_fast:
                    offset_mapping = tokens.pop("offset_mapping").cpu().numpy()[0]
                elif offset_mappings:
                    offset_mapping = offset_mappings[i]
                else:
                    offset_mapping = None

                special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()[0]

                # Forward
                if self.framework == "tf":
                    entities = self.model(tokens.data)[0][0].numpy()
                    input_ids = tokens["input_ids"].numpy()[0]
                else:
                    with torch.no_grad():
                        tokens = self.ensure_tensor_on_device(**tokens)
                        entities = self.model(**tokens)[0][0].cpu().numpy()
                        input_ids = tokens["input_ids"].cpu().numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)

            entities = []
            # Filter special_tokens
            filtered_labels_idx = [idx for idx in range(score.shape[0]) if not special_tokens_mask[idx]]

            for idx in filtered_labels_idx:
                entity = {}
                if offset_mapping is not None:
                    start_ind, end_ind = offset_mapping[idx]
                    entity["start"], entity["end"] = (start_ind, end_ind)
                    word_ref = sentence[start_ind:end_ind]
                    word = self.tokenizer.convert_ids_to_tokens([int(input_ids[idx])])[0]
                    entity["word"] = word
                    entity["is_subword"] = len(word_ref) != len(word)

                    if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                        entity["word"] = word_ref
                        entity["is_subword"] = False
                else:
                    entity["word"] = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))

                    entity["start"] = None
                    entity["end"] = None

                entity["score"] = score[idx]
                entity["index"] = idx

                entities += [entity]

            if self.ignore_subwords:
                entities = self.set_subwords_label(entities)
            else:
                for entity in entities:
                    label_idx = entity["score"].argmax()
                    label = self.model.config.id2label[label_idx]
                    entity["entity"] = label
                    entity["score"] = entity["score"][label_idx].item()

            # filter out ignored labels
            entities = [entity for entity in entities if entity["entity"] not in self.ignore_labels]
            if self.grouped_entities:
                answers += [self.group_entities(entities)]
            # Append ungrouped entities
            else:
                answers += [entities]

        if len(answers) == 1:
            return answers[0]
        return answers

    def set_subwords_label(self, entities: List[dict]) -> List[dict]:
        strategy = self.aggregation_strategy

        def set_labels(sub_words: List[dict]) -> dict:
            scores = np.stack([sub["score"] for sub in sub_words])
            if strategy == AggregationStrategy.FIRST:
                # get label of first sub-word
                max_label_idx = scores[0].argmax()
                score = scores[0][max_label_idx]
                label = self.model.config.id2label[max_label_idx]
            elif strategy == AggregationStrategy.MAX:
                max_label_idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)[1]
                score = scores.max()
                label = self.model.config.id2label[max_label_idx]
            elif strategy == AggregationStrategy.AVERAGE:
                avg_scores = np.mean(scores, axis=0)
                max_label_idx = avg_scores.argmax()
                score = avg_scores[max_label_idx]
                label = self.model.config.id2label[max_label_idx]
            else:
                raise ValueError(f"Invalid value {strategy} for option `aggregation_strategy`")

            for idx, sub in enumerate(sub_words):
                sub["entity"] = label
                sub["score"] = score.item()

            return sub_words

        subword_indices = np.where([entity["is_subword"] for entity in entities])[0]
        if subword_indices.size == 0:
            adjacent_subwords = []
        else:
            # find non-consecutive indices to identify separate clusters of subwords
            cluster_edges = np.where(np.diff(subword_indices) != 1)[0]
            # Sets of adjacent subwords indices, e.g.
            # ['Sir', 'Test', '##y', 'M', '##c', '##T', '##est', 'is', 'test', '##iful']
            # --> [[2],[4,5,6],[9]]
            adjacent_subwords = np.split(subword_indices, cluster_edges + 1)  # shift edge by 1

        word_indices = []
        start = 0
        for subwords in adjacent_subwords:
            root_word = subwords[0] - 1
            word_indices += [[idx] for idx in range(start, root_word)]
            word_indices += [[root_word] + list(subwords)]
            start = subwords[-1] + 1
        word_indices += [[idx] for idx in range(start, len(entities))]

        entities_with_label = []
        for word_idx in word_indices:
            subwords = [entities[idx] for idx in word_idx]
            subwords = set_labels(subwords)
            if self.ignore_subwords and len(subwords) > 1:
                subwords[0]["word"] += "".join([sub["word"].split("##")[1] for sub in subwords[1:]])
                subwords = [subwords[0]]
            entities_with_label += subwords

        return entities_with_label

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
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

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:

            is_last_idx = entity["index"] == last_idx
            is_subword = self.ignore_subwords and entity["is_subword"]
            if not entity_group_disagg:
                entity_group_disagg += [entity]
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" suffixes
            # Shouldn't merge if both entities are B-type
            if (
                (
                    entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                    and entity["entity"].split("-")[0] != "B"
                )
                and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ) or is_subword:
                # Modify subword type to be previous_type
                if is_subword:
                    entity["entity"] = entity_group_disagg[-1]["entity"].split("-")[-1]
                    entity["score"] = np.nan  # set ignored scores to nan and use np.nanmean

                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]

        return entity_groups


NerPipeline = TokenClassificationPipeline
