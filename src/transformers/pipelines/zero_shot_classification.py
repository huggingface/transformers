from typing import List, Union

import numpy as np

from ..file_utils import add_end_docstrings
from ..tokenization_utils import TruncationStrategy
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline


logger = logging.get_logger(__name__)


class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",")]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )

        if isinstance(sequences, str):
            sequences = [sequences]
        labels = self._parse_labels(labels)

        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

        return sequence_pairs


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotClassificationPipeline(Pipeline):
    """
    NLI-based zero-shot classification pipeline using a :obj:`ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model. Then, the logit for `entailment` is taken as the logit for the candidate
    label being valid. Any NLI model can be used, but the id of the `entailment` label must be included in the model
    config's :attr:`~transformers.PretrainedConfig.label2id`.

    This NLI pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task identifier:
    :obj:`"zero-shot-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list
    of available models on `huggingface.co/models <https://huggingface.co/models?search=nli>`__.
    """

    def __init__(self, args_parser=ZeroShotClassificationArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args_parser = args_parser
        if self.entailment_id == -1:
            logger.warning(
                "Failed to determine 'entailment' label id from the label2id mapping in the model config. Setting to "
                "-1. Define a descriptive label2id mapping in the model config to ensure correct outputs."
            )

    @property
    def entailment_id(self):
        for label, ind in self.model.config.label2id.items():
            if label.lower().startswith("entail"):
                return ind
        return -1

    def _parse_and_tokenize(
        self,
        sequences,
        candidate_labels,
        hypothesis_template,
        padding=True,
        add_special_tokens=True,
        truncation=TruncationStrategy.ONLY_FIRST,
        **kwargs
    ):
        """
        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
        """
        sequence_pairs = self._args_parser(sequences, candidate_labels, hypothesis_template)
        inputs = self.tokenizer(
            sequence_pairs,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            padding=padding,
            truncation=truncation,
        )

        return inputs

    def __call__(
        self,
        sequences: Union[str, List[str]],
        candidate_labels,
        hypothesis_template="This example is {}.",
        multi_label=False,
        **kwargs,
    ):
        """
        Classify the sequence(s) given as inputs. See the :obj:`~transformers.ZeroShotClassificationPipeline`
        documentation for more information.

        Args:
            sequences (:obj:`str` or :obj:`List[str]`):
                The sequence(s) to classify, will be truncated if the model input is too large.
            candidate_labels (:obj:`str` or :obj:`List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (:obj:`str`, `optional`, defaults to :obj:`"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. For example, the default
                template is :obj:`"This example is {}."` With the candidate label :obj:`"sports"`, this would be fed
                into the model like :obj:`"<cls> sequence to classify <sep> This example is sports . <sep>"`. The
                default template works well in many cases, but it may be worthwhile to experiment with different
                templates depending on the task setting.
            multi_label (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not multiple candidate labels can be true. If :obj:`False`, the scores are normalized such
                that the sum of the label likelihoods for each sequence is 1. If :obj:`True`, the labels are considered
                independent and probabilities are normalized for each candidate by doing a softmax of the entailment
                score vs. the contradiction score.

        Return:
            A :obj:`dict` or a list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **sequence** (:obj:`str`) -- The sequence for which this is the output.
            - **labels** (:obj:`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (:obj:`List[float]`) -- The probabilities for each of the labels.
        """
        if "multi_class" in kwargs and kwargs["multi_class"] is not None:
            multi_label = kwargs.pop("multi_class")
            logger.warn(
                "The `multi_class` argument has been deprecated and renamed to `multi_label`. "
                "`multi_class` will be removed in a future version of Transformers."
            )

        if sequences and isinstance(sequences, str):
            sequences = [sequences]

        outputs = super().__call__(sequences, candidate_labels, hypothesis_template)
        num_sequences = len(sequences)
        candidate_labels = self._args_parser._parse_labels(candidate_labels)
        reshaped_outputs = outputs.reshape((num_sequences, len(candidate_labels), -1))

        if len(candidate_labels) == 1:
            multi_label = True

        if not multi_label:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = reshaped_outputs[..., self.entailment_id]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)
        else:
            # softmax over the entailment vs. contradiction dim for each label independently
            entailment_id = self.entailment_id
            contradiction_id = -1 if entailment_id == 0 else 0
            entail_contr_logits = reshaped_outputs[..., [contradiction_id, entailment_id]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            scores = scores[..., 1]

        result = []
        for iseq in range(num_sequences):
            top_inds = list(reversed(scores[iseq].argsort()))
            result.append(
                {
                    "sequence": sequences if isinstance(sequences, str) else sequences[iseq],
                    "labels": [candidate_labels[i] for i in top_inds],
                    "scores": scores[iseq][top_inds].tolist(),
                }
            )

        if len(result) == 1:
            return result[0]
        return result
