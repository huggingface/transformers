from typing import Dict

import numpy as np

from ..file_utils import ExplicitEnum, add_end_docstrings, is_tf_available, is_torch_available
from .base import PIPELINE_INIT_ARGS, GenericTensor, Pipeline


if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class ClassificationFunction(ExplicitEnum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        return_all_scores (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to return all prediction scores or just the one of the predicted class.
        function_to_apply (:obj:`str`, `optional`, defaults to :obj:`"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - :obj:`"default"`: if the model has a single label, will apply the sigmoid function on the output. If the
              model has several labels, will apply the softmax function on the output.
            - :obj:`"sigmoid"`: Applies the sigmoid function on the output.
            - :obj:`"softmax"`: Applies the softmax function on the output.
            - :obj:`"none"`: Does not apply any function on the output.
    """,
)
class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using any :obj:`ModelForSequenceClassification`. See the `sequence classification
    examples <../task_summary.html#sequence-classification>`__ for more information.

    This text classification pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"sentiment-analysis"` (for classifying sequences according to positive or negative
    sentiments).

    If multiple classification labels are available (:obj:`model.config.num_labels >= 2`), the pipeline will run a
    softmax over the results. If there is a single label, the pipeline will run a sigmoid over the result.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    the up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=text-classification>`__.
    """

    return_all_scores = False
    function_to_apply = ClassificationFunction.NONE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "function_to_apply" not in kwargs:
            # Default value before `set_parameters`
            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply
            else:
                function_to_apply = ClassificationFunction.NONE
            self.function_to_apply = function_to_apply

        self.check_model_type(
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
        )

    def set_parameters(self, return_all_scores=None, function_to_apply=None, **kwargs):
        self.tokenizer_kwargs = {}
        if "truncation" in kwargs:
            self.tokenizer_kwargs["truncation"] = kwargs["truncation"]

        if hasattr(self.model.config, "return_all_scores") and return_all_scores is None:
            return_all_scores = self.model.config.return_all_scores
        if return_all_scores is not None:
            self.return_all_scores = return_all_scores

        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction[function_to_apply.upper()]

        if function_to_apply is not None:
            self.function_to_apply = function_to_apply

        self.tokenizer_kwargs = {}

    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of prompts) to classify.
            return_all_scores (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to return scores for all labels.
            function_to_apply (:obj:`str`, `optional`, defaults to :obj:`"default"`):
                The function to apply to the model outputs in order to retrieve the scores. Accepts four different
                values:

                If this argument is not specified, then it will apply the following functions according to the number
                of labels:

                - If the model has a single label, will apply the sigmoid function on the output.
                - If the model has several labels, will apply the softmax function on the output.

                Possible values are:

                - :obj:`"sigmoid"`: Applies the sigmoid function on the output.
                - :obj:`"softmax"`: Applies the softmax function on the output.
                - :obj:`"none"`: Does not apply any function on the output.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **label** (:obj:`str`) -- The label predicted.
            - **score** (:obj:`float`) -- The corresponding probability.

            If ``self.return_all_scores=True``, one such dictionary is returned per label.
        """
        return super().__call__(*args, **kwargs)

    def preprocess(self, inputs) -> Dict[str, GenericTensor]:
        return_tensors = self.framework
        return self.tokenizer(inputs, return_tensors=return_tensors, **self.tokenizer_kwargs)

    def forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        outputs = model_outputs["logits"][0]
        if self.framework == "pt":
            outputs = outputs.cpu().numpy()
        else:
            outputs = outputs.numpy()

        function_to_apply = self.function_to_apply
        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(outputs)
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(outputs)
        elif function_to_apply == ClassificationFunction.NONE:
            scores = outputs
        else:
            raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")

        if self.return_all_scores:
            return [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)]
        else:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores.max().item()}

    def run_multi(self, inputs):
        return [self.run_single(item)[0] for item in inputs]

    def run_single(self, inputs):
        "This pipeline is odd, and return a list when single item is run"
        return [super().run_single(inputs)]
