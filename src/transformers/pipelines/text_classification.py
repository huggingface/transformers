import types
import warnings
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

from ..utils import ExplicitEnum, add_end_docstrings, is_tf_available, is_torch_available
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, ChunkPipeline, Dataset, GenericTensor


if is_tf_available():
    import tensorflow as tf

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


class TextClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for text classification.
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


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        return_all_scores (`bool`, *optional*, defaults to `False`):
            Whether to return all prediction scores or just the one of the predicted class.
        function_to_apply (`str`, *optional*, defaults to `"default"`):
            The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:

            - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model
              has several labels, will apply the softmax function on the output.
            - `"sigmoid"`: Applies the sigmoid function on the output.
            - `"softmax"`: Applies the softmax function on the output.
            - `"none"`: Does not apply any function on the output.
    """,
)
class TextClassificationPipeline(ChunkPipeline):
    """
    Text classification pipeline using any `ModelForSequenceClassification`. See the [sequence classification
    examples](../task_summary#sequence-classification) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
    >>> classifier("This movie is disgustingly good !")
    [{'label': 'POSITIVE', 'score': 1.0}]

    >>> classifier("Director tried too much.")
    [{'label': 'NEGATIVE', 'score': 0.996}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This text classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"sentiment-analysis"` (for classifying sequences according to positive or negative sentiments).

    If multiple classification labels are available (`model.config.num_labels >= 2`), the pipeline will run a softmax
    over the results. If there is a single label, the pipeline will run a sigmoid over the result.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=text-classification).
    """

    return_all_scores = False
    function_to_apply = ClassificationFunction.NONE

    def __init__(self, args_parser=TextClassificationArgumentHandler(), **kwargs):
        self._postprocess_params = {}

        super().__init__(**kwargs)

        self.check_model_type(
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
        )
        self._args_parser = args_parser

    def _sanitize_parameters(
        self,
        return_all_scores=None,
        function_to_apply=None,
        top_k="",
        stride: Optional[int] = None,
        **preprocess_params,
    ):
        # Using "" as default argument because we're going to use `top_k=None` in user code to declare
        # "No top_k"

        postprocess_params = {}
        if hasattr(self.model.config, "return_all_scores") and return_all_scores is None:
            return_all_scores = self.model.config.return_all_scores

        if isinstance(top_k, int) or top_k is None:
            postprocess_params["top_k"] = top_k
            postprocess_params["_legacy"] = self._postprocess_params.get("_legacy", False)
        elif return_all_scores is not None:
            postprocess_params["_legacy"] = self._postprocess_params.get("_legacy", True)
            warnings.warn(
                "`return_all_scores` is now deprecated,  if want a similar funcionality use `top_k=None` instead of"
                " `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`.",
                UserWarning,
            )
            if return_all_scores:
                postprocess_params["top_k"] = None
            else:
                postprocess_params["top_k"] = 1
        else:
            postprocess_params["_legacy"] = self._postprocess_params.get("_legacy", True)

        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction[function_to_apply.upper()]

        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
        if stride is not None:
            if stride >= self.tokenizer.model_max_length:
                raise ValueError(
                    "`stride` must be less than `tokenizer.model_max_length` (or even lower if the tokenizer adds special tokens)"
                )
            if self.tokenizer.is_fast:
                tokenizer_params = {
                    "return_overflowing_tokens": True,
                    "padding": True,
                    "stride": stride,
                    "truncation": True,
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
        Classify the text(s) given as inputs.

        Args:
            inputs (`str` or `List[str]`):
                One or several texts to classify. In order to use text pairs for your classification, you can send a
                dictionary containing `{"text", "text_pair"}` keys, or a list of those.
            top_k (`int`, *optional*, defaults to `1`):
                How many results to return.
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                The function to apply to the model outputs in order to retrieve the scores. Accepts four different
                values:

                If this argument is not specified, then it will apply the following functions according to the number
                of labels:

                - If the model has a single label, will apply the sigmoid function on the output.
                - If the model has several labels, will apply the softmax function on the output.

                Possible values are:

                - `"sigmoid"`: Applies the sigmoid function on the output.
                - `"softmax"`: Applies the softmax function on the output.
                - `"none"`: Does not apply any function on the output.

        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:

            - **label** (`str`) -- The label predicted.
            - **score** (`float`) -- The corresponding probability.

            If `top_k` is used, one such dictionary is returned per label.
        """
        _inputs, offset_mapping = self._args_parser(inputs, **kwargs)
        if offset_mapping:
            kwargs["offset_mapping"] = offset_mapping

        result = super().__call__(inputs, **kwargs)
        if isinstance(result, dict):
            result = [result]
        if kwargs.get("return_all_scores", False) and isinstance(result[0], dict):
            result = [result]
        return result

    def preprocess(self, sentence, offset_mapping=None, **preprocess_params) -> Iterator[Dict[str, GenericTensor]]:
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        if isinstance(sentence, dict):
            inputs = self.tokenizer(
                sentence,
                return_tensors=self.framework,
                **tokenizer_params,
            )
        elif (
            isinstance(sentence, list)
            and len(sentence) == 1
            and isinstance(sentence[0], list)
            and len(sentence[0]) == 2
        ):
            # It used to be valid to use a list of list of list for text pairs, keeping this path for BC
            inputs = self.tokenizer(
                text=sentence[0][0], text_pair=sentence[0][1], return_tensors=self.framework, **tokenizer_params
            )
        elif isinstance(sentence, list):
            # This is likely an invalid usage of the pipeline attempting to pass text pairs.
            raise ValueError(
                "The pipeline received invalid inputs, if you are trying to send text pairs, you can try to send a"
                ' dictionary `{"text": "My text", "text_pair": "My pair"}` in order to send a text pair.'
            )
        else:
            inputs = self.tokenizer(
                sentence,
                return_tensors=self.framework,
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
            model_inputs["is_last"] = i == num_chunks - 1

            yield model_inputs

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        if self.framework == "tf":
            logits = self.model(**model_inputs)[0]
        else:
            output = self.model(**model_inputs)
            logits = output["logits"] if isinstance(output, dict) else output[0]
        return {"logits": logits, "is_last": is_last, **model_inputs}

    def postprocess(self, all_outputs, function_to_apply=None, top_k=1, _legacy=True):
        # `_legacy` is used to determine if we're running the naked pipeline and in backward
        # compatibility mode, or if running the pipeline with `pipeline(..., top_k=1)` we're running
        # the more natural result containing the list.
        # Default value before `set_parameters`
        if function_to_apply is None:
            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply
            else:
                function_to_apply = ClassificationFunction.NONE

        all_logits = []
        for model_outputs in all_outputs:
            outputs = model_outputs["logits"][0]
            all_logits.append(outputs.numpy())
        mean_logits = np.mean(all_logits, axis=0)

        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(mean_logits)
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(mean_logits)
        elif function_to_apply == ClassificationFunction.NONE:
            scores = mean_logits
        else:
            raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")

        if top_k == 1 and _legacy:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores.max().item()}

        dict_scores = [
            {"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)
        ]
        if not _legacy:
            dict_scores.sort(key=lambda x: x["score"], reverse=True)
            if top_k is not None:
                dict_scores = dict_scores[:top_k]
        return dict_scores
