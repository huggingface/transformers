from collections import OrderedDict
from functools import partial
from typing import Callable, Tuple

# from ..models.albert import AlbertOnnxConfig
import transformers.models.albert
from ..models.bart import BartOnnxConfig
# from ..models.bert import BertOnnxConfig
import transformers.models.bert.BertOnnxConfig as BertOnnxConfig
from ..models.distilbert import DistilBertOnnxConfig
from ..models.gpt2 import GPT2OnnxConfig
from ..models.longformer import LongformerOnnxConfig
from ..models.roberta import RobertaOnnxConfig
from ..models.t5 import T5OnnxConfig
from ..models.xlm_roberta import XLMRobertaOnnxConfig
from ..models.gpt_neo import GPTNeoOnnxConfig

from .. import is_torch_available

if is_torch_available():
    from transformers import PreTrainedModel
    from transformers.models.auto import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForMultipleChoice,
        AutoModelForQuestionAnswering,
    )


def supported_features_mapping(*supported_features, onnx_config_cls=None):
    """Generates the mapping between supported features and their corresponding OnnxConfig."""
    if onnx_config_cls is None:
        raise ValueError("A OnnxConfig class must be provided")

    mapping = {}
    for feature in supported_features:
        if "-with-past" in feature:
            task = feature.replace("-with-past", "")
            mapping[task] = partial(onnx_config_cls.with_past, task=task)
        else:
            mapping[feature] = partial(onnx_config_cls.from_model_config, task=feature)

    return mapping


class FeaturesManager:
    _TASKS_TO_COMMON_OUTPUTS = {
        "default": OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}}),
        "causal-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "sequence-classification": OrderedDict({"logits": {0: "batch"}}),
        "token-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "multiple-choice": OrderedDict({"logits": {0: "batch"}}),
        "question-answering": OrderedDict({
            "start_logits": {0: "batch", 1: "sequence"},
            "end_logits": {0: "batch", 1: "sequence"},
        }),
    }

    _TASKS_TO_AUTOMODELS = {
        "default": AutoModel,
        "causal-lm": AutoModelForCausalLM,
        "sequence-classification": AutoModelForSequenceClassification,
        "token-classification": AutoModelForTokenClassification,
        "multiple-choice": AutoModelForMultipleChoice,
        "question-answering": AutoModelForQuestionAnswering,
    }

    # Set of model topologies we support associated to the features supported by each topology and the factory
    _SUPPORTED_MODEL_KIND = {
        "albert": supported_features_mapping("default", onnx_config_cls=AlbertOnnxConfig),
        "bart": supported_features_mapping("default", onnx_config_cls=BartOnnxConfig),
        "bert": supported_features_mapping("default", onnx_config_cls=BertOnnxConfig),
        "distilbert": supported_features_mapping("default", onnx_config_cls=DistilBertOnnxConfig),
        "gpt2": supported_features_mapping("default", onnx_config_cls=GPT2OnnxConfig),
        "longformer": supported_features_mapping("default", onnx_config_cls=LongformerOnnxConfig),
        "roberta": supported_features_mapping("default", onnx_config_cls=RobertaOnnxConfig),
        "t5": supported_features_mapping("default", onnx_config_cls=T5OnnxConfig),
        "xlm-roberta": supported_features_mapping("default", onnx_config_cls=XLMRobertaOnnxConfig),
        "gpt-neo": supported_features_mapping(
            "default",
            "causal-lm",
            "sequence-classification",
            "default-with-past",
            "causal-lm-with-past",
            "sequence-classification-with-past",
            onnx_config_cls=GPTNeoOnnxConfig
        ),
    }

    @staticmethod
    def feature_to_task(feature: str) -> str:
        return feature.replace("-with-past", "")

    @staticmethod
    def get_model_from_feature(feature: str, model: str):
        """
        Attempt to retrieve a model from a model's name and the feature to be enabled.

        Args:
            feature: The feature required
            model: The name of the model to export

        Returns:

        """
        task = FeaturesManager.feature_to_task(feature)
        if task not in FeaturesManager._TASKS_TO_AUTOMODELS:
            raise KeyError(f"Unknown task: {feature}." f"Possible values are {list(FeaturesManager._TASKS_TO_AUTOMODELS.values())}")

        return FeaturesManager._TASKS_TO_AUTOMODELS[feature].from_pretrained(model)

    @staticmethod
    def check_supported_model_or_raise(model: PreTrainedModel, feature: str = "default") -> Tuple[str, Callable]:
        """
        Check whether or not the model has the requested features

        Args:
            model: The model to export
            feature: The name of the feature to check if it is avaiable

        Returns:
            (str) The type of the model (OnnxConfig) The OnnxConfig instance holding the model export properties

        """
        if model.config.model_type not in FeaturesManager._SUPPORTED_MODEL_KIND:
            raise KeyError(
                f"{model.config.model_type} ({model.name}) is not supported yet. "
                f"Only {FeaturesManager._SUPPORTED_MODEL_KIND} are supported. "
                f"If you want to support ({model.config.model_type}) please propose a PR or open up an issue."
            )

        # Look for the features
        model_features = FeaturesManager._SUPPORTED_MODEL_KIND[model.config.model_type]
        if feature not in model_features:
            raise ValueError(
                f"{model.config.model_type} doesn't support feature {feature}. "
                f"Supported values are: {list(model_features.keys())}"
            )

        return model.config.model_type, FeaturesManager._SUPPORTED_MODEL_KIND[model.config.model_type][feature]
