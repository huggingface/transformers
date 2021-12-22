from functools import partial, reduce
from typing import Callable, Dict, Optional, Tuple, Type

from .. import PretrainedConfig, is_torch_available
from ..models.albert import AlbertOnnxConfig
from ..models.bart import BartOnnxConfig
from ..models.bert import BertOnnxConfig
from ..models.camembert import CamembertOnnxConfig
from ..models.distilbert import DistilBertOnnxConfig
from ..models.gpt2 import GPT2OnnxConfig
from ..models.gpt_neo import GPTNeoOnnxConfig
from ..models.layoutlm import LayoutLMOnnxConfig
from ..models.longformer import LongformerOnnxConfig
from ..models.mbart import MBartOnnxConfig
from ..models.roberta import RobertaOnnxConfig
from ..models.t5 import T5OnnxConfig
from ..models.xlm_roberta import XLMRobertaOnnxConfig
from ..utils import logging
from .config import OnnxConfig


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_available():
    from transformers import PreTrainedModel
    from transformers.models.auto import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )
else:
    logger.warning(
        "The ONNX export features are only supported for PyTorch, you will not be able to export models without it."
    )


def supported_features_mapping(
    *supported_features: str, onnx_config_cls: Type[OnnxConfig] = None
) -> Dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
    """
    Generate the mapping between supported the features and their corresponding OnnxConfig for a given model.

    Args:
        *supported_features: The names of the supported features.
        onnx_config_cls: The OnnxConfig class corresponding to the model.

    Returns:
        The dictionary mapping a feature to an OnnxConfig constructor.
    """
    if onnx_config_cls is None:
        raise ValueError("A OnnxConfig class must be provided")

    mapping = {}
    for feature in supported_features:
        if "-with-past" in feature:
            task = feature.replace("-with-past", "")
            mapping[feature] = partial(onnx_config_cls.with_past, task=task)
        else:
            mapping[feature] = partial(onnx_config_cls.from_model_config, task=feature)

    return mapping


class FeaturesManager:
    if is_torch_available():
        _TASKS_TO_AUTOMODELS = {
            "default": AutoModel,
            "masked-lm": AutoModelForMaskedLM,
            "causal-lm": AutoModelForCausalLM,
            "seq2seq-lm": AutoModelForSeq2SeqLM,
            "sequence-classification": AutoModelForSequenceClassification,
            "token-classification": AutoModelForTokenClassification,
            "multiple-choice": AutoModelForMultipleChoice,
            "question-answering": AutoModelForQuestionAnswering,
        }
    else:
        _TASKS_TO_AUTOMODELS = {}

    # Set of model topologies we support associated to the features supported by each topology and the factory
    _SUPPORTED_MODEL_TYPE = {
        "albert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            # "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=AlbertOnnxConfig,
        ),
        "bart": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            "sequence-classification",
            "question-answering",
            onnx_config_cls=BartOnnxConfig,
        ),
        "mbart": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            "sequence-classification",
            "question-answering",
            onnx_config_cls=MBartOnnxConfig,
        ),
        "bert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            # "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=BertOnnxConfig,
        ),
        "camembert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            # "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=CamembertOnnxConfig,
        ),
        "distilbert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            # "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=DistilBertOnnxConfig,
        ),
        "longformer": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            # "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=LongformerOnnxConfig,
        ),
        "roberta": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            # "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=RobertaOnnxConfig,
        ),
        "t5": supported_features_mapping(
            "default", "default-with-past", "seq2seq-lm", "seq2seq-lm-with-past", onnx_config_cls=T5OnnxConfig
        ),
        "xlm-roberta": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            # "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=XLMRobertaOnnxConfig,
        ),
        "gpt2": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            "token-classification",
            onnx_config_cls=GPT2OnnxConfig,
        ),
        "gpt-neo": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            onnx_config_cls=GPTNeoOnnxConfig,
        ),
        "layoutlm": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "token-classification",
            onnx_config_cls=LayoutLMOnnxConfig,
        ),
    }

    AVAILABLE_FEATURES = sorted(reduce(lambda s1, s2: s1 | s2, (v.keys() for v in _SUPPORTED_MODEL_TYPE.values())))

    @staticmethod
    def get_supported_features_for_model_type(
        model_type: str, model_name: Optional[str] = None
    ) -> Dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
        """
        Try to retrieve the feature -> OnnxConfig constructor map from the model type.

        Args:
            model_type: The model type to retrieve the supported features for.
            model_name: The name attribute of the model object, only used for the exception message.

        Returns:
            The dictionary mapping each feature to a corresponding OnnxConfig constructor.
        """
        model_type = model_type.lower()
        if model_type not in FeaturesManager._SUPPORTED_MODEL_TYPE:
            model_type_and_model_name = f"{model_type} ({model_name})" if model_name else model_type
            raise KeyError(
                f"{model_type_and_model_name} is not supported yet. "
                f"Only {list(FeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type]

    @staticmethod
    def feature_to_task(feature: str) -> str:
        return feature.replace("-with-past", "")

    @staticmethod
    def get_model_class_for_feature(feature: str) -> Type:
        """
        Attempt to retrieve an AutoModel class from a feature name.

        Args:
            feature: The feature required.

        Returns:
            The AutoModel class corresponding to the feature.
        """
        task = FeaturesManager.feature_to_task(feature)
        if task not in FeaturesManager._TASKS_TO_AUTOMODELS:
            raise KeyError(
                f"Unknown task: {feature}. "
                f"Possible values are {list(FeaturesManager._TASKS_TO_AUTOMODELS.values())}"
            )
        return FeaturesManager._TASKS_TO_AUTOMODELS[task]

    def get_model_from_feature(feature: str, model: str) -> PreTrainedModel:
        """
        Attempt to retrieve a model from a model's name and the feature to be enabled.

        Args:
            feature: The feature required.
            model: The name of the model to export.

        Returns:
            The instance of the model.

        """
        model_class = FeaturesManager.get_model_class_for_feature(feature)
        return model_class.from_pretrained(model)

    @staticmethod
    def check_supported_model_or_raise(model: PreTrainedModel, feature: str = "default") -> Tuple[str, Callable]:
        """
        Check whether or not the model has the requested features.

        Args:
            model: The model to export.
            feature: The name of the feature to check if it is available.

        Returns:
            (str) The type of the model (OnnxConfig) The OnnxConfig instance holding the model export properties.

        """
        model_type = model.config.model_type.replace("_", "-")
        model_name = getattr(model, "name", "")
        model_features = FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model_name)
        if feature not in model_features:
            raise ValueError(
                f"{model.config.model_type} doesn't support feature {feature}. "
                f"Supported values are: {model_features}"
            )

        return model.config.model_type, FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]
