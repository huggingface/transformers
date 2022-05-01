from functools import partial, reduce
from typing import Callable, Dict, Optional, Tuple, Type, Union

from .. import PretrainedConfig, PreTrainedModel, TFPreTrainedModel, is_tf_available, is_torch_available
from ..models.albert import AlbertOnnxConfig
from ..models.bart import BartOnnxConfig
from ..models.beit import BeitOnnxConfig
from ..models.bert import BertOnnxConfig
from ..models.big_bird import BigBirdOnnxConfig
from ..models.blenderbot import BlenderbotOnnxConfig
from ..models.blenderbot_small import BlenderbotSmallOnnxConfig
from ..models.camembert import CamembertOnnxConfig
from ..models.convbert import ConvBertOnnxConfig
from ..models.data2vec import Data2VecTextOnnxConfig
from ..models.deit import DeiTOnnxConfig
from ..models.distilbert import DistilBertOnnxConfig
from ..models.electra import ElectraOnnxConfig
from ..models.flaubert import FlaubertOnnxConfig
from ..models.gpt2 import GPT2OnnxConfig
from ..models.gpt_neo import GPTNeoOnnxConfig
from ..models.gptj import GPTJOnnxConfig
from ..models.ibert import IBertOnnxConfig
from ..models.layoutlm import LayoutLMOnnxConfig
from ..models.m2m_100 import M2M100OnnxConfig
from ..models.marian import MarianOnnxConfig
from ..models.mbart import MBartOnnxConfig
from ..models.roberta import RobertaOnnxConfig
from ..models.roformer import RoFormerOnnxConfig
from ..models.t5 import T5OnnxConfig
from ..models.vit import ViTOnnxConfig
from ..models.xlm_roberta import XLMRobertaOnnxConfig
from ..utils import logging
from .config import OnnxConfig


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_available():
    from transformers.models.auto import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoModelForMaskedImageModeling,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )
if is_tf_available():
    from transformers.models.auto import (
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForMultipleChoice,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTokenClassification,
    )
if not is_torch_available() and not is_tf_available():
    logger.warning(
        "The ONNX export features are only supported for PyTorch or TensorFlow. You will not be able to export models without one of these libraries installed."
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
    _TASKS_TO_AUTOMODELS = {}
    _TASKS_TO_TF_AUTOMODELS = {}
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
            "image-classification": AutoModelForImageClassification,
            "masked-im": AutoModelForMaskedImageModeling,
        }
    if is_tf_available():
        _TASKS_TO_TF_AUTOMODELS = {
            "default": TFAutoModel,
            "masked-lm": TFAutoModelForMaskedLM,
            "causal-lm": TFAutoModelForCausalLM,
            "seq2seq-lm": TFAutoModelForSeq2SeqLM,
            "sequence-classification": TFAutoModelForSequenceClassification,
            "token-classification": TFAutoModelForTokenClassification,
            "multiple-choice": TFAutoModelForMultipleChoice,
            "question-answering": TFAutoModelForQuestionAnswering,
        }

    # Set of model topologies we support associated to the features supported by each topology and the factory
    _SUPPORTED_MODEL_TYPE = {
        "albert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
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
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=BertOnnxConfig,
        ),
        "big-bird": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=BigBirdOnnxConfig,
        ),
        "ibert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=IBertOnnxConfig,
        ),
        "camembert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=CamembertOnnxConfig,
        ),
        "convbert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=ConvBertOnnxConfig,
        ),
        "distilbert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=DistilBertOnnxConfig,
        ),
        "flaubert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=FlaubertOnnxConfig,
        ),
        "marian": supported_features_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            "causal-lm",
            "causal-lm-with-past",
            onnx_config_cls=MarianOnnxConfig,
        ),
        "m2m-100": supported_features_mapping(
            "default", "default-with-past", "seq2seq-lm", "seq2seq-lm-with-past", onnx_config_cls=M2M100OnnxConfig
        ),
        "roberta": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
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
            "multiple-choice",
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
        "gptj": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "question-answering",
            "sequence-classification",
            onnx_config_cls=GPTJOnnxConfig,
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
        "electra": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=ElectraOnnxConfig,
        ),
        "vit": supported_features_mapping(
            "default", "image-classification", "masked-im", onnx_config_cls=ViTOnnxConfig
        ),
        "beit": supported_features_mapping(
            "default", "image-classification", "masked-im", onnx_config_cls=BeitOnnxConfig
        ),
        "deit": supported_features_mapping(
            "default", "image-classification", "masked-im", onnx_config_cls=DeiTOnnxConfig
        ),
        "blenderbot": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx_config_cls=BlenderbotOnnxConfig,
        ),
        "blenderbot-small": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx_config_cls=BlenderbotSmallOnnxConfig,
        ),
        "data2vec-text": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls=Data2VecTextOnnxConfig,
        ),
        "roformer": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "token-classification",
            "multiple-choice",
            "question-answering",
            "token-classification",
            onnx_config_cls=RoFormerOnnxConfig,
        ),
    }

    AVAILABLE_FEATURES = sorted(reduce(lambda s1, s2: s1 | s2, (v.keys() for v in _SUPPORTED_MODEL_TYPE.values())))

    @staticmethod
    def get_supported_features_for_model_type(
        model_type: str, model_name: Optional[str] = None
    ) -> Dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
        """
        Tries to retrieve the feature -> OnnxConfig constructor map from the model type.

        Args:
            model_type (`str`):
                The model type to retrieve the supported features for.
            model_name (`str`, *optional*):
                The name attribute of the model object, only used for the exception message.

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
    def _validate_framework_choice(framework: str):
        """
        Validates if the framework requested for the export is both correct and available, otherwise throws an
        exception.
        """
        if framework not in ["pt", "tf"]:
            raise ValueError(
                f"Only two frameworks are supported for ONNX export: pt or tf, but {framework} was provided."
            )
        elif framework == "pt" and not is_torch_available():
            raise RuntimeError("Cannot export model to ONNX using PyTorch because no PyTorch package was found.")
        elif framework == "tf" and not is_tf_available():
            raise RuntimeError("Cannot export model to ONNX using TensorFlow because no TensorFlow package was found.")

    @staticmethod
    def get_model_class_for_feature(feature: str, framework: str = "pt") -> Type:
        """
        Attempts to retrieve an AutoModel class from a feature name.

        Args:
            feature (`str`):
                The feature required.
            framework (`str`, *optional*, defaults to `"pt"`):
                The framework to use for the export.

        Returns:
            The AutoModel class corresponding to the feature.
        """
        task = FeaturesManager.feature_to_task(feature)
        FeaturesManager._validate_framework_choice(framework)
        if framework == "pt":
            task_to_automodel = FeaturesManager._TASKS_TO_AUTOMODELS
        else:
            task_to_automodel = FeaturesManager._TASKS_TO_TF_AUTOMODELS
        if task not in task_to_automodel:
            raise KeyError(
                f"Unknown task: {feature}. "
                f"Possible values are {list(FeaturesManager._TASKS_TO_AUTOMODELS.values())}"
            )
        return task_to_automodel[task]

    @staticmethod
    def get_model_from_feature(
        feature: str, model: str, framework: str = "pt", cache_dir: str = None
    ) -> Union[PreTrainedModel, TFPreTrainedModel]:
        """
        Attempts to retrieve a model from a model's name and the feature to be enabled.

        Args:
            feature (`str`):
                The feature required.
            model (`str`):
                The name of the model to export.
            framework (`str`, *optional*, defaults to `"pt"`):
                The framework to use for the export.

        Returns:
            The instance of the model.

        """
        model_class = FeaturesManager.get_model_class_for_feature(feature, framework)
        try:
            model = model_class.from_pretrained(model, cache_dir=cache_dir)
        except OSError:
            if framework == "pt":
                model = model_class.from_pretrained(model, from_tf=True, cache_dir=cache_dir)
            else:
                model = model_class.from_pretrained(model, from_pt=True, cache_dir=cache_dir)
        return model

    @staticmethod
    def check_supported_model_or_raise(
        model: Union[PreTrainedModel, TFPreTrainedModel], feature: str = "default"
    ) -> Tuple[str, Callable]:
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
