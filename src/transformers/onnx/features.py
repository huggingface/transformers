from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Optional

import transformers

from .. import PretrainedConfig, is_torch_available
from ..utils import logging
from .config import OnnxConfig


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_available():
    from transformers.models.auto import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForMaskedImageModeling,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        AutoModelForSemanticSegmentation,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,
        AutoModelForTokenClassification,
        AutoModelForVision2Seq,
    )
else:
    logger.warning(
        "The ONNX export features is only supported for PyTorch. You will not be able to export models without it installed."
    )


def supported_features_mapping(
    *supported_features: str, onnx_config_cls: Optional[str] = None
) -> dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
    """
    Generate the mapping between supported the features and their corresponding OnnxConfig for a given model.

    Args:
        *supported_features: The names of the supported features.
        onnx_config_cls: The OnnxConfig full name corresponding to the model.

    Returns:
        The dictionary mapping a feature to an OnnxConfig constructor.
    """
    if onnx_config_cls is None:
        raise ValueError("A OnnxConfig class must be provided")

    config_cls = transformers
    for attr_name in onnx_config_cls.split("."):
        config_cls = getattr(config_cls, attr_name)
    mapping = {}
    for feature in supported_features:
        if "-with-past" in feature:
            task = feature.replace("-with-past", "")
            mapping[feature] = partial(config_cls.with_past, task=task)
        else:
            mapping[feature] = partial(config_cls.from_model_config, task=feature)

    return mapping


class FeaturesManager:
    _TASKS_TO_AUTOMODELS = {}
    if is_torch_available():
        _TASKS_TO_AUTOMODELS = {
            "default": AutoModel,
            "masked-lm": AutoModelForMaskedLM,
            "causal-lm": AutoModelForCausalLM,
            "seq2seq-lm": AutoModelForSeq2SeqLM,
            "sequence-classification": AutoModelForSequenceClassification,
            "token-classification": AutoModelForTokenClassification,
            "multiple-choice": AutoModelForMultipleChoice,
            "object-detection": AutoModelForObjectDetection,
            "question-answering": AutoModelForQuestionAnswering,
            "image-classification": AutoModelForImageClassification,
            "image-segmentation": AutoModelForImageSegmentation,
            "masked-im": AutoModelForMaskedImageModeling,
            "semantic-segmentation": AutoModelForSemanticSegmentation,
            "vision2seq-lm": AutoModelForVision2Seq,
            "speech2seq-lm": AutoModelForSpeechSeq2Seq,
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
            onnx_config_cls="models.albert.AlbertOnnxConfig",
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
            onnx_config_cls="models.bart.BartOnnxConfig",
        ),
        # BEiT cannot be used with the masked image modeling autoclass, so this feature is excluded here
        "beit": supported_features_mapping(
            "default", "image-classification", onnx_config_cls="models.beit.BeitOnnxConfig"
        ),
        "bert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.bert.BertOnnxConfig",
        ),
        "big-bird": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.big_bird.BigBirdOnnxConfig",
        ),
        "bigbird-pegasus": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            "sequence-classification",
            "question-answering",
            onnx_config_cls="models.bigbird_pegasus.BigBirdPegasusOnnxConfig",
        ),
        "blenderbot": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx_config_cls="models.blenderbot.BlenderbotOnnxConfig",
        ),
        "blenderbot-small": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx_config_cls="models.blenderbot_small.BlenderbotSmallOnnxConfig",
        ),
        "bloom": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            "token-classification",
            onnx_config_cls="models.bloom.BloomOnnxConfig",
        ),
        "camembert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.camembert.CamembertOnnxConfig",
        ),
        "clip": supported_features_mapping(
            "default",
            onnx_config_cls="models.clip.CLIPOnnxConfig",
        ),
        "codegen": supported_features_mapping(
            "default",
            "causal-lm",
            onnx_config_cls="models.codegen.CodeGenOnnxConfig",
        ),
        "convbert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.convbert.ConvBertOnnxConfig",
        ),
        "convnext": supported_features_mapping(
            "default",
            "image-classification",
            onnx_config_cls="models.convnext.ConvNextOnnxConfig",
        ),
        "data2vec-text": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.data2vec.Data2VecTextOnnxConfig",
        ),
        "data2vec-vision": supported_features_mapping(
            "default",
            "image-classification",
            # ONNX doesn't support `adaptive_avg_pool2d` yet
            # "semantic-segmentation",
            onnx_config_cls="models.data2vec.Data2VecVisionOnnxConfig",
        ),
        "deberta": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.deberta.DebertaOnnxConfig",
        ),
        "deberta-v2": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.deberta_v2.DebertaV2OnnxConfig",
        ),
        "deit": supported_features_mapping(
            "default", "image-classification", onnx_config_cls="models.deit.DeiTOnnxConfig"
        ),
        "detr": supported_features_mapping(
            "default",
            "object-detection",
            "image-segmentation",
            onnx_config_cls="models.detr.DetrOnnxConfig",
        ),
        "distilbert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.distilbert.DistilBertOnnxConfig",
        ),
        "electra": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.electra.ElectraOnnxConfig",
        ),
        "flaubert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.flaubert.FlaubertOnnxConfig",
        ),
        "gpt2": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            "token-classification",
            onnx_config_cls="models.gpt2.GPT2OnnxConfig",
        ),
        "gptj": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "question-answering",
            "sequence-classification",
            onnx_config_cls="models.gptj.GPTJOnnxConfig",
        ),
        "gpt-neo": supported_features_mapping(
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            onnx_config_cls="models.gpt_neo.GPTNeoOnnxConfig",
        ),
        "groupvit": supported_features_mapping(
            "default",
            onnx_config_cls="models.groupvit.GroupViTOnnxConfig",
        ),
        "ibert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.ibert.IBertOnnxConfig",
        ),
        "imagegpt": supported_features_mapping(
            "default", "image-classification", onnx_config_cls="models.imagegpt.ImageGPTOnnxConfig"
        ),
        "layoutlm": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "token-classification",
            onnx_config_cls="models.layoutlm.LayoutLMOnnxConfig",
        ),
        "layoutlmv3": supported_features_mapping(
            "default",
            "question-answering",
            "sequence-classification",
            "token-classification",
            onnx_config_cls="models.layoutlmv3.LayoutLMv3OnnxConfig",
        ),
        "levit": supported_features_mapping(
            "default", "image-classification", onnx_config_cls="models.levit.LevitOnnxConfig"
        ),
        "longt5": supported_features_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx_config_cls="models.longt5.LongT5OnnxConfig",
        ),
        "longformer": supported_features_mapping(
            "default",
            "masked-lm",
            "multiple-choice",
            "question-answering",
            "sequence-classification",
            "token-classification",
            onnx_config_cls="models.longformer.LongformerOnnxConfig",
        ),
        "marian": supported_features_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            "causal-lm",
            "causal-lm-with-past",
            onnx_config_cls="models.marian.MarianOnnxConfig",
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
            onnx_config_cls="models.mbart.MBartOnnxConfig",
        ),
        "mobilebert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.mobilebert.MobileBertOnnxConfig",
        ),
        "mobilenet-v1": supported_features_mapping(
            "default",
            "image-classification",
            onnx_config_cls="models.mobilenet_v1.MobileNetV1OnnxConfig",
        ),
        "mobilenet-v2": supported_features_mapping(
            "default",
            "image-classification",
            onnx_config_cls="models.mobilenet_v2.MobileNetV2OnnxConfig",
        ),
        "mobilevit": supported_features_mapping(
            "default",
            "image-classification",
            onnx_config_cls="models.mobilevit.MobileViTOnnxConfig",
        ),
        "mt5": supported_features_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx_config_cls="models.mt5.MT5OnnxConfig",
        ),
        "m2m-100": supported_features_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx_config_cls="models.m2m_100.M2M100OnnxConfig",
        ),
        "owlvit": supported_features_mapping(
            "default",
            onnx_config_cls="models.owlvit.OwlViTOnnxConfig",
        ),
        "perceiver": supported_features_mapping(
            "image-classification",
            "masked-lm",
            "sequence-classification",
            onnx_config_cls="models.perceiver.PerceiverOnnxConfig",
        ),
        "poolformer": supported_features_mapping(
            "default", "image-classification", onnx_config_cls="models.poolformer.PoolFormerOnnxConfig"
        ),
        "rembert": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.rembert.RemBertOnnxConfig",
        ),
        "resnet": supported_features_mapping(
            "default",
            "image-classification",
            onnx_config_cls="models.resnet.ResNetOnnxConfig",
        ),
        "roberta": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.roberta.RobertaOnnxConfig",
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
            onnx_config_cls="models.roformer.RoFormerOnnxConfig",
        ),
        "segformer": supported_features_mapping(
            "default",
            "image-classification",
            "semantic-segmentation",
            onnx_config_cls="models.segformer.SegformerOnnxConfig",
        ),
        "squeezebert": supported_features_mapping(
            "default",
            "masked-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.squeezebert.SqueezeBertOnnxConfig",
        ),
        "swin": supported_features_mapping(
            "default", "image-classification", onnx_config_cls="models.swin.SwinOnnxConfig"
        ),
        "t5": supported_features_mapping(
            "default",
            "default-with-past",
            "seq2seq-lm",
            "seq2seq-lm-with-past",
            onnx_config_cls="models.t5.T5OnnxConfig",
        ),
        "vision-encoder-decoder": supported_features_mapping(
            "vision2seq-lm", onnx_config_cls="models.vision_encoder_decoder.VisionEncoderDecoderOnnxConfig"
        ),
        "vit": supported_features_mapping(
            "default", "image-classification", onnx_config_cls="models.vit.ViTOnnxConfig"
        ),
        "whisper": supported_features_mapping(
            "default",
            "default-with-past",
            "speech2seq-lm",
            "speech2seq-lm-with-past",
            onnx_config_cls="models.whisper.WhisperOnnxConfig",
        ),
        "xlm": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.xlm.XLMOnnxConfig",
        ),
        "xlm-roberta": supported_features_mapping(
            "default",
            "masked-lm",
            "causal-lm",
            "sequence-classification",
            "multiple-choice",
            "token-classification",
            "question-answering",
            onnx_config_cls="models.xlm_roberta.XLMRobertaOnnxConfig",
        ),
        "yolos": supported_features_mapping(
            "default",
            "object-detection",
            onnx_config_cls="models.yolos.YolosOnnxConfig",
        ),
    }

    AVAILABLE_FEATURES = sorted(reduce(lambda s1, s2: s1 | s2, (v.keys() for v in _SUPPORTED_MODEL_TYPE.values())))

    @staticmethod
    def get_supported_features_for_model_type(
        model_type: str, model_name: Optional[str] = None
    ) -> dict[str, Callable[[PretrainedConfig], OnnxConfig]]:
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
    def get_model_class_for_feature(feature: str) -> type:
        """
        Attempts to retrieve an AutoModel class from a feature name.

        Args:
            feature (`str`):
                The feature required.

        Returns:
            The AutoModel class corresponding to the feature.
        """
        task = FeaturesManager.feature_to_task(feature)
        task_to_automodel = FeaturesManager._TASKS_TO_AUTOMODELS
        if task not in task_to_automodel:
            raise KeyError(
                f"Unknown task: {feature}. Possible values are {list(FeaturesManager._TASKS_TO_AUTOMODELS.values())}"
            )

        return task_to_automodel[task]

    @staticmethod
    def get_model_from_feature(feature: str, model: str, cache_dir: Optional[str] = None) -> "PreTrainedModel":
        """
        Attempts to retrieve a model from a model's name and the feature to be enabled.

        Args:
            feature (`str`):
                The feature required.
            model (`str`):
                The name of the model to export.

        Returns:
            The instance of the model.

        """
        model_class = FeaturesManager.get_model_class_for_feature(feature)
        model = model_class.from_pretrained(model, cache_dir=cache_dir)
        return model

    @staticmethod
    def check_supported_model_or_raise(model: "PreTrainedModel", feature: str = "default") -> tuple[str, Callable]:
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
                f"{model.config.model_type} doesn't support feature {feature}. Supported values are: {model_features}"
            )

        return model.config.model_type, FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]

    def get_config(model_type: str, feature: str) -> OnnxConfig:
        """
        Gets the OnnxConfig for a model_type and feature combination.

        Args:
            model_type (`str`):
                The model type to retrieve the config for.
            feature (`str`):
                The feature to retrieve the config for.

        Returns:
            `OnnxConfig`: config for the combination
        """
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]
