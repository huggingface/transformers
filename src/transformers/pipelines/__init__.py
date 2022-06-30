# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

import io
import json
import os

# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from ..configuration_utils import PretrainedConfig
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..models.auto.configuration_auto import AutoConfig
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from ..tokenization_utils import PreTrainedTokenizer
from ..tokenization_utils_fast import PreTrainedTokenizerFast
from ..utils import HUGGINGFACE_CO_RESOLVE_ENDPOINT, http_get, is_tf_available, is_torch_available, logging
from .audio_classification import AudioClassificationPipeline
from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from .base import (
    ArgumentHandler,
    CsvPipelineDataFormat,
    JsonPipelineDataFormat,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    PipelineException,
    PipelineRegistry,
    get_default_model_and_revision,
    infer_framework_load_model,
)
from .conversational import Conversation, ConversationalPipeline
from .feature_extraction import FeatureExtractionPipeline
from .fill_mask import FillMaskPipeline
from .image_classification import ImageClassificationPipeline
from .image_segmentation import ImageSegmentationPipeline
from .object_detection import ObjectDetectionPipeline
from .question_answering import QuestionAnsweringArgumentHandler, QuestionAnsweringPipeline
from .table_question_answering import TableQuestionAnsweringArgumentHandler, TableQuestionAnsweringPipeline
from .text2text_generation import SummarizationPipeline, Text2TextGenerationPipeline, TranslationPipeline
from .text_classification import TextClassificationPipeline
from .text_generation import TextGenerationPipeline
from .token_classification import (
    AggregationStrategy,
    NerPipeline,
    TokenClassificationArgumentHandler,
    TokenClassificationPipeline,
)
from .visual_question_answering import VisualQuestionAnsweringPipeline
from .zero_shot_classification import ZeroShotClassificationArgumentHandler, ZeroShotClassificationPipeline
from .zero_shot_image_classification import ZeroShotImageClassificationPipeline


if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import (
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TF_MODEL_WITH_LM_HEAD_MAPPING,
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTableQuestionAnswering,
        TFAutoModelForTokenClassification,
    )

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import (
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
        AutoModel,
        AutoModelForAudioClassification,
        AutoModelForCausalLM,
        AutoModelForCTC,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForMaskedLM,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        AutoModelForSemanticSegmentation,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,
        AutoModelForTableQuestionAnswering,
        AutoModelForTokenClassification,
        AutoModelForVisualQuestionAnswering,
    )
if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)


# Register all the supported tasks here
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
    "vqa": "visual-question-answering",
}
SUPPORTED_TASKS = {
    "audio-classification": {
        "impl": AudioClassificationPipeline,
        "tf": (),
        "pt": (AutoModelForAudioClassification,) if is_torch_available() else (),
        "default": {"model": {"pt": ("superb/wav2vec2-base-superb-ks", "372e048")}},
        "type": "audio",
    },
    "automatic-speech-recognition": {
        "impl": AutomaticSpeechRecognitionPipeline,
        "tf": (),
        "pt": (AutoModelForCTC, AutoModelForSpeechSeq2Seq) if is_torch_available() else (),
        "default": {"model": {"pt": ("facebook/wav2vec2-base-960h", "55bb623")}},
        "type": "multimodal",
    },
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "tf": (TFAutoModel,) if is_tf_available() else (),
        "pt": (AutoModel,) if is_torch_available() else (),
        "default": {"model": {"pt": ("distilbert-base-cased", "935ac13"), "tf": ("distilbert-base-cased", "935ac13")}},
        "type": "multimodal",
    },
    "text-classification": {
        "impl": TextClassificationPipeline,
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("distilbert-base-uncased-finetuned-sst-2-english", "af0f99b"),
                "tf": ("distilbert-base-uncased-finetuned-sst-2-english", "af0f99b"),
            },
        },
        "type": "text",
    },
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "tf": (TFAutoModelForTokenClassification,) if is_tf_available() else (),
        "pt": (AutoModelForTokenClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("dbmdz/bert-large-cased-finetuned-conll03-english", "f2482bf"),
                "tf": ("dbmdz/bert-large-cased-finetuned-conll03-english", "f2482bf"),
            },
        },
        "type": "text",
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "tf": (TFAutoModelForQuestionAnswering,) if is_tf_available() else (),
        "pt": (AutoModelForQuestionAnswering,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("distilbert-base-cased-distilled-squad", "626af31"),
                "tf": ("distilbert-base-cased-distilled-squad", "626af31"),
            },
        },
        "type": "text",
    },
    "table-question-answering": {
        "impl": TableQuestionAnsweringPipeline,
        "pt": (AutoModelForTableQuestionAnswering,) if is_torch_available() else (),
        "tf": (TFAutoModelForTableQuestionAnswering,) if is_tf_available() else (),
        "default": {
            "model": {
                "pt": ("google/tapas-base-finetuned-wtq", "69ceee2"),
                "tf": ("google/tapas-base-finetuned-wtq", "69ceee2"),
            },
        },
        "type": "text",
    },
    "visual-question-answering": {
        "impl": VisualQuestionAnsweringPipeline,
        "pt": (AutoModelForVisualQuestionAnswering,) if is_torch_available() else (),
        "tf": (),
        "default": {
            "model": {"pt": ("dandelin/vilt-b32-finetuned-vqa", "4355f59")},
        },
        "type": "multimodal",
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "tf": (TFAutoModelForMaskedLM,) if is_tf_available() else (),
        "pt": (AutoModelForMaskedLM,) if is_torch_available() else (),
        "default": {"model": {"pt": ("distilroberta-base", "ec58a5b"), "tf": ("distilroberta-base", "ec58a5b")}},
        "type": "text",
    },
    "summarization": {
        "impl": SummarizationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {"model": {"pt": ("sshleifer/distilbart-cnn-12-6", "a4f8f3e"), "tf": ("t5-small", "d769bba")}},
        "type": "text",
    },
    # This task is a special case as it's parametrized by SRC, TGT languages.
    "translation": {
        "impl": TranslationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {
            ("en", "fr"): {"model": {"pt": ("t5-base", "686f1db"), "tf": ("t5-base", "686f1db")}},
            ("en", "de"): {"model": {"pt": ("t5-base", "686f1db"), "tf": ("t5-base", "686f1db")}},
            ("en", "ro"): {"model": {"pt": ("t5-base", "686f1db"), "tf": ("t5-base", "686f1db")}},
        },
        "type": "text",
    },
    "text2text-generation": {
        "impl": Text2TextGenerationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {"model": {"pt": ("t5-base", "686f1db"), "tf": ("t5-base", "686f1db")}},
        "type": "text",
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "tf": (TFAutoModelForCausalLM,) if is_tf_available() else (),
        "pt": (AutoModelForCausalLM,) if is_torch_available() else (),
        "default": {"model": {"pt": ("gpt2", "6c0e608"), "tf": ("gpt2", "6c0e608")}},
        "type": "text",
    },
    "zero-shot-classification": {
        "impl": ZeroShotClassificationPipeline,
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        "default": {
            "model": {"pt": ("facebook/bart-large-mnli", "c626438"), "tf": ("roberta-large-mnli", "130fb28")},
            "config": {"pt": ("facebook/bart-large-mnli", "c626438"), "tf": ("roberta-large-mnli", "130fb28")},
        },
        "type": "text",
    },
    "zero-shot-image-classification": {
        "impl": ZeroShotImageClassificationPipeline,
        "tf": (TFAutoModel,) if is_tf_available() else (),
        "pt": (AutoModel,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("openai/clip-vit-base-patch32", "f4881ba"),
                "tf": ("openai/clip-vit-base-patch32", "f4881ba"),
            }
        },
        "type": "multimodal",
    },
    "conversational": {
        "impl": ConversationalPipeline,
        "tf": (TFAutoModelForSeq2SeqLM, TFAutoModelForCausalLM) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM, AutoModelForCausalLM) if is_torch_available() else (),
        "default": {
            "model": {"pt": ("microsoft/DialoGPT-medium", "8bada3b"), "tf": ("microsoft/DialoGPT-medium", "8bada3b")}
        },
        "type": "text",
    },
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "tf": (),
        "pt": (AutoModelForImageClassification,) if is_torch_available() else (),
        "default": {"model": {"pt": ("google/vit-base-patch16-224", "5dca96d")}},
        "type": "image",
    },
    "image-segmentation": {
        "impl": ImageSegmentationPipeline,
        "tf": (),
        "pt": (AutoModelForImageSegmentation, AutoModelForSemanticSegmentation) if is_torch_available() else (),
        "default": {"model": {"pt": ("facebook/detr-resnet-50-panoptic", "fc15262")}},
        "type": "image",
    },
    "object-detection": {
        "impl": ObjectDetectionPipeline,
        "tf": (),
        "pt": (AutoModelForObjectDetection,) if is_torch_available() else (),
        "default": {"model": {"pt": ("facebook/detr-resnet-50", "2729413")}},
        "type": "image",
    },
}

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_TOKENIZER_TASKS = set()
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
    elif values["type"] in {"audio", "image"}:
        NO_TOKENIZER_TASKS.add(task)
    elif values["type"] != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values['type']}")

PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)


def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    return PIPELINE_REGISTRY.get_supported_tasks()


def get_task(model: str, use_auth_token: Optional[str] = None) -> str:
    tmp = io.BytesIO()
    headers = {}
    if use_auth_token:
        headers["Authorization"] = f"Bearer {use_auth_token}"

    try:
        http_get(f"https://huggingface.co/api/models/{model}", tmp, headers=headers)
        tmp.seek(0)
        body = tmp.read()
        data = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"Instantiating a pipeline without a task set raised an error: {e}")
    if "pipeline_tag" not in data:
        raise RuntimeError(
            f"The model {model} does not seem to have a correct `pipeline_tag` set to infer the task automatically"
        )
    if data.get("library_name", "transformers") != "transformers":
        raise RuntimeError(f"This model is meant to be used with {data['library_name']} not with transformers")
    task = data["pipeline_tag"]
    return task


def check_task(task: str) -> Tuple[Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`
            - `"automatic-speech-recognition"`
            - `"conversational"`
            - `"feature-extraction"`
            - `"fill-mask"`
            - `"image-classification"`
            - `"question-answering"`
            - `"table-question-answering"`
            - `"text2text-generation"`
            - `"text-classification"` (alias `"sentiment-analysis"` available)
            - `"text-generation"`
            - `"token-classification"` (alias `"ner"` available)
            - `"translation"`
            - `"translation_xx_to_yy"`
            - `"summarization"`
            - `"zero-shot-classification"`
            - `"zero-shot-image-classification"`

    Returns:
        (task_defaults`dict`, task_options: (`tuple`, None)) The actual dictionary required to initialize the pipeline
        and some extra task options for parametrized tasks like "translation_XX_to_YY"


    """
    return PIPELINE_REGISTRY.check_task(task)


def pipeline(
    task: str = None,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs
) -> Pipeline:
    """
    Utility factory method to build a [`Pipeline`].

    Pipelines are made of:

        - A [tokenizer](tokenizer) in charge of mapping raw textual input to token.
        - A [model](model) to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`: will return a [`AudioClassificationPipeline`].
            - `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
            - `"conversational"`: will return a [`ConversationalPipeline`].
            - `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
            - `"fill-mask"`: will return a [`FillMaskPipeline`]:.
            - `"image-classification"`: will return a [`ImageClassificationPipeline`].
            - `"question-answering"`: will return a [`QuestionAnsweringPipeline`].
            - `"table-question-answering"`: will return a [`TableQuestionAnsweringPipeline`].
            - `"text2text-generation"`: will return a [`Text2TextGenerationPipeline`].
            - `"text-classification"` (alias `"sentiment-analysis"` available): will return a
              [`TextClassificationPipeline`].
            - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
            - `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
            - `"translation"`: will return a [`TranslationPipeline`].
            - `"translation_xx_to_yy"`: will return a [`TranslationPipeline`].
            - `"summarization"`: will return a [`SummarizationPipeline`].
            - `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].

        model (`str` or [`PreTrainedModel`] or [`TFPreTrainedModel`], *optional*):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from [`PreTrainedModel`] (for PyTorch) or
            [`TFPreTrainedModel`] (for TensorFlow).

            If not provided, the default for the `task` will be loaded.
        config (`str` or [`PretrainedConfig`], *optional*):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from [`PretrainedConfig`].

            If not provided, the default configuration file for the requested model will be used. That means that if
            `model` is given, its default configuration will be used. However, if `model` is not supplied, this
            `task`'s default model's config is used instead.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].

            If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
            However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
            will be loaded.
        feature_extractor (`str` or [`PreTrainedFeatureExtractor`], *optional*):
            The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained feature extractor inheriting from [`PreTrainedFeatureExtractor`].

            Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
            models. Multi-modal models will also require a tokenizer to be passed.

            If not provided, the default feature extractor for the given `model` will be loaded (if it is a string). If
            `model` is not specified or not a string, then the default feature extractor for `config` is loaded (if it
            is a string). However, if `config` is also not given or not a string, then the default feature extractor
            for the given `task` will be loaded.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        revision (`str`, *optional*, defaults to `"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
        use_fast (`bool`, *optional*, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.
        kwargs:
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        [`Pipeline`]: A suitable pipeline for the task.

    Examples:

    ```python
    >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

    >>> # Sentiment analysis pipeline
    >>> pipeline("sentiment-analysis")

    >>> # Question answering pipeline, specifying the checkpoint identifier
    >>> pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased")

    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
    >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    >>> pipeline("ner", model=model, tokenizer=tokenizer)
    ```"""
    if model_kwargs is None:
        model_kwargs = {}

    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model "
            "being specified. "
            "Please provide a task class or a model"
        )

    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer"
            " may not be compatible with the default model. Please provide a PreTrainedModel class or a"
            " path/identifier to a pretrained model when providing tokenizer."
        )
    if model is None and feature_extractor is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided"
            " feature_extractor may not be compatible with the default model. Please provide a PreTrainedModel class"
            " or a path/identifier to a pretrained model when providing feature_extractor."
        )

    if task is None and model is not None:
        if not isinstance(model, str):
            raise RuntimeError(
                "Inferring the task automatically requires to check the hub with a model_id defined as a `str`."
                f"{model} is not a valid model_id."
            )
        task = get_task(model, use_auth_token)

    # Retrieve the task
    targeted_task, task_options = check_task(task)
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        # At that point framework might still be undetermined
        model, default_revision = get_default_model_and_revision(targeted_task, framework, task_options)
        revision = revision if revision is not None else default_revision
        logger.warning(
            f"No model was supplied, defaulted to {model} and revision"
            f" {revision} ({HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{model}).\n"
            "Using a pipeline without specifying a model name and revision in production is not recommended."
        )

    # Retrieve use_auth_token and add it to model_kwargs to be used in .from_pretrained
    model_kwargs["use_auth_token"] = model_kwargs.get("use_auth_token", use_auth_token)

    # Config is the primordial information item.
    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config, revision=revision, _from_pipeline=task, **model_kwargs)
    elif config is None and isinstance(model, str):
        config = AutoConfig.from_pretrained(model, revision=revision, _from_pipeline=task, **model_kwargs)

    model_name = model if isinstance(model, str) else None

    # Infer the framework from the model
    # Forced if framework already defined, inferred if it's None
    # Will load the correct model if possible
    model_classes = {"tf": targeted_task["tf"], "pt": targeted_task["pt"]}
    framework, model = infer_framework_load_model(
        model,
        model_classes=model_classes,
        config=config,
        framework=framework,
        revision=revision,
        task=task,
        **model_kwargs,
    )

    model_config = model.config

    load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None
    load_feature_extractor = type(model_config) in FEATURE_EXTRACTOR_MAPPING or feature_extractor is not None

    if task in NO_TOKENIZER_TASKS:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False
    if task in NO_FEATURE_EXTRACTOR_TASKS:
        load_feature_extractor = False

    if load_tokenizer:
        # Try to infer tokenizer from model or config name (if provided as str)
        if tokenizer is None:
            if isinstance(model_name, str):
                tokenizer = model_name
            elif isinstance(config, str):
                tokenizer = config
            else:
                # Impossible to guess what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        # Instantiate tokenizer if needed
        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                use_fast = tokenizer[1].pop("use_fast", use_fast)
                tokenizer_identifier = tokenizer[0]
                tokenizer_kwargs = tokenizer[1]
            else:
                tokenizer_identifier = tokenizer
                tokenizer_kwargs = model_kwargs

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, revision=revision, use_fast=use_fast, _from_pipeline=task, **tokenizer_kwargs
            )

    if load_feature_extractor:
        # Try to infer feature extractor from model or config name (if provided as str)
        if feature_extractor is None:
            if isinstance(model_name, str):
                feature_extractor = model_name
            elif isinstance(config, str):
                feature_extractor = config
            else:
                # Impossible to guess what is the right feature_extractor here
                raise Exception(
                    "Impossible to guess which feature extractor to use. "
                    "Please provide a PreTrainedFeatureExtractor class or a path/identifier "
                    "to a pretrained feature extractor."
                )

        # Instantiate feature_extractor if needed
        if isinstance(feature_extractor, (str, tuple)):
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                feature_extractor, revision=revision, _from_pipeline=task, **model_kwargs
            )

            if (
                feature_extractor._processor_class
                and feature_extractor._processor_class.endswith("WithLM")
                and isinstance(model_name, str)
            ):
                try:
                    import kenlm  # to trigger `ImportError` if not installed
                    from pyctcdecode import BeamSearchDecoderCTC

                    if os.path.isdir(model_name) or os.path.isfile(model_name):
                        decoder = BeamSearchDecoderCTC.load_from_dir(model_name)
                    else:
                        language_model_glob = os.path.join(
                            BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY, "*"
                        )
                        alphabet_filename = BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
                        allow_regex = [language_model_glob, alphabet_filename]
                        decoder = BeamSearchDecoderCTC.load_from_hf_hub(model_name, allow_regex=allow_regex)

                    kwargs["decoder"] = decoder
                except ImportError as e:
                    logger.warning(
                        f"Could not load the `decoder` for {model_name}. Defaulting to raw CTC. Try to install"
                        " `pyctcdecode` and `kenlm`: (`pip install pyctcdecode`, `pip install"
                        f" https://github.com/kpu/kenlm/archive/master.zip`): Error: {e}"
                    )

    if task == "translation" and model.config.task_specific_params:
        for key in model.config.task_specific_params:
            if key.startswith("translation"):
                task = key
                warnings.warn(
                    f'"translation" task was used, instead of "translation_XX_to_YY", defaulting to "{task}"',
                    UserWarning,
                )
                break

    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer

    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor

    return pipeline_class(model=model, framework=framework, task=task, **kwargs)
