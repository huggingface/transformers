# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

from transformers import *

import argparse
import collections.abc
import copy
import inspect
import json
import multiprocessing
import os
import shutil
import tempfile
import traceback
from pathlib import Path

from check_config_docstrings import get_checkpoint_from_config_class
from datasets import load_dataset
from get_test_info import get_model_to_tester_mapping, get_tester_classes_for_model, get_test_module
from huggingface_hub import create_repo, hf_api, upload_folder

from transformers import (
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    PROCESSOR_MAPPING,
    TOKENIZER_MAPPING,
    AutoTokenizer,
    LayoutLMv3TokenizerFast,
    PreTrainedTokenizerFast,
    PythonBackend,
    logging, ViTImageProcessor, AutoImageProcessor, AutoFeatureExtractor, AutoVideoProcessor,
)
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.file_utils import is_torch_available
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.auto.configuration_auto import AutoConfig, model_type_to_module_name
from transformers.models.fsmt import configuration_fsmt
from transformers.processing_utils import ProcessorMixin, transformers_module
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# make sure tokenizer plays nice with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.set_verbosity_error()
logging.disable_progress_bar()
logger = logging.get_logger(__name__)

if not is_torch_available():
    raise ValueError("Please install PyTorch.")


INVALID_ARCH = []
TARGET_VOCAB_SIZE = 1024

data = {"training_ds": None, "testing_ds": None}

COMPOSITE_MODELS = {
    "EncoderDecoderModel": "EncoderDecoderModel-bert-bert",
    "SpeechEncoderDecoderModel": "SpeechEncoderDecoderModel-wav2vec2-bert",
    "VisionEncoderDecoderModel": "VisionEncoderDecoderModel-vit-gpt2",
    "VisionTextDualEncoderModel": "VisionTextDualEncoderModel-vit-bert",
}

# This list contains the model architectures for which a tiny version could not be created.
# Avoid to add new architectures here - unless we have verified carefully that it's (almost) impossible to create them.
# One such case is: no model tester class is implemented for a model type (like `MT5`) because its architecture is
# identical to another one (`MT5` is based on `T5`), but trained on different datasets or with different techniques.
UNCONVERTIBLE_MODEL_ARCHITECTURES = {
    "BertGenerationEncoder",
    "BertGenerationDecoder",
    "CamembertForSequenceClassification",
    "CamembertForMultipleChoice",
    "CamembertForMaskedLM",
    "CamembertForCausalLM",
    "CamembertForTokenClassification",
    "CamembertForQuestionAnswering",
    "CamembertModel",
    "DecisionTransformerModel",
    "GraphormerModel",
    "InformerModel",
    "JukeboxModel",
    "MarianForCausalLM",
    "MaskFormerSwinModel",
    "MaskFormerSwinBackbone",
    "MT5Model",
    "MT5ForConditionalGeneration",
    "UMT5ForConditionalGeneration",
    "QDQBertForSequenceClassification",
    "QDQBertForMaskedLM",
    "QDQBertModel",
    "QDQBertForTokenClassification",
    "QDQBertLMHeadModel",
    "QDQBertForMultipleChoice",
    "QDQBertForQuestionAnswering",
    "QDQBertForNextSentencePrediction",
    "ReformerModelWithLMHead",
    "RetriBertModel",
    "Speech2Text2ForCausalLM",
    "TimeSeriesTransformerModel",
    "TrajectoryTransformerModel",
    "TrOCRForCausalLM",
    "XLMProphetNetForConditionalGeneration",
    "XLMProphetNetForCausalLM",
    "XLMProphetNetModel",
    "XLMRobertaModel",
    "XLMRobertaForTokenClassification",
    "XLMRobertaForMultipleChoice",
    "XLMRobertaForMaskedLM",
    "XLMRobertaForCausalLM",
    "XLMRobertaForSequenceClassification",
    "XLMRobertaForQuestionAnswering",
}


config_class_to_model_tester_map = {
    "Qwen3OmniMoeConfig": None,  # Only has `Qwen3OmniMoeThinkerForConditionalGenerationTester` which returns `Qwen3OmniMoeThinkerConfig`
    "Qwen2_5OmniConfig": None,  # Only has `Qwen2_5OmniThinkerForConditionalGenerationTester` which returns `Qwen2_5OmniThinkerConfig`
    "PeAudioVideoConfig": None,  # Only has `PeAudioVideoEncoderTester` which returns `PeAudioVideoEncoderConfig`
    "Qwen3_5Config": "Qwen3_5VisionText2TextModelTester",
    "Qwen3_5MoeConfig": "Qwen3_5MoeVisionText2TextModelTester",
    "InstructBlipConfig": "InstructBlipForConditionalGenerationDecoderOnlyModelTester",
    "InstructBlipVideoConfig": "InstructBlipVideoForConditionalGenerationDecoderOnlyModelTester",
    "MllamaConfig": "MllamaVisionText2TextModelTester",
    "Gemma3nConfig": "Gemma3nVision2TextModelTester",
    "Gemma3Config": "Gemma3Vision2TextModelTester",
    "VideoLlama3Config": "VideoLlama3VisionText2TextModelTester",
    "JanusConfig": "JanusVisionText2TextModelTester",
    "Emu3Config": "Emu3Vision2TextModelTester",
    # Need `torchcodec` and `ffmpeg` --> fixed now
    # Note that: `ClvpModel` and `ClvpForCausalLM` is to `ClvpDecoderConfig`
    # Also: in auto map: only ("clvp", "ClvpModelForConditionalGeneration")
    "ClvpConfig": "ClvpModelForConditionalGenerationTester",
    "BarkConfig": "BarkModelTester",
    "FastSpeech2ConformerWithHifiGanConfig": "FastSpeech2ConformerWithHifiGanTester",
    "Gemma3nAudioConfig": "Gemma3nAudioModelTester",
    # "Blip2QFormerConfig": "Blip2QFormerModelTester",
}


no_model_tester_at_all = {
    "EdgeTamVideoConfig",
    "Llama4Config",
    "Llama4TextConfig",
    "Sam2VideoConfig",
    "Sam3TrackerVideoConfig",
    "Sam3VideoConfig",
    "ShieldGemma2Config",
}

deprecated_models = {
    "DinatConfig",
}


config_without_meaningful_model_class = {
    "Gemma3nVisionConfig",  # It has `TimmWrapperModel`, which is already created under `TimmWrapperConfig` (there is no `Gemma3nVisionModel`)
}


configs_requiring_too_exotic_dependency = {
    # require `detectron2`. It has no `get_config` method: we can implement it but this model is not maintained anymore.
    "LayoutLMv2Config",
}



def get_processor_types_from_config_class(config_class, allowed_mappings=None):
    """Return a tuple of processors for `config_class`.

    We use `tuple` here to include (potentially) both slow & fast tokenizers.
    """

    # To make a uniform return type
    def _to_tuple(x):
        if not isinstance(x, collections.abc.Sequence):
            x = (x,)
        else:
            x = tuple(x)
        return x

    if allowed_mappings is None:
        allowed_mappings = ["processor", "tokenizer", "image_processor", "feature_extractor"]

    processor_types = ()

    # Check first if a model has `ProcessorMixin`. Otherwise, check if it has tokenizers, and/or an image processor or
    # a feature extractor
    if config_class in PROCESSOR_MAPPING and "processor" in allowed_mappings:
        processor_types = _to_tuple(PROCESSOR_MAPPING[config_class])
    else:
        if config_class in TOKENIZER_MAPPING and "tokenizer" in allowed_mappings:
            processor_types = _to_tuple(TOKENIZER_MAPPING[config_class])

        if config_class in IMAGE_PROCESSOR_MAPPING and "image_processor" in allowed_mappings:
            processor_types += _to_tuple(IMAGE_PROCESSOR_MAPPING[config_class])
        elif config_class in FEATURE_EXTRACTOR_MAPPING and "feature_extractor" in allowed_mappings:
            processor_types += _to_tuple(FEATURE_EXTRACTOR_MAPPING[config_class])

    # Remark: some configurations have no processor at all. For example, generic composite models like
    # `EncoderDecoderModel` is used for any (compatible) text models. Also, `DecisionTransformer` doesn't
    # require any processor.

    # We might get `None` for some tokenizers - remove them here.
    processor_types = tuple(p for p in processor_types if p is not None)

    # Add what ever auto types
    processor_types += (AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor, AutoVideoProcessor)

    return processor_types


def get_architectures_from_config_class(config_class, arch_mappings, models_to_skip=None):
    """Return a tuple of all possible architectures attributed to a configuration class `config_class`.

    For example, BertConfig -> [BertModel, BertForMaskedLM, ..., BertForQuestionAnswering].
    """
    # A model architecture could appear in several mappings. For example, `BartForConditionalGeneration` is in
    #   - MODEL_FOR_PRETRAINING_MAPPING_NAMES
    #   - MODEL_FOR_MASKED_LM_MAPPING_NAMES
    #   - MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    # We avoid the duplication.
    architectures = set()

    if models_to_skip is None:
        models_to_skip = []
    models_to_skip = UNCONVERTIBLE_MODEL_ARCHITECTURES.union(models_to_skip)

    for mapping in arch_mappings:
        if config_class in mapping:

            try:
                models = mapping[config_class]
            except ValueError as e:
                import re, importlib, inspect

                # Extract missing model name from error message
                match = re.search(r'Could not find (\w+)', str(e))
                missing_model_name = match.group(1) if match else None

                # Get the package module from config_class
                module_path = config_class.__module__.rsplit('.', 1)[0]  # e.g. 'transformers.models.voxtral_realtime'
                module = importlib.import_module(module_path)

                # Find modeling_* submodule names
                modeling_names = [name for name in dir(module) if name.startswith('modeling_')]

                models = ()
                for modeling_name in modeling_names:
                    modeling_module = getattr(module, modeling_name)
                    _models = getattr(modeling_module, missing_model_name, None)
                    if models is not None:
                        models = _models
                        break

            models = tuple(models) if isinstance(models, collections.abc.Sequence) else (models,)
            for model in models:
                if model.__name__ not in models_to_skip:
                    architectures.add(model)

    architectures = tuple(architectures)

    return architectures


def get_config_class_from_processor_class(processor_class):
    """Get the config class from a processor class.

    Some config/model classes use tokenizers/feature_extractors from other models. For example, `GPT-J` uses
    `GPT2Tokenizer`. If no checkpoint is found for a config class, or a checkpoint is found without necessary file(s) to
    create the processor for `processor_class`, we get the config class that corresponds to `processor_class` and use it
    to find a checkpoint in order to create the processor.
    """

    processor_prefix = processor_class.__name__
    # The order is important: e.g. `TokenizerFast` must before `Tokenizer` etc.
    for postfix in ["TokenizerFast", "Tokenizer", "ImageProcessorFast", "ImageProcessor", "FeatureExtractor", "Processor"]:
        processor_prefix = processor_prefix.replace(postfix, "")

    # `Wav2Vec2CTCTokenizer` -> `Wav2Vec2Config`
    if processor_prefix == "Wav2Vec2CTC":
        processor_prefix = "Wav2Vec2"

    # Find the new configuration class
    # breakpoint()
    new_config_name = f"{processor_prefix}Config"
    new_config_class = getattr(transformers_module, new_config_name)

    return new_config_class


def build_processor(config_class, processor_class, allow_no_checkpoint=False):
    """Create a processor for `processor_class`.

    If a processor is not able to be built with the original arguments, this method tries to change the arguments and
    call itself recursively, by inferring a new `config_class` or a new `processor_class` from another one, in order to
    find a checkpoint containing the necessary files to build a processor.

    The processor is not saved here. Instead, it will be saved in `convert_processors` after further changes in
    `convert_processors`. For each model architecture`, a copy will be created and saved along the built model.
    """
    # Currently, this solely uses the docstring in the source file of `config_class` to find a checkpoint.
    # breakpoint()
    checkpoint = get_checkpoint_from_config_class(config_class)

    # New method that is more robust to get checkpoints!

    # breakpoint()

    if checkpoint is None and not processor_class.__name__.startswith("Auto"):
        # try to get the checkpoint from the config class for `processor_class`.
        # This helps cases like `XCLIPConfig` and `VideoMAEFeatureExtractor` to find a checkpoint from `VideoMAEConfig`.
        config_class_from_processor_class = get_config_class_from_processor_class(processor_class)
        checkpoint = get_checkpoint_from_config_class(config_class_from_processor_class)

    processor = None
    try:
        # breakpoint()
        revision = None
        # TODO: a better handle for revisions
        if config_class.__name__ == 'NanoChatConfig':
            revision = "refs/pr/1"
        elif config_class.__name__ == 'Ernie4_5_VL_MoeConfig':
            revision = "refs/pr/10"

        sub_folder = ""
        if config_class.__name__ in ['GlmImageTextConfig', 'GlmImageVisionConfig', 'GlmImageVQVAEConfig']:
            sub_folder = "processor"

        # breakpoint()
        processor = processor_class.from_pretrained(checkpoint, revision=revision, subfolder=sub_folder)
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {e}")

    # Try to get a new processor class from checkpoint. This is helpful for a checkpoint without necessary file to load
    # processor while `processor_class` is an Auto class. For example, `sew` has `Wav2Vec2Processor` in
    # `PROCESSOR_MAPPING_NAMES`, its `tokenizer_class` is `AutoTokenizer`, and the checkpoint
    # `https://huggingface.co/asapp/sew-tiny-100k` has no tokenizer file, but we can get
    # `tokenizer_class: Wav2Vec2CTCTokenizer` from the config file. (The new processor class won't be able to load from
    # `checkpoint`, but it helps this recursive method to find a way to build a processor).
    # try:
    #     issubclass(processor_class, (PreTrainedTokenizerBase, AutoTokenizer))
    # except:
    #     breakpoint()


    if (
        processor is None
        and checkpoint is not None
        and issubclass(processor_class, (PreTrainedTokenizerBase, AutoTokenizer))
    ):
        try:
            # breakpoint()
            revision = None
            # TODO: a better handle for revisions
            if config_class.__name__ == 'NanoChatConfig':
                revision = "refs/pr/1"
            config = AutoConfig.from_pretrained(checkpoint, revision=revision)
        except Exception as e:
            # breakpoint()
            logger.error(f"{e.__class__.__name__}: {e}")
            config = None
        if config is not None:
            # TODO: sam2 (Sam2Config) from `facebook/sam2.1-hiera-tiny` will fail if we don't add `getattr(config, "tokenizer_class", None) is not None`
            # (as we get `Sam2VideoConfig` instead of `Sam2Config`)
            if getattr(config, "tokenizer_class", None) is not None and not isinstance(config, config_class):
                raise ValueError(
                    f"`config` (which is of type {config.__class__.__name__}) should be an instance of `config_class`"
                    f" ({config_class.__name__})!"
                )
            if getattr(config, "tokenizer_class", None) is not None:
                tokenizer_class = config.tokenizer_class
                new_processor_class = None
                if tokenizer_class is not None:
                    # breakpoint()

                    # Some hub configs have the wrong values!!! (e.g. it is `CPMAntTokenizer` but should be `CpmAntTokenizer`)
                    new_processor_class = getattr(transformers_module, tokenizer_class, None)

                    if new_processor_class is not None and new_processor_class != processor_class:
                        processor = build_processor(config_class, new_processor_class)
                # If `tokenizer_class` is not specified in `config`, let's use `config` to get the process class via auto
                # mappings, but only allow the tokenizer mapping being used. This is to make `Wav2Vec2Conformer` build
                if processor is None:
                    new_processor_classes = get_processor_types_from_config_class(
                        config.__class__, allowed_mappings=["tokenizer"]
                    )
                    # breakpoint()
                    # Used to avoid infinite recursion between a pair of fast/slow tokenizer types
                    names = [
                        x.__name__.replace("Fast", "") for x in [processor_class, new_processor_class] if x is not None
                    ]
                    new_processor_classes = [
                        x for x in new_processor_classes if x is not None and x.__name__.replace("Fast", "") not in names
                    ]
                    if len(new_processor_classes) > 0:
                        new_processor_class = new_processor_classes[0]
                        # Let's use fast tokenizer if there is any
                        # TODO: this is likely be very misleading!!!
                        for x in new_processor_classes:
                            if x.__name__.endswith("Fast"):
                                new_processor_class = x
                                break
                        processor = build_processor(config_class, new_processor_class)

    if processor is None:
        # # Try to build each component (tokenizer & feature extractor) of a `ProcessorMixin`.
        # if issubclass(processor_class, ProcessorMixin):
        #     attrs = {}
        #     for attr_name in processor_class.get_attributes():
        #         attrs[attr_name] = []
        #         # This could be a tuple (for tokenizers). For example, `CLIPProcessor` has
        #         #   - feature_extractor_class = "CLIPFeatureExtractor"
        #         #   - tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")
        #         try:
        #             attr_class_names = getattr(processor_class, f"{attr_name}_class")
        #         except:
        #             # breakpoint()
        #         if not isinstance(attr_class_names, tuple):
        #             attr_class_names = (attr_class_names,)
        #
        #         for name in attr_class_names:
        #             attr_class = getattr(transformers_module, name)
        #             attr = build_processor(config_class, attr_class)
        #             if attr is not None:
        #                 attrs[attr_name].append(attr)
        #
        #     # try to build a `ProcessorMixin`, so we can return a single value
        #     if all(len(v) > 0 for v in attrs.values()):
        #         try:
        #             processor = processor_class(**{k: v[0] for k, v in attrs.items()})
        #         except Exception as e:
        #             logger.error(f"{e.__class__.__name__}: {e}")
        if not processor_class.__name__.startswith("Auto"):
            # `checkpoint` might lack some file(s) to load a processor. For example, `facebook/hubert-base-ls960`
            # has no tokenizer file to load `Wav2Vec2CTCTokenizer`. In this case, we try to build a processor
            # with the configuration class (for example, `Wav2Vec2Config`) corresponding to `processor_class`.
            config_class_from_processor_class = get_config_class_from_processor_class(processor_class)
            if config_class_from_processor_class != config_class:
                processor = build_processor(config_class_from_processor_class, processor_class)

    # breakpoint()
    # Try to create an image processor or a feature extractor without any checkpoint
    if (
        processor is None
        and allow_no_checkpoint
        and (issubclass(processor_class, BaseImageProcessor) or issubclass(processor_class, FeatureExtractionMixin))
    ):
        try:
            processor = processor_class()
        except Exception as e:
            logger.error(f"{e.__class__.__name__}: {e}")

    # validation
    # breakpoint()
    # TODO: We might get `TokenizersBackend` in a recursive call (using `AutoTokenizer` class) and might fail if we don't add the condition
    # `isinstance(processor, TokenizersBackend)`!! (e.g. Yoso!)
    if processor is not None:
        from transformers import TokenizersBackend
        if not (isinstance(processor, processor_class) or isinstance(processor, TokenizersBackend) or processor_class.__name__.startswith("Auto")):
            raise ValueError(
                f"`processor` (which is of type {processor.__class__.__name__}) should be an instance of"
                f" {processor_class.__name__} or an Auto class!"
            )

    # breakpoint()
    return processor


# recursively get the correct config
def _get_exact_config(_config, config_class):

    # breakpoint()
    if isinstance(_config, config_class):
        return _config

    # TODO: T5Gemma2 has `encoder` and `decoder` instead `_config`

    # We consider both cases: real config or dict (for `FastSpeech2ConformerConfig`'s encoder/decoder config, which are only module config)
    config_dict = _config.to_dict() if not isinstance(_config, dict) else _config

    keys = [x for x in config_dict.keys() if x.endswith("_config") or x in ["encoder", "decoder"]]
    for key in keys:
        sub_config = getattr(_config, key) if not isinstance(_config, dict) else _config[key]
        if sub_config is not None:
            maybe_config = _get_exact_config(sub_config, config_class)
            if isinstance(maybe_config, config_class):
                return maybe_config

    return _config


# TODO: Sam2Video will fail here
def get_tiny_config(config_class, model_class=None, **model_tester_kwargs):
    """Retrieve a tiny configuration from `config_class` using each model's `ModelTester`.

    Args:
        config_class: Subclass of `PreTrainedConfig`.

    Returns:
        An instance of `config_class` with tiny hyperparameters
    """
    # breakpoint()
    model_type = config_class.model_type

    # For model type like `data2vec-vision` and `donut-swin`, we can't get the config/model file name directly via
    # `model_type` as it would be sth. like `configuration_data2vec_vision.py`.
    # A simple way is to use `inspect.getsourcefile(config_class)`.
    config_source_file = inspect.getsourcefile(config_class)
    # The modeling file name without prefix (`modeling_`) and postfix (`.py`)
    modeling_name = config_source_file.split(os.path.sep)[-1].replace("configuration_", "").replace(".py", "")
    # TODO: remark: several configuration classes might be defined in the same modeling directory.
    #   The test directory is still the same, so we are good here.

    try:
        print("Importing", model_type_to_module_name(model_type))
        module_name = model_type_to_module_name(model_type)
        # breakpoint()
        if not modeling_name.startswith(module_name):
            raise ValueError(f"{modeling_name} doesn't start with {module_name}!")
        test_file = os.path.join("tests", "models", module_name, f"test_modeling_{modeling_name}.py")
        models_to_model_testers = get_model_to_tester_mapping(test_file)
        # Find the model tester class
        model_tester_class = None
        tester_classes = []
        if model_class is not None:
            tester_classes = get_tester_classes_for_model(test_file, model_class)
        else:
            for _tester_classes in models_to_model_testers.values():
                tester_classes.extend(_tester_classes)
        if len(tester_classes) > 0:
            # breakpoint()
            # sort with the length of the class names first, then the alphabetical order
            # This is to avoid `T5EncoderOnlyModelTest` is used instead of `T5ModelTest`, which has
            # `is_encoder_decoder=False` and causes some pipeline tests failing (also failures in `Optimum` CI).
            # TODO: More fine grained control of the desired tester class.
            model_tester_class = min(tester_classes, key=lambda x: (len(x.__name__), x.__name__))

            # TODO: SpeechT5ForSpeechToText needs a particular tester to get the working config
            # TODO: this is hacky however, as all model classes share the same config class but having different tester
            # TODO: make this more flexible and roubst
            if config_class.__name__ == "SpeechT5Config":
                for x in tester_classes:
                    if x.__name__ == "SpeechT5ForSpeechToTextTester":
                        model_tester_class = x
                        break

    except ModuleNotFoundError:
        error = f"Tiny config not created for {model_type} - cannot find the testing module from the model name."
        raise ValueError(error)

    if model_tester_class is None:
        # breakpoint()
        error = f"Tiny config not created for {model_type} - no model tester is found in the testing module."
        raise ValueError(error)

    # CLIP-like models have `text_model_tester` and `vision_model_tester`, and we need to pass `vocab_size` to
    # `text_model_tester` via `text_kwargs`. The same trick is also necessary for `Flava`.

    if "vocab_size" in model_tester_kwargs:
        if "text_kwargs" in inspect.signature(model_tester_class.__init__).parameters:
            vocab_size = model_tester_kwargs.pop("vocab_size")
            model_tester_kwargs["text_kwargs"] = {"vocab_size": vocab_size}

    # `parent` is an instance of `unittest.TestCase`, but we don't need it here.

    # TODO: we need to make sure the kwargs are actually arguments!
    #   But we are likely NOT to override anymore! Let's do something easy and quick here despite ugly.
    try:
        # breakpoint()
        model_tester = model_tester_class(parent=None, **model_tester_kwargs)
    except TypeError as e:

        # if "vocab_size" in model_tester_kwargs:
        model_tester_kwargs_new = {k: v for k, v in model_tester_kwargs.items() if k != "vocab_size"}

        # we need to handle unusual arguments, like "config_kwargs" in `PeVideoTextModelTester` (not good practice but understandable)
        for k, v in model_tester_kwargs_new.items():
            if isinstance(v, dict):
                model_tester_kwargs_new[k] = {k1: v1 for k1, v1 in v.items() if k1 != "vocab_size"}

        # breakpoint()
        model_tester = model_tester_class(parent=None, **model_tester_kwargs_new)

    # breakpoint()
    if hasattr(model_tester, "get_pipeline_config"):
        config = model_tester.get_pipeline_config()
    elif hasattr(model_tester, "prepare_config_and_inputs"):
        # `PoolFormer` has no `get_config` defined. Furthermore, it's better to use `prepare_config_and_inputs` even if
        # `get_config` is defined, since there might be some extra changes in `prepare_config_and_inputs`.
        # breakpoint()
        # We don't really need to call `prepare_config_and_inputs` which might require more dependencies
        if hasattr(model_tester, "get_config"):
            try:
                config = model_tester.prepare_config_and_inputs()[0]
            except Exception as e:
                config = model_tester.get_config()
        else:
            config = model_tester.prepare_config_and_inputs()[0]

    elif hasattr(model_tester, "get_config"):
        config = model_tester.get_config()
    else:
        error = (
            f"Tiny config not created for {model_type} - the model tester {model_tester_class.__name__} lacks"
            " necessary method to create config."
        )
        raise ValueError(error)

    # breakpoint()
    config = _get_exact_config(config, config_class)

    # TODO: For `pe_audio_video`: the tester only gives `PeAudioVideoEncoderConfig` and can't create model for `PeAudioVideoModel`
    # TODO: This part is necessary for Gemma3Model!
    # TODO: Make this part much better without duplicating the code and less error prone
    # breakpoint()
    if not isinstance(config, config_class):
        model_tester_class_name = config_class_to_model_tester_map.get(config_class.__name__, None)
        if model_tester_class_name is not None:
            test_module = get_test_module(test_file)
            new_model_tester_class = getattr(test_module, model_tester_class_name)


            #　TODO: Avoid code duplication
            # TODO: we need to make sure the kwargs are actually arguments!
            #   But we are likely NOT to override anymore! Let's do something easy and quick here despite ugly.
            try:
                # breakpoint()
                new_model_tester = new_model_tester_class(parent=None, **model_tester_kwargs)
            except TypeError as e:

                # if "vocab_size" in model_tester_kwargs:
                model_tester_kwargs_new = {k: v for k, v in model_tester_kwargs.items() if k != "vocab_size"}

                # we need to handle unusual arguments, like "config_kwargs" in `PeVideoTextModelTester` (not good practice but understandable)
                for k, v in model_tester_kwargs_new.items():
                    if isinstance(v, dict):
                        model_tester_kwargs_new[k] = {k1: v1 for k1, v1 in v.items() if k1 != "vocab_size"}

                new_model_tester = new_model_tester_class(parent=None, **model_tester_kwargs_new)

            ### new_model_tester = new_model_tester_class(parent=None, **model_tester_kwargs)

            # breakpoint()
            model_tester = new_model_tester

            if hasattr(model_tester, "get_pipeline_config"):
                config = model_tester.get_pipeline_config()
            elif hasattr(model_tester, "prepare_config_and_inputs"):
                # `PoolFormer` has no `get_config` defined. Furthermore, it's better to use `prepare_config_and_inputs` even if
                # `get_config` is defined, since there might be some extra changes in `prepare_config_and_inputs`.

                # We don't really need to call `prepare_config_and_inputs` which might require more dependencies
                if hasattr(model_tester, "get_config"):
                    try:
                        config = model_tester.prepare_config_and_inputs()[0]
                    except Exception as e:
                        config = model_tester.get_config()
                else:
                    config = model_tester.prepare_config_and_inputs()[0]

            elif hasattr(model_tester, "get_config"):
                config = model_tester.get_config()
            else:
                error = (
                    f"Tiny config not created for {model_type} - the model tester {model_tester_class.__name__} lacks"
                    " necessary method to create config."
                )
                raise ValueError(error)


        # TODO: Disabled as this causes issues due to much larger models
        # # TODO: For `pe_audio_video`: the tester only gives `PeAudioVideoEncoderConfig` and can't create model for `PeAudioVideoModel`
        # #   we try to find if `config` is a subconfig for `config_class`. If so, return `config_class()` after setting that attr. to `config`
        # # TODO: But this might get very large model?
        # # TODO: This part is necessary for Gemma3Model!
        # config_from_class = config_class()
        # keys = config_from_class.to_dict().keys()
        # for key in keys:
        #     if key.endswith("_config"):
        #         o = getattr(config_from_class, key)
        #         if isinstance(config, o.__class__):
        #             setattr(config_from_class, key, config)
        #             config = config_from_class
        #             break

    # breakpoint()
    # make sure this is long enough (some model tester has `20` for this attr.) to pass `text-generation`
    # pipeline tests.
    max_positions = []
    for key in ["max_position_embeddings", "max_source_positions", "max_target_positions"]:
        if getattr(config, key, 0) > 0:
            max_positions.append(getattr(config, key))
        if getattr(config, "text_config", None) is not None:
            if getattr(config.text_config, key, None) is not None:
                max_positions.append(getattr(config.text_config, key))
    if len(max_positions) > 0:
        max_position = max(200, min(max_positions))
        for key in ["max_position_embeddings", "max_source_positions", "max_target_positions"]:
            if getattr(config, key, 0) > 0:
                setattr(config, key, max_position)
            if getattr(config, "text_config", None) is not None:
                if getattr(config.text_config, key, None) is not None:
                    setattr(config.text_config, key, max_position)

    # TODO: We have this `self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size` in `InstructBlipConfig`,
    #   and we need to do it here otherwise shape issue!!!
    # TODO: But the actual problem is that we should try to get `InstructBlipConfig` in the first place instead of `InstructBlipVisionConfig`.
    # (At this moment, we get tiny `InstructBlipVisionConfig`, and then full `InstructBlipConfig` with tiny `InstructBlipVisionConfig`: from the trick above)
    if config.__class__.__name__ in ["InstructBlipConfig", "InstructBlipVideoConfig"]:
        config.qformer_config.encoder_hidden_size = config.vision_config.hidden_size

    # breakpoint()
    return config


def convert_tokenizer(tokenizer_fast: PreTrainedTokenizerFast):
    new_tokenizer = tokenizer_fast.train_new_from_iterator(
        data["training_ds"]["text"], TARGET_VOCAB_SIZE, show_progress=False
    )

    # Make sure it at least runs
    if not isinstance(new_tokenizer, LayoutLMv3TokenizerFast):
        new_tokenizer(list(data["testing_ds"]["text"]))

    return new_tokenizer


def convert_feature_extractor(feature_extractor, tiny_config):
    to_convert = False
    kwargs = {}
    if hasattr(tiny_config, "image_size"):
        kwargs["size"] = tiny_config.image_size
        kwargs["crop_size"] = tiny_config.image_size
        to_convert = True
    elif (
        hasattr(tiny_config, "vision_config")
        and tiny_config.vision_config is not None
        and hasattr(tiny_config.vision_config, "image_size")
    ):
        kwargs["size"] = tiny_config.vision_config.image_size
        kwargs["crop_size"] = tiny_config.vision_config.image_size
        to_convert = True

    # Speech2TextModel specific.
    if hasattr(tiny_config, "input_feat_per_channel"):
        kwargs["feature_size"] = tiny_config.input_feat_per_channel
        kwargs["num_mel_bins"] = tiny_config.input_feat_per_channel
        to_convert = True

    if to_convert:
        feature_extractor = feature_extractor.__class__(**kwargs)

    # Sanity check: on tiny image feature extractors, a large image size results in slow CI -- up to the point where it
    # can result in timeout issues.
    if (
        isinstance(feature_extractor, BaseImageProcessor)
        and hasattr(feature_extractor, "size")
        and isinstance(feature_extractor.size, dict)
    ):
        largest_image_size = max(feature_extractor.size.values())
        if largest_image_size > 64:
            # hardcoded exceptions
            models_with_large_image_size = ("deformable_detr", "flava", "grounding_dino", "mgp_str", "swiftformer")
            if any(model_name in tiny_config.model_type for model_name in models_with_large_image_size):
                pass

            # TODO: Disabling this might get very slow tests!! Need to check the run time !!!
            # else:
            #     raise ValueError(
            #         f"Image size of {tiny_config.model_type} is too large ({feature_extractor.size}). "
            #         "Please reduce it to 64 or less on each dimension. The following steps are usually the "
            #         "easiest solution: 1) confirm that you're setting `image_size` in your ModelTester class; "
            #         "2) ensure that it gets passed to the tester config init, `get_config()`."
            #     )

    return feature_extractor


def convert_processors(processors, tiny_config, output_folder, result):
    """Change a processor to work with smaller inputs.

    For tokenizers, we try to reduce their vocabulary size.

    For feature extractor, we use smaller image size or change
    other attributes using the values from `tiny_config`. See `convert_feature_extractor`.

    This method should not fail: we catch the errors and put them in `result["warnings"]` with descriptive messages.
    """

    def _sanity_check(fast_tokenizer, slow_tokenizer, keep_fast_tokenizer=False):
        """Set tokenizer(s) to `None` if the fast/slow tokenizers have different values for `vocab_size` or `length`.

        If `keep_fast_tokenizer=True`, the fast tokenizer will be kept.
        """
        # sanity check 1: fast and slow tokenizers should be compatible (vocab_size)
        if fast_tokenizer is not None and slow_tokenizer is not None:
            if fast_tokenizer.vocab_size != slow_tokenizer.vocab_size:
                warning_message = (
                    "The fast/slow tokenizers "
                    f"({fast_tokenizer.__class__.__name__}/{slow_tokenizer.__class__.__name__}) have different "
                    "vocabulary size: "
                    f"fast_tokenizer.vocab_size = {fast_tokenizer.vocab_size} and "
                    f"slow_tokenizer.vocab_size = {slow_tokenizer.vocab_size}."
                )
                result["warnings"].append(warning_message)
                if not keep_fast_tokenizer:
                    fast_tokenizer = None
                slow_tokenizer = None

        # sanity check 2: fast and slow tokenizers should be compatible (length)
        if fast_tokenizer is not None and slow_tokenizer is not None:
            if len(fast_tokenizer) != len(slow_tokenizer):
                warning_message = (
                    f"The fast/slow tokenizers () have different length: "
                    f"len(fast_tokenizer) = {len(fast_tokenizer)} and "
                    f"len(slow_tokenizer) = {len(slow_tokenizer)}."
                )
                result["warnings"].append(warning_message)
                if not keep_fast_tokenizer:
                    fast_tokenizer = None
                slow_tokenizer = None

        return fast_tokenizer, slow_tokenizer

    tokenizers = []
    # breakpoint()
    feature_extractors = []
    for processor in processors:
        if isinstance(processor, PreTrainedTokenizerBase):
            if processor.__class__.__name__ not in {x.__class__.__name__ for x in tokenizers}:
                tokenizers.append(processor)
        elif isinstance(processor, BaseImageProcessor):
            if processor.__class__.__name__ not in {x.__class__.__name__ for x in feature_extractors}:
                feature_extractors.append(processor)
        elif isinstance(processor, FeatureExtractionMixin):
            if processor.__class__.__name__ not in {x.__class__.__name__ for x in feature_extractors}:
                feature_extractors.append(processor)
        elif isinstance(processor, ProcessorMixin):
            if hasattr(processor, "tokenizer"):
                if processor.tokenizer.__class__.__name__ not in {x.__class__.__name__ for x in tokenizers}:
                    tokenizers.append(processor.tokenizer)
            # Currently, we only have these 2 possibilities
            if hasattr(processor, "image_processor"):
                if processor.image_processor.__class__.__name__ not in {
                    x.__class__.__name__ for x in feature_extractors
                }:
                    feature_extractors.append(processor.image_processor)
            elif hasattr(processor, "feature_extractor"):
                if processor.feature_extractor.__class__.__name__ not in {
                    x.__class__.__name__ for x in feature_extractors
                }:
                    feature_extractors.append(processor.feature_extractor)

    # breakpoint()
    # check the built processors have the unique type
    num_types = len({x.__class__.__name__ for x in feature_extractors})
    # if num_types >= 2:
    #     raise ValueError(f"`feature_extractors` should contain at most 1 type, but it contains {num_types} types!")
    num_types = len({x.__class__.__name__.replace("Fast", "") for x in tokenizers})
    # breakpoint()

    # TODO: we might have {'TokenizersBackend', 'MistralCommonBackend'} now! For example, mixtral!
    # TODO: Question: if we need to have "tokenizer.model" or "special_tokens_map.json"?
    # if num_types >= 2:
    #     raise ValueError(f"`tokenizers` should contain at most 1 tokenizer type, but it contains {num_types} types!")

    fast_tokenizer = None
    slow_tokenizer = None

    for tokenizer in tokenizers:
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            fast_tokenizer = tokenizer
        else:
            slow_tokenizer = tokenizer

    # If the (original) fast/slow tokenizers don't correspond, keep only the fast tokenizer.
    # This doesn't necessarily imply the fast/slow tokenizers in a single Hub repo. has issues.
    # It's more of an issue in `build_processor` which tries to get a checkpoint with as much effort as possible.
    # For `YosoModel` (which uses `AlbertTokenizer(Fast)`), its real (Hub) checkpoint doesn't contain valid files to
    # load the slower tokenizer (`AlbertTokenizer`), and it ends up finding the (canonical) checkpoint of `AlbertModel`,
    # which has different vocabulary.
    # TODO: Try to improve `build_processor`'s definition and/or usage to avoid the above situation in the first place.
    fast_tokenizer, slow_tokenizer = _sanity_check(fast_tokenizer, slow_tokenizer, keep_fast_tokenizer=True)
    original_fast_tokenizer, original_slow_tokenizer = fast_tokenizer, slow_tokenizer

    if fast_tokenizer:
        try:
            # Wav2Vec2ForCTC , ByT5Tokenizer etc. all are already small enough and have no fast version that can
            # be retrained
            if fast_tokenizer.vocab_size > TARGET_VOCAB_SIZE:
                pass
                # fast_tokenizer = convert_tokenizer(fast_tokenizer)
        except Exception:
            result["warnings"].append(
                (
                    f"Failed to convert the fast tokenizer for {fast_tokenizer.__class__.__name__}.",
                    traceback.format_exc(),
                )
            )

    # If `fast_tokenizer` exists, `slow_tokenizer` should correspond to it.
    if fast_tokenizer:
        # Make sure the fast tokenizer can be saved
        try:
            # We don't save it to `output_folder` at this moment - only at the end of this function.
            with tempfile.TemporaryDirectory() as tmpdir:
                fast_tokenizer.save_pretrained(tmpdir)
                try:
                    slow_tokenizer = AutoTokenizer.from_pretrained(tmpdir, use_fast=False)
                except Exception:
                    result["warnings"].append(
                        (
                            f"Failed to load the slow tokenizer saved from {fast_tokenizer.__class__.__name__}.",
                            traceback.format_exc(),
                        )
                    )
                    # Let's just keep the fast version
                    slow_tokenizer = None
        except Exception:
            result["warnings"].append(
                (
                    f"Failed to save the fast tokenizer for {fast_tokenizer.__class__.__name__}.",
                    traceback.format_exc(),
                )
            )
            fast_tokenizer = None

    # If the (possibly converted) fast/slow tokenizers don't correspond, set them to `None`, and use the original
    # tokenizers.
    fast_tokenizer, slow_tokenizer = _sanity_check(fast_tokenizer, slow_tokenizer, keep_fast_tokenizer=False)

    # If there is any conversion failed, we keep the original tokenizers.
    if (original_fast_tokenizer is not None and fast_tokenizer is None) or (
        original_slow_tokenizer is not None and slow_tokenizer is None
    ):
        warning_messagae = (
            "There are some issues when converting the fast/slow tokenizers. The original tokenizers from the Hub "
            " will be used instead."
        )
        result["warnings"].append(warning_messagae)
        # Let's use the original version at the end (`original_fast_tokenizer` and `original_slow_tokenizer`)
        fast_tokenizer = original_fast_tokenizer
        slow_tokenizer = original_slow_tokenizer

    # Make sure the fast tokenizer can be saved
    if fast_tokenizer:
        # We don't save it to `output_folder` at this moment - only at the end of this function.
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                fast_tokenizer.save_pretrained(tmpdir)
            except Exception:
                result["warnings"].append(
                    (
                        f"Failed to save the fast tokenizer for {fast_tokenizer.__class__.__name__}.",
                        traceback.format_exc(),
                    )
                )
                fast_tokenizer = None
    # Make sure the slow tokenizer can be saved
    if slow_tokenizer:
        # We don't save it to `output_folder` at this moment - only at the end of this function.
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                slow_tokenizer.save_pretrained(tmpdir)
            except Exception:
                result["warnings"].append(
                    (
                        f"Failed to save the slow tokenizer for {slow_tokenizer.__class__.__name__}.",
                        traceback.format_exc(),
                    )
                )
                slow_tokenizer = None

    # breakpoint()
    # update feature extractors using the tiny config
    try:
        feature_extractors = [convert_feature_extractor(p, tiny_config) for p in feature_extractors]
    except Exception:
        result["warnings"].append(
            (
                "Failed to convert feature extractors.",
                traceback.format_exc(),
            )
        )
        feature_extractors = []
    # breakpoint()

    if hasattr(tiny_config, "max_position_embeddings") and tiny_config.max_position_embeddings > 0:
        if fast_tokenizer is not None:
            if fast_tokenizer.__class__.__name__ in [
                "RobertaTokenizerFast",
                "XLMRobertaTokenizerFast",
                "LongformerTokenizerFast",
                "MPNetTokenizerFast",
            ]:
                fast_tokenizer.model_max_length = tiny_config.max_position_embeddings - 2
            else:
                fast_tokenizer.model_max_length = tiny_config.max_position_embeddings
        if slow_tokenizer is not None:
            if slow_tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "XLMRobertaTokenizer",
                "LongformerTokenizer",
                "MPNetTokenizer",
            ]:
                slow_tokenizer.model_max_length = tiny_config.max_position_embeddings - 2
            else:
                slow_tokenizer.model_max_length = tiny_config.max_position_embeddings

    processors = [fast_tokenizer, slow_tokenizer] + feature_extractors
    processors = [p for p in processors if p is not None]
    for p in processors:
        p.save_pretrained(output_folder)

    return processors


def get_checkpoint_dir(output_dir, model_arch):
    """Get architecture name."""
    arch_name = model_arch.__name__
    return os.path.join(output_dir, arch_name)


def build_model(model_arch, tiny_config, output_dir):
    """Create and save a model for `model_arch`.

    Also copy the set of processors to each model (under the same model type) output folder.
    """

    # breakpoint()
    checkpoint_dir = get_checkpoint_dir(output_dir, model_arch)

    processor_output_dir = os.path.join(output_dir, "processors")
    # copy the (same set of) processors (for a model type) to the model arch. specific folder
    if os.path.isdir(processor_output_dir):
        shutil.copytree(processor_output_dir, checkpoint_dir, dirs_exist_ok=True)

    tiny_config = copy.deepcopy(tiny_config)

    if any(model_arch.__name__.endswith(x) for x in ["ForCausalLM", "LMHeadModel"]):
        tiny_config.is_encoder_decoder = False
        tiny_config.is_decoder = True

    # breakpoint()
    model = model_arch(config=tiny_config)
    # breakpoint()

    # with tempfile.TemporaryDirectory(dir=checkpoint_dir) as tmpdir:
    checkpoint_dir_tmp = checkpoint_dir
    model.save_pretrained(checkpoint_dir_tmp)

    # can't call from_pretrained from saved one
    if not tiny_config.__class__.__name__.endswith(("TimmBackboneConfig",)):
        # breakpoint()
        model.from_pretrained(checkpoint_dir_tmp)

    return model


def fill_result_with_error(result, error, trace, models_to_create):
    """Fill `result` with errors for all target model arch if we can't build processor"""
    error = (error, trace)
    result["error"] = error

    if "pytorch" in models_to_create:
        result["pytorch"] = {}
        for model_arch in models_to_create["pytorch"]:
            result["pytorch"][model_arch.__name__] = {"model": None, "checkpoint": None, "error": error}

    # TODO: check what should we do with error/warning etc.
    #   if we can't get any processor, we fill the report with the class name
    #   otherwise, we could not build with these obtained processors, as we get the error
    #   `error = f"No processor is returned by `convert_processors` for {config_class.__name__}."`
    if len(result["processor"]) == 0:
        # TODO: this dosen't make any sense???
        result["processor"] = {p.__class__.__name__: p.__class__.__name__ for p in result["processor"].values()}


def upload_model(model_dir, organization, token):
    """Upload the tiny models"""

    arch_name = model_dir.split(os.path.sep)[-1]
    repo_name = f"tiny-random-{arch_name}"
    repo_id = f"{organization}/{repo_name}"

    repo_exist = False
    error = None
    try:
        create_repo(repo_id=repo_id, exist_ok=False, repo_type="model", token=token)
    except Exception as e:
        error = e
        if "You already created" in str(e):
            error = None
            logger.warning("Remote repository exists and will be cloned.")
            repo_exist = True
            try:
                create_repo(repo_id=repo_id, exist_ok=True, repo_type="model", token=token)
            except Exception as e:
                error = e
    if error is not None:
        raise error

    create_pr = repo_exist  # Open a PR on existing repo, otherwise push directly
    commit = upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update tiny models for {arch_name}",
        commit_description=f"Upload tiny models for {arch_name}",
        create_pr=create_pr,
        token=token,
    )

    msg = f"PR open in {commit.pr_url}." if create_pr else f"Tiny models {arch_name} pushed to {repo_id}."
    logger.warning(msg)


def build_composite_models(config_class, output_dir):
    import tempfile

    from transformers import (
        BertConfig,
        BertLMHeadModel,
        BertModel,
        BertTokenizer,
        BertTokenizerFast,
        EncoderDecoderModel,
        GPT2Config,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        GPT2TokenizerFast,
        SpeechEncoderDecoderModel,
        VisionEncoderDecoderModel,
        VisionTextDualEncoderModel,
        ViTConfig,
        ViTModel,
        Wav2Vec2Config,
        Wav2Vec2Model,
        Wav2Vec2Processor,
    )

    # These will be removed at the end if they are empty
    result = {"error": None, "warnings": []}

    if config_class.model_type == "encoder-decoder":
        encoder_config_class = BertConfig
        decoder_config_class = BertConfig
        encoder_processor = (BertTokenizerFast, BertTokenizer)
        decoder_processor = (BertTokenizerFast, BertTokenizer)
        encoder_class = BertModel
        decoder_class = BertLMHeadModel
        model_class = EncoderDecoderModel
    elif config_class.model_type == "vision-encoder-decoder":
        encoder_config_class = ViTConfig
        decoder_config_class = GPT2Config
        encoder_processor = (ViTImageProcessor,)
        decoder_processor = (GPT2TokenizerFast, GPT2Tokenizer)
        encoder_class = ViTModel
        decoder_class = GPT2LMHeadModel
        model_class = VisionEncoderDecoderModel
    elif config_class.model_type == "speech-encoder-decoder":
        encoder_config_class = Wav2Vec2Config
        decoder_config_class = BertConfig
        encoder_processor = (Wav2Vec2Processor,)
        decoder_processor = (BertTokenizerFast, BertTokenizer)
        encoder_class = Wav2Vec2Model
        decoder_class = BertLMHeadModel
        model_class = SpeechEncoderDecoderModel
    elif config_class.model_type == "vision-text-dual-encoder":
        # Not encoder-decoder, but encoder-encoder. We just keep the same name as above to make code easier
        encoder_config_class = ViTConfig
        decoder_config_class = BertConfig
        encoder_processor = (ViTImageProcessor,)
        decoder_processor = (BertTokenizerFast, BertTokenizer)
        encoder_class = ViTModel
        decoder_class = BertModel
        model_class = VisionTextDualEncoderModel

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # build encoder
            models_to_create = {"processor": encoder_processor, "pytorch": (encoder_class,)}
            encoder_output_dir = os.path.join(tmpdir, "encoder")
            build(encoder_config_class, models_to_create, encoder_output_dir)

            # build decoder
            models_to_create = {"processor": decoder_processor, "pytorch": (decoder_class,)}
            decoder_output_dir = os.path.join(tmpdir, "decoder")
            build(decoder_config_class, models_to_create, decoder_output_dir)

            # build encoder-decoder
            encoder_path = os.path.join(encoder_output_dir, encoder_class.__name__)
            decoder_path = os.path.join(decoder_output_dir, decoder_class.__name__)

            if config_class.model_type != "vision-text-dual-encoder":
                # Specify these explicitly for encoder-decoder like models, but not for `vision-text-dual-encoder` as it
                # has no decoder.
                decoder_config = decoder_config_class.from_pretrained(decoder_path)
                decoder_config.is_decoder = True
                decoder_config.add_cross_attention = True
                model = model_class.from_encoder_decoder_pretrained(
                    encoder_path,
                    decoder_path,
                    decoder_config=decoder_config,
                )
            elif config_class.model_type == "vision-text-dual-encoder":
                model = model_class.from_vision_text_pretrained(encoder_path, decoder_path)

            model_path = os.path.join(
                output_dir,
                f"{model_class.__name__}-{encoder_config_class.model_type}-{decoder_config_class.model_type}",
            )
            model.save_pretrained(model_path)

            # copy the processors
            encoder_processor_path = os.path.join(encoder_output_dir, "processors")
            decoder_processor_path = os.path.join(decoder_output_dir, "processors")
            if os.path.isdir(encoder_processor_path):
                shutil.copytree(encoder_processor_path, model_path, dirs_exist_ok=True)
            if os.path.isdir(decoder_processor_path):
                shutil.copytree(decoder_processor_path, model_path, dirs_exist_ok=True)

            # fill `result`
            result["processor"] = {x.__name__: x.__name__ for x in encoder_processor + decoder_processor}

            result["pytorch"] = {model_class.__name__: {"model": model_class.__name__, "checkpoint": model_path}}

        except Exception:
            result["error"] = (
                f"Failed to build models for {config_class.__name__}.",
                traceback.format_exc(),
            )

    if not result["error"]:
        del result["error"]
    if not result["warnings"]:
        del result["warnings"]

    return result


def get_token_id_from_tokenizer(token_id_name, tokenizer, original_token_id):
    """Use `tokenizer` to get the values of `bos_token_id`, `eos_token_ids`, etc.

    The argument `token_id_name` should be a string ending with `_token_id`, and `original_token_id` should be an
    integer that will be return if `tokenizer` has no token corresponding to `token_id_name`.
    """

    token_id = original_token_id

    if not token_id_name.endswith("_token_id"):
        raise ValueError(f"`token_id_name` is {token_id_name}, which doesn't end with `_token_id`!")

    token = getattr(tokenizer, token_id_name.replace("_token_id", "_token"), None)
    if token is not None:
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            token_id = tokenizer._convert_token_to_id_with_added_voc(token)
        else:
            token_id = tokenizer._convert_token_to_id(token)

    return token_id


def get_config_overrides(config_class, processors):
    # `Bark` configuration is too special. Let's just not handle this for now.
    if config_class.__name__ == "BarkConfig":
        return {}

    config_overrides = {}

    # Check if there is any tokenizer (prefer fast version if any)
    tokenizer = None
    for processor in processors:
        if isinstance(processor, PreTrainedTokenizerFast):
            tokenizer = processor
            break
        elif isinstance(processor, PythonBackend):
            tokenizer = processor

    if tokenizer is None:
        return config_overrides

    # Get some properties of the (already converted) tokenizer (smaller vocab size, special token ids, etc.)
    # We use `len(tokenizer)` instead of `tokenizer.vocab_size` to avoid potential issues for tokenizers with non-empty
    # `added_tokens_encoder`. One example is the `DebertaV2Tokenizer` where the mask token is the extra token.
    vocab_size = len(tokenizer)

    # The original checkpoint has length `35998`, but it doesn't have ids `30400` and `30514` but instead `35998` and
    # `35999`.
    if config_class.__name__ == "GPTSanJapaneseConfig":
        vocab_size += 2

    config_overrides["vocab_size"] = vocab_size

    # Used to create a new model tester with `tokenizer.vocab_size` in order to get the (updated) special token ids.
    model_tester_kwargs = {"vocab_size": vocab_size}
    # `FSMTModelTester` accepts `src_vocab_size` and `tgt_vocab_size` but not `vocab_size`.
    if config_class.__name__ == "FSMTConfig":
        del model_tester_kwargs["vocab_size"]
        model_tester_kwargs["src_vocab_size"] = tokenizer.src_vocab_size
        model_tester_kwargs["tgt_vocab_size"] = tokenizer.tgt_vocab_size

    _tiny_config = get_tiny_config(config_class, **model_tester_kwargs)

    # handle the possibility of `text_config` inside `_tiny_config` for clip-like models (`owlvit`, `groupvit`, etc.)
    if hasattr(_tiny_config, "text_config"):
        _tiny_config = _tiny_config.text_config

    # Collect values of some special token ids
    for attr in dir(_tiny_config):
        if attr.endswith("_token_id"):
            token_id = getattr(_tiny_config, attr)
            if token_id is not None:
                # Using the token id values from `tokenizer` instead of from `_tiny_config`.
                token_id = get_token_id_from_tokenizer(attr, tokenizer, original_token_id=token_id)
                config_overrides[attr] = token_id

    if config_class.__name__ == "FSMTConfig":
        config_overrides["src_vocab_size"] = tokenizer.src_vocab_size
        config_overrides["tgt_vocab_size"] = tokenizer.tgt_vocab_size
        # `FSMTConfig` has `DecoderConfig` as `decoder` attribute.
        config_overrides["decoder"] = configuration_fsmt.DecoderConfig(
            vocab_size=tokenizer.tgt_vocab_size, bos_token_id=config_overrides["eos_token_id"]
        )

    # Marian failed to convert the tokenzier, and has `'vocab_size': 58101` and `'pad_token_id': 58100`.
    # which gives `Padding_idx must be within num_embeddings`
    if config_class.__name__ == "MarianConfig":
        config_overrides["decoder_vocab_size"] = config_overrides["vocab_size"]

    return config_overrides


def build(config_class, models_to_create, output_dir):
    """Create all models for a certain model type.

    Args:
        config_class (`PreTrainedConfig`):
            A subclass of `PreTrainedConfig` that is used to determine `models_to_create`.
        models_to_create (`dict`):
            A dictionary containing the processor/model classes that we want to create the instances. These models are
            of the same model type which is associated to `config_class`.
        output_dir (`str`):
            The directory to save all the checkpoints. Each model architecture will be saved in a subdirectory under
            it.
    """
    if data["training_ds"] is None or data["testing_ds"] is None:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        data["training_ds"] = ds["train"]
        data["testing_ds"] = ds["test"]

    if config_class.model_type in [
        "encoder-decoder",
        "vision-encoder-decoder",
        "speech-encoder-decoder",
        "vision-text-dual-encoder",
    ]:
        return build_composite_models(config_class, output_dir)

    result = {k: {} for k in models_to_create}

    # These will be removed at the end if they are empty
    result["error"] = None
    result["warnings"] = []

    # Build processors
    processor_classes = models_to_create["processor"]

    # AutoTokenizer can't load from hub repo ...
    if config_class.__name__ in ["FastSpeech2ConformerWithHifiGanConfig"]:
        from transformers import FastSpeech2ConformerTokenizer
        processor_classes = (FastSpeech2ConformerTokenizer,) + processor_classes

    if len(processor_classes) == 0:
        error = f"No processor class could be found in {config_class.__name__}."
        fill_result_with_error(result, error, None, models_to_create)
        logger.error(result["error"][0])
        processor_names = [p.__name__ if not isinstance(p, str) else p for p in result["processor"]]
        result["processor"] = {p:p for p in processor_names}

        return result

    traces = []
    errors = []
    # breakpoint()
    for processor_class in processor_classes:
        try:
            # breakpoint()
            processor = build_processor(config_class, processor_class, allow_no_checkpoint=True)
            if processor is not None:
                if type(processor) not in result["processor"]:
                    # breakpoint()
                    result["processor"][type(processor)] = processor
        except Exception:
            # breakpoint()
            error = f"Failed to build processor for {processor_class.__name__}."
            trace = traceback.format_exc()
            errors.append(error)
            traces.append(trace)
            # fill_result_with_error(result, error, trace, models_to_create)
            logger.error((error, trace))
            # TODO: add trace and error anyway?
            # Let's return all what we could build
            # return result


    # TODO: We might get some errors while still having some processors!
    if len(errors) > 0:
        error = "\n".join(errors)
        trace = "\n".join(traces)
        fill_result_with_error(result, error, trace, models_to_create)

    if len(result["processor"]) == 0:
        # TODO: Some models use NO processor (and no processor files exist on their hub repos.)
        if config_class.__name__ not in ["PatchTSMixerConfig", "PatchTSTConfig", "TimesFmConfig", "TimmBackboneConfig", "TimmWrapperConfig", "VitDetConfig", "AutoformerConfig"]:
            # breakpoint()
            error = f"No processor could be built for {config_class.__name__}."
            fill_result_with_error(result, error, None, models_to_create)
            logger.error(result["error"][0])
            processor_names = [p.__name__ if not isinstance(p, str) else p for p in result["processor"]]
            result["processor"] = {p: p for p in processor_names}
            return result

    # breakpoint()
    try:
        tiny_config = get_tiny_config(config_class)
        # breakpoint()
    except Exception as e:
        # breakpoint()
        error = f"Failed to get tiny config for {config_class.__name__}: {e}"
        trace = traceback.format_exc()
        fill_result_with_error(result, error, trace, models_to_create)
        logger.error(result["error"][0])
        processor_names = [p.__name__ if not isinstance(p, str) else p for p in result["processor"]]
        result["processor"] = {p: p for p in processor_names}
        return result

    # Convert the processors (reduce vocabulary size, smaller image size, etc.)
    processors = list(result["processor"].values())
    processor_output_folder = os.path.join(output_dir, "processors")
    # breakpoint()
    try:
        # breakpoint()
        processors = convert_processors(processors, tiny_config, processor_output_folder, result)
    except Exception:
        error = "Failed to convert the processors."
        trace = traceback.format_exc()
        result["warnings"].append((error, trace))


    # # TODO: if we don't call `convert_processors`, we will need to save here.
    # #   (some conversion might be very slow)
    # processors = [p for p in processors if p is not None]
    # for p in processors:
    #     p.save_pretrained(processor_output_folder)


    if len(processors) == 0:
        # breakpoint()
        # TODO: Some models use NO processor (and no processor files exist on their hub repos.)
        if config_class.__name__ not in ["PatchTSMixerConfig", "PatchTSTConfig", "TimesFmConfig", "TimmBackboneConfig", "TimmWrapperConfig", "VitDetConfig", "AutoformerConfig"]:
            error = f"No processor is returned by `convert_processors` for {config_class.__name__}."
            fill_result_with_error(result, error, None, models_to_create)
            logger.error(result["error"][0])
            processor_names = [p.__name__ if not isinstance(p, str) else p for p in result["processor"]]
            result["processor"] = {p: p for p in processor_names}
            return result

    try:
        config_overrides = get_config_overrides(config_class, processors)
        # breakpoint()
    except Exception as e:
        error = f"Failure occurs while calling `get_config_overrides`: {e}"
        trace = traceback.format_exc()
        fill_result_with_error(result, error, trace, models_to_create)
        logger.error(result["error"][0])
        processor_names = [p.__name__ if not isinstance(p, str) else p for p in result["processor"]]
        result["processor"] = {p: p for p in processor_names}
        return result

    # Just for us to see this easily in the report
    if "vocab_size" in config_overrides:
        result["vocab_size"] = config_overrides["vocab_size"]

    # Update attributes that `vocab_size` involves
    for k, v in config_overrides.items():
        if hasattr(tiny_config, k):
            setattr(tiny_config, k, v)
        # So far, we only have to deal with `text_config`, as `config_overrides` contains text-related attributes only.
        # `FuyuConfig` saves data under both FuyuConfig and its `text_config`. This is not good, but let's just update
        # every involved fields to avoid potential failure.
        if (
            hasattr(tiny_config, "text_config")
            and tiny_config.text_config is not None
            and hasattr(tiny_config.text_config, k)
        ):
            setattr(tiny_config.text_config, k, v)
            # If `text_config_dict` exists, we need to update its value here too in order to # make
            # `save_pretrained -> from_pretrained` work.
            if hasattr(tiny_config, "text_config_dict"):
                tiny_config.text_config_dict[k] = v

    if result["warnings"]:
        logger.warning(result["warnings"][0][0])

    # breakpoint()
    # update `result["processor"]`
    result["processor"] = {type(p).__name__: p.__class__.__name__ for p in processors}

    # breakpoint()
    for pytorch_arch in models_to_create["pytorch"]:
        result["pytorch"][pytorch_arch.__name__] = {}
        error = None
        try:
            # breakpoint()

            used_tiny_config = tiny_config

            # TODO: Some model_type will include multiple `pytorch_arch` but they might actually have different `self.config_class`
            #   (e.g. Qwen3_5Config from qwen3_5, and `Qwen3_5ForCausalLM`
            #   Let's first try to get the component maybe
            # breakpoint()
            if pytorch_arch.config_class != config_class:
                used_tiny_config = _get_exact_config(tiny_config, pytorch_arch.config_class)

            # TODO: If we can't get the exact config, let's skip to avoid issue
            # TODO: Maybe add as an error info
            if pytorch_arch.config_class != used_tiny_config.__class__:
                print(f"Skip `{pytorch_arch.__name__}`: its config class is {pytorch_arch.config_class} != {used_tiny_config.__class__} Oh la la!!!")
                del result["pytorch"][pytorch_arch.__name__]
                continue

            # breakpoint()
            model = build_model(pytorch_arch, used_tiny_config, output_dir=output_dir)
        except Exception as e:

            # TODO: hacky way to make `T5GemmaEncoderModel` work
            if pytorch_arch.__name__ == "T5GemmaEncoderModel":
                _tiny_config = copy.deepcopy(tiny_config)
                _tiny_config.is_encoder_decoder = False
                model = build_model(pytorch_arch, _tiny_config, output_dir=output_dir)
            else:
                model = None
                error = f"Failed to create the pytorch model for {pytorch_arch}: {e}"
                trace = traceback.format_exc()

        result["pytorch"][pytorch_arch.__name__]["model"] = model.__class__.__name__ if model is not None else None
        result["pytorch"][pytorch_arch.__name__]["checkpoint"] = (
            get_checkpoint_dir(output_dir, pytorch_arch) if model is not None else None
        )
        if error is not None:
            result["pytorch"][pytorch_arch.__name__]["error"] = (error, trace)
            logger.error(f"{pytorch_arch.__name__}: {error}")

    if not result["error"]:
        del result["error"]
    if not result["warnings"]:
        del result["warnings"]

    return result


def build_tiny_model_summary(results, organization=None, token=None):
    """Build a summary: a dictionary of the form
    {
      model architecture name:
        {
          "tokenizer_classes": [...],
          "processor_classes": [...],
          "model_classes": [...],
        }
      ..
    }
    """
    # breakpoint()
    tiny_model_summary = {}
    for config_name in results:
        try:
            processors = [key for key, value in results[config_name]["processor"].items()]
            # breakpoint()
            # TODO: we update `fill_result_with_error`: at the end, with the cond `if len(result["processor"]) == 0`
            #   But sometimes, in `def build`, we can't reach `result["processor"] = {type(p).__name__: p.__class__.__name__ for p in processors}`
            #   (i.e. some other errors occur, like `Sam2VideoConfig`), and we need convert `results[config_name]["processor"]` to avid failure!
            #   (for sam2_video, the error is `"Failed to get tiny config for Sam2VideoConfig: Tiny config not created for sam2_video - no model tester is found in the testing module.`)
            processors = [p.__name__ if not isinstance(p, str) else p for p in processors]
            results[config_name]["processor"] = {x: x for x in processors}
        except:
            # This happens for `VisionEncoderDecoderConfig` and `SpeechEncoderDecoderConfig`.
            # Not a prority however.
            print(config_name)
            print(results[config_name])
            print("******************************")
        # breakpoint()
        tokenizer_classes = sorted([x for x in processors if x.endswith(("TokenizerFast", "Tokenizer", "TokenizersBackend'"))])
        processor_classes = sorted([x for x in processors if x not in tokenizer_classes])

        if "pytorch" not in results[config_name]:
            continue
        for arch_name in results[config_name]["pytorch"]:
            model_classes = [arch_name]
            base_arch_name = arch_name
            # tiny model is not created for `arch_name`
            if results[config_name]["pytorch"][arch_name]["model"] is None:
                model_classes = []
            if base_arch_name not in tiny_model_summary:
                tiny_model_summary[base_arch_name] = {}
            tiny_model_summary[base_arch_name].update(
                {
                    "tokenizer_classes": tokenizer_classes,
                    "processor_classes": processor_classes,
                }
            )
            tiny_model_summary[base_arch_name]["model_classes"] = sorted(
                tiny_model_summary[base_arch_name].get("model_classes", []) + model_classes
            )
            if organization is not None:
                repo_name = f"tiny-random-{base_arch_name}"
                # composite models' checkpoints have more precise repo. names on the Hub.
                if base_arch_name in COMPOSITE_MODELS:
                    repo_name = f"tiny-random-{COMPOSITE_MODELS[base_arch_name]}"
                repo_id = f"{organization}/{repo_name}"
                try:
                    commit_hash = hf_api.repo_info(repo_id, token=token).sha
                except Exception:
                    # The directory is not created, but processor(s) is/are included in `results`.
                    logger.warning(f"Failed to get information for {repo_id}.\n{traceback.format_exc()}")
                    del tiny_model_summary[base_arch_name]
                    continue
                tiny_model_summary[base_arch_name]["sha"] = commit_hash

    return tiny_model_summary


def build_failed_report(results, include_warning=True):
    failed_results = {}
    for config_name in results:
        if "error" in results[config_name]:
            if config_name not in failed_results:
                failed_results[config_name] = {}
            failed_results[config_name] = {"error": results[config_name]["error"]}

        if include_warning and "warnings" in results[config_name]:
            if config_name not in failed_results:
                failed_results[config_name] = {}
            failed_results[config_name]["warnings"] = results[config_name]["warnings"]

        if "pytorch" not in results[config_name]:
            continue
        for arch_name in results[config_name]["pytorch"]:
            if "error" in results[config_name]["pytorch"][arch_name]:
                if config_name not in failed_results:
                    failed_results[config_name] = {}
                if "pytorch" not in failed_results[config_name]:
                    failed_results[config_name]["pytorch"] = {}
                if arch_name not in failed_results[config_name]["pytorch"]:
                    failed_results[config_name]["pytorch"][arch_name] = {}
                error = results[config_name]["pytorch"][arch_name]["error"]
                failed_results[config_name]["pytorch"][arch_name]["error"] = error

    return failed_results


def build_simple_report(results):
    text = ""
    failed_text = ""
    for config_name in results:
        if "pytorch" not in results[config_name]:
            continue
        for arch_name in results[config_name]["pytorch"]:
            if "error" in results[config_name]["pytorch"][arch_name]:
                result = results[config_name]["pytorch"][arch_name]["error"]
                failed_text += f"{arch_name}: {result[0]}\n"
            else:
                result = ("OK",)
            text += f"{arch_name}: {result[0]}\n"

    return text, failed_text


def update_tiny_model_summary_file(report_path):
    with open(os.path.join(report_path, "tiny_model_summary.json")) as fp:
        new_data = json.load(fp)
    with open("tests/utils/tiny_model_summary.json") as fp:
        data = json.load(fp)
    for key, value in new_data.items():
        if key not in data:
            data[key] = value
        else:
            for attr in ["tokenizer_classes", "processor_classes", "model_classes"]:
                # we might get duplication here. We will remove them below when creating `updated_data`.
                data[key][attr].extend(value[attr])
            new_sha = value.get("sha", None)
            if new_sha is not None:
                data[key]["sha"] = new_sha

    updated_data = {}
    for key in sorted(data.keys()):
        updated_data[key] = {}
        for attr, value in data[key].items():
            # deduplication and sort
            updated_data[key][attr] = sorted(set(value)) if attr != "sha" else value

    with open(os.path.join(report_path, "updated_tiny_model_summary.json"), "w") as fp:
        json.dump(updated_data, fp, indent=4, ensure_ascii=False)


def create_tiny_models(
    output_path,
    all,
    model_types,
    models_to_skip,
    no_check,
    upload,
    organization,
    token,
    num_workers=1,
):
    clone_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if os.getcwd() != clone_path:
        raise ValueError(f"This script should be run from the root of the clone of `transformers` {clone_path}")

    report_path = os.path.join(output_path, "reports")
    os.makedirs(report_path, exist_ok=True)

    _pytorch_arch_mappings = [
        x
        for x in dir(transformers_module)
        if x.startswith("MODEL_") and x.endswith("_MAPPING") and x != "MODEL_NAMES_MAPPING"
    ]

    pytorch_arch_mappings = [getattr(transformers_module, x) for x in _pytorch_arch_mappings]

    config_classes = CONFIG_MAPPING.values()
    if not all:
        config_classes = [CONFIG_MAPPING[model_type] for model_type in model_types]

    # config_classes = [x for x in config_classes if x.__name__ in ["JanusConfig", "Emu3Config", "ClvpConfig", "BarkConfig", "FastSpeech2ConformerWithHifiGanConfig", "FastSpeech2ConformerConfig", "Pop2PianoConfig"]]
    # TODO: we should add information to the reports instead of skip them
    config_classes = [x for x in config_classes if x.__name__ not in no_model_tester_at_all]
    config_classes = [x for x in config_classes if x.__name__ not in configs_requiring_too_exotic_dependency]
    config_classes = [x for x in config_classes if x.__name__ not in deprecated_models]
    config_classes = [x for x in config_classes if x.__name__ not in config_without_meaningful_model_class]

    config_classes = config_classes[178:179]

    # import random
    # for i in range(100):
    #     random.shuffle(config_classes)

    # mamba = {"BambaConfig", "FalconMambaConfig", "GraniteMoeHybridConfig", "JambaConfig", "MambaConfig", "Mamba2Config", ""}
    # config_classes = [x for x in config_classes if x.__name__ in mamba]

    # for x in config_classes:
    #     if x.__name__ == "Pop2PianoConfig":
    #         break
    #
    # config_classes = config_classes[:1]
    # config_classes += [x]

    # A map from config classes to tuples of processors (tokenizer, feature extractor, processor) classes
    processor_type_map = {c: get_processor_types_from_config_class(c) for c in config_classes}

    to_create = {}
    for c in config_classes:
        processors = processor_type_map[c]
        models = get_architectures_from_config_class(c, pytorch_arch_mappings, models_to_skip)
        if len(models) > 0:
            to_create[c] = {"processor": processors, "pytorch": models}

    results = {}
    if num_workers <= 1:
        for c, models_to_create in list(to_create.items()):
            print(f"Create models for {c.__name__} ...")
            result = build(c, models_to_create, output_dir=os.path.join(output_path, c.model_type))
            results[c.__name__] = result
            print("=" * 40)
    else:
        all_build_args = []
        for c, models_to_create in list(to_create.items()):
            all_build_args.append((c, models_to_create, os.path.join(output_path, c.model_type)))
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(build, all_build_args)
            results = {build_args[0].__name__: result for build_args, result in zip(all_build_args, results)}

    print(results)

    if upload:
        if organization is None:
            raise ValueError("The argument `organization` could not be `None`. No model is uploaded")

        to_upload = []
        for model_type in os.listdir(output_path):
            # This is the directory containing the reports
            if model_type == "reports":
                continue
            for arch in os.listdir(os.path.join(output_path, model_type)):
                if arch == "processors":
                    continue
                to_upload.append(os.path.join(output_path, model_type, arch))
        to_upload = sorted(to_upload)

        upload_results = {}
        if len(to_upload) > 0:
            for model_dir in to_upload:
                try:
                    upload_model(model_dir, organization, token)
                except Exception as e:
                    error = f"Failed to upload {model_dir}. {e.__class__.__name__}: {e}"
                    logger.error(error)
                    upload_results[model_dir] = error

        with open(os.path.join(report_path, "failed_uploads.json"), "w") as fp:
            json.dump(upload_results, fp, indent=4)

    # Build the tiny model summary file. The `tokenizer_classes` and `processor_classes` could be both empty lists.
    # When using the items in this file to update the file `tests/utils/tiny_model_summary.json`, the model
    # architectures with `tokenizer_classes` and `processor_classes` being both empty should **NOT** be added to
    # `tests/utils/tiny_model_summary.json`.



    tiny_model_summary = build_tiny_model_summary(results, organization=organization, token=token)
    with open(os.path.join(report_path, "tiny_model_summary.json"), "w") as fp:
        json.dump(tiny_model_summary, fp, indent=4)

    with open(os.path.join(report_path, "tiny_model_creation_report.json"), "w") as fp:
        json.dump(results, fp, indent=4)

    # Build the warning/failure report (json format): same format as the complete `results` except this contains only
    # warnings or errors.
    failed_results = build_failed_report(results)
    with open(os.path.join(report_path, "failed_report.json"), "w") as fp:
        json.dump(failed_results, fp, indent=4)

    simple_report, failed_report = build_simple_report(results)
    # The simplified report: a .txt file with each line of format:
    # {model architecture name}: {OK or error message}
    with open(os.path.join(report_path, "simple_report.txt"), "w") as fp:
        fp.write(simple_report)

    # The simplified failure report: same above except this only contains line with errors
    with open(os.path.join(report_path, "simple_failed_report.txt"), "w") as fp:
        fp.write(failed_report)

    update_tiny_model_summary_file(report_path=os.path.join(output_path, "reports"))


if __name__ == "__main__":
    # This has to be `spawn` to avoid hanging forever!
    multiprocessing.set_start_method("spawn")

    def list_str(values):
        return values.split(",")

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Will create all tiny models.")
    parser.add_argument(
        "--no_check",
        action="store_true",
        help="If set, will not check the validity of architectures. Use with caution.",
    )
    parser.add_argument(
        "-m",
        "--model_types",
        type=list_str,
        help="Comma-separated list of model type(s) from which the tiny models will be created.",
    )
    parser.add_argument(
        "--models_to_skip",
        type=list_str,
        help=(
            "Comma-separated list of model class names(s) from which the tiny models won't be created.\nThis is usually "
            "the list of model classes that have their tiny versions already uploaded to the Hub."
        ),
    )
    parser.add_argument("--upload", action="store_true", help="If to upload the created tiny models to the Hub.")
    parser.add_argument(
        "--organization",
        default=None,
        type=str,
        help="The organization on the Hub to which the tiny models will be uploaded.",
    )
    parser.add_argument(
        "--token", default=None, type=str, help="A valid authentication token for HuggingFace Hub with write access."
    )
    parser.add_argument("output_path", type=Path, help="Path indicating where to store generated model.")
    parser.add_argument("--num_workers", default=1, type=int, help="The number of workers to run.")

    args = parser.parse_args()

    if not args.all and not args.model_types:
        raise ValueError("Please provide at least one model type or pass `--all` to export all architectures.")

    # os.environ["HF_TOKEN"] = args.token

    create_tiny_models(
        args.output_path,
        args.all,
        args.model_types,
        args.models_to_skip,
        args.no_check,
        args.upload,
        args.organization,
        args.token,
        args.num_workers,
    )


# FastSpeech2ConformerConfig --> needs `pip install g2p-en`
# FastSpeech2ConformerTokenizer vs AutTokenizer --> the later can't load from `espnet/fastspeech2_conformer` ??




# "Pop2PianoConfig" Require `pip install essentia==2.1b6.dev1034`
# NemotronConfig ==> can't convert fast tokenizer because `Exception: Unk token `<unk>` not found in the vocabulary`


# track but get large model:
# BarkConfig, ClvpConfig,
#
# Emu3Config,
#
# JanusConfig





# no model tester
#
# EdgeTamVideoConfig
# Llama4Config
# Llama4TextConfig
# Sam2Video
# Sam3TrackerVideo
# Sam3VideoConfig
# ShieldGemma2Config

# TODO!!!
# has model tester, but there is no model tester gives the exact config class for some model classes


# PeAudioVideoConfig : Only deal with PeAudioVideoEncoderConfig and PeAudioVideoEncoder, no model tester
# Qwen3OmniMoeConfig: Only deal with Qwen3OmniMoeThinkerConfig and Qwen3OmniMoeThinkerForConditionalGeneration
# Qwen2_5OmniConfig: Only deal with Qwen2_5OmniThinkerConfig and Qwen2_5OmniThinkerForConditionalGenerationTester


# LayoutLMv2Config: needs detectron2 and there is no `get_config`


# Qwen3_5Config: Deal with both `Qwen3_5TextConfig` and `Qwen3_5Config` but only get the first
# Qwen3_5MoeConfig: Deal with both `Qwen3_5MoeTextConfig` and `Qwen3_5MoeConfig`




# InstructBlipConfig: deal 4 types
# InstructBlipVideoConfig: deal 4 types

# MllamaConfig: deal 2 types
# Gemma3nConfig: deal 3 types
# Gemma3Config: deal 2 types
# VideoLlama3Config: deal 3 tyipes

