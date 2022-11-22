# coding=utf-8
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

import argparse
import collections.abc
import copy
import importlib
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from datasets import load_dataset

from check_config_docstrings import get_checkpoint_from_config_class
from huggingface_hub import Repository, create_repo, upload_folder
from transformers import (
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    PROCESSOR_MAPPING,
    TOKENIZER_MAPPING,
    AutoTokenizer,
    LayoutLMv3TokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    logging,
)
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.file_utils import is_tf_available, is_torch_available
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

sys.path.append(".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if not is_torch_available():
    raise ValueError("Please install PyTorch.")

if not is_tf_available():
    raise ValueError("Please install TensorFlow.")

FRAMEWORKS = ["pytorch", "tensorflow"]
INVALID_ARCH = []
TARGET_VOCAB_SIZE = 1024


def get_processor_types_from_config_class(config_class, allowed_mappings=None):
    """Return a tuple of processors for `config_class`.

    We use `tuple` here to include (potentially) both slow & fast tokenizers.
    """
    if allowed_mappings is None:
        allowed_mappings = ["processor", "tokenizer", "feature_extractor"]

    processor_types = ()

    # Check first if a model has `ProcessorMixin`. Otherwise, check if it has tokenizers or a feature extractor.
    if config_class in PROCESSOR_MAPPING and "processor" in allowed_mappings:
        processor_types = PROCESSOR_MAPPING[config_class]
    elif config_class in TOKENIZER_MAPPING and "tokenizer" in allowed_mappings:
        processor_types = TOKENIZER_MAPPING[config_class]
    elif config_class in FEATURE_EXTRACTOR_MAPPING and "feature_extractor" in allowed_mappings:
        processor_types = FEATURE_EXTRACTOR_MAPPING[config_class]
    else:
        # Some configurations have no processor at all. For example, generic composite models like
        # `EncoderDecoderModel` is used for any (compatible) text models. Also, `DecisionTransformer` doesn't
        # require any processor.
        pass

    # make a uniform return type
    if not isinstance(processor_types, collections.abc.Sequence):
        processor_types = (processor_types,)
    else:
        processor_types = tuple(processor_types)

    # We might get `None` for some tokenizers - remove them here.
    processor_types = tuple(p for p in processor_types if p is not None)

    return processor_types


def get_architectures_from_config_class(config_class, arch_mappings):
    """Return a tuple of all possible architectures attributed to a configuration class `config_class`.

    For example, BertConfig -> [BertModel, BertForMaskedLM, ..., BertForQuestionAnswering].
    """
    # A model architecture could appear in several mappings. For example, `BartForConditionalGeneration` is in
    #   - MODEL_FOR_PRETRAINING_MAPPING_NAMES
    #   - MODEL_WITH_LM_HEAD_MAPPING_NAMES
    #   - MODEL_FOR_MASKED_LM_MAPPING_NAMES
    #   - MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    # We avoid the duplication.
    architectures = set()

    for mapping in arch_mappings:
        if config_class in mapping:
            models = mapping[config_class]
            models = tuple(models) if isinstance(models, collections.abc.Sequence) else (models,)
            for model in models:
                if model.__name__ not in unexportable_model_architectures:
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
    for postfix in ["TokenizerFast", "Tokenizer", "ImageProcessor", "FeatureExtractor", "Processor"]:
        processor_prefix = processor_prefix.replace(postfix, "")

    # `Wav2Vec2CTCTokenizer` -> `Wav2Vec2Config`
    if processor_prefix == "Wav2Vec2CTC":
        processor_prefix = "Wav2Vec2"

    # Find the new configuration class
    new_config_name = f"{processor_prefix}Config"
    new_config_class = getattr(transformers_module, new_config_name)

    return new_config_class


def build_processor(config_class, processor_class):
    """Create a processor for `processor_class`.

    If a processor is not able to be built with the original arguments, this method tries to change the arguments and
    call itself recursively, by inferring a new `config_class` or a new `processor_class` from another one, in order to
    find a checkpoint containing the necessary files to build a processor.

    The processor is not saved here. Instead, it will be saved in `convert_processors` after further changes in
    `convert_processors`. For each model architecture`, a copy will be created and saved along the built model.
    """
    # Currently, this solely uses the docstring in the source file of `config_class` to find a checkpoint.
    checkpoint = get_checkpoint_from_config_class(config_class)

    if checkpoint is None:
        # try to get the checkpoint from the config class for `processor_class`.
        # This helps cases like `XCLIPConfig` and `VideoMAEFeatureExtractor` to find a checkpoint from `VideoMAEConfig`.
        config_class_from_processor_class = get_config_class_from_processor_class(processor_class)
        checkpoint = get_checkpoint_from_config_class(config_class_from_processor_class)

    processor = None
    try:
        processor = processor_class.from_pretrained(checkpoint)
    except Exception as e:
        logger.error(e)
        pass

    # Try to get a new processor class from checkpoint. This is helpful for a checkpoint without necessary file to load
    # processor while `processor_class` is an Auto class. For example, `sew` has `Wav2Vec2Processor` in
    # `PROCESSOR_MAPPING_NAMES`, its `tokenizer_class` is `AutoTokenizer`, and the checkpoint
    # `https://huggingface.co/asapp/sew-tiny-100k` has no tokenizer file, but we can get
    # `tokenizer_class: Wav2Vec2CTCTokenizer` from the config file. (The new processor class won't be able to load from
    # `checkpoint`, but it helps this recursive method to find a way to build a processor).
    if (
        processor is None
        and checkpoint is not None
        and issubclass(processor_class, (PreTrainedTokenizerBase, AutoTokenizer))
    ):
        try:
            config = AutoConfig.from_pretrained(checkpoint)
        except Exception as e:
            logger.error(e)
            config = None
        if config is not None:
            if not isinstance(config, config_class):
                raise ValueError(
                    f"`config` (which is of type {config.__class__.__name__}) should be an instance of `config_class`"
                    f" ({config_class.__name__})!"
                )
            tokenizer_class = config.tokenizer_class
            new_processor_class = None
            if tokenizer_class is not None:
                new_processor_class = getattr(transformers_module, tokenizer_class)
                if new_processor_class != processor_class:
                    processor = build_processor(config_class, new_processor_class)
            # If `tokenizer_class` is not specified in `config`, let's use `config` to get the process class via auto
            # mappings, but only allow the tokenizer mapping being used. This is to make `Wav2Vec2Conformer` build
            if processor is None:
                new_processor_classes = get_processor_types_from_config_class(
                    config.__class__, allowed_mappings=["tokenizer"]
                )
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
                    for x in new_processor_classes:
                        if x.__name__.endswith("Fast"):
                            new_processor_class = x
                            break
                    processor = build_processor(config_class, new_processor_class)

    if processor is None:
        # Try to build each component (tokenizer & feature extractor) of a `ProcessorMixin`.
        if issubclass(processor_class, ProcessorMixin):
            attrs = {}
            for attr_name in processor_class.attributes:
                attrs[attr_name] = []
                # This could be a tuple (for tokenizers). For example, `CLIPProcessor` has
                #   - feature_extractor_class = "CLIPFeatureExtractor"
                #   - tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")
                attr_class_names = getattr(processor_class, f"{attr_name}_class")
                if not isinstance(attr_class_names, tuple):
                    attr_class_names = (attr_class_names,)

                for name in attr_class_names:
                    attr_class = getattr(transformers_module, name)
                    attr = build_processor(config_class, attr_class)
                    if attr is not None:
                        attrs[attr_name].append(attr)

            # try to build a `ProcessorMixin`, so we can return a single value
            if all(len(v) > 0 for v in attrs.values()):
                try:
                    processor = processor_class(**{k: v[0] for k, v in attrs.items()})
                except Exception as e:
                    logger.error(e)
                    pass
        else:
            # `checkpoint` might lack some file(s) to load a processor. For example, `facebook/hubert-base-ls960`
            # has no tokenizer file to load `Wav2Vec2CTCTokenizer`. In this case, we try to build a processor
            # with the configuration class (for example, `Wav2Vec2Config`) corresponding to `processor_class`.
            config_class_from_processor_class = get_config_class_from_processor_class(processor_class)
            if config_class_from_processor_class != config_class:
                processor = build_processor(config_class_from_processor_class, processor_class)

    # validation
    if processor is not None:
        if not (isinstance(processor, processor_class) or processor_class.__name__.startswith("Auto")):
            raise ValueError(
                f"`processor` (which is of type {processor.__class__.__name__}) should be an instance of"
                f" {processor_class.__name__} or an Auto class!"
            )

    return processor


def get_tiny_config(config_class, **model_tester_kwargs):
    """Retrieve a tiny configuration from `config_class` using each model's `ModelTester`.

    Args:
        config_class: Subclass of `PreTrainedConfig`.

    Returns:
        An instance of `config_class` with tiny hyperparameters
    """
    model_type = config_class.model_type

    # For model type like `data2vec-vision` and `donut-swin`, we can't get the config/model file name directly via
    # `model_type` as it would be sth. like `configuration_data2vec_vision.py`.
    # A simple way is to use `inspect.getsourcefile(config_class)`.
    config_source_file = inspect.getsourcefile(config_class)
    # The modeling file name without prefix (`modeling_`) and postfix (`.py`)
    modeling_name = config_source_file.split(os.path.sep)[-1].replace("configuration_", "").replace(".py", "")

    try:
        print("Importing", model_type_to_module_name(model_type))
        module_name = model_type_to_module_name(model_type)
        if not modeling_name.startswith(module_name):
            raise ValueError(f"{modeling_name} doesn't start with {module_name}!")
        module = importlib.import_module(f".models.{module_name}.test_modeling_{modeling_name}", package="tests")
        camel_case_model_name = config_class.__name__.split("Config")[0]
        model_tester_class = getattr(module, f"{camel_case_model_name}ModelTester", None)
    except ModuleNotFoundError as e:
        error = f"Tiny config not created for {model_type} - cannot find the testing module from the model name"
        raise ValueError(f"{error}: {e}")

    if model_tester_class is None:
        error = f"Tiny config not created for {model_type} - no model tester is found in the testing module"
        raise ValueError(error)

    # `parent` is an instance of `unittest.TestCase`, but we don't need it here.
    model_tester = model_tester_class(parent=None, **model_tester_kwargs)

    if hasattr(model_tester, "get_pipeline_config"):
        return model_tester.get_pipeline_config()
    elif hasattr(model_tester, "prepare_config_and_inputs"):
        # `PoolFormer` has no `get_config` defined. Furthermore, it's better to use `prepare_config_and_inputs` even if
        # `get_config` is defined, since there might be some extra changes in `prepare_config_and_inputs`.
        return model_tester.prepare_config_and_inputs()[0]
    elif hasattr(model_tester, "get_config"):
        return model_tester.get_config()
    else:
        error = (
            f"Tiny config not created for {model_type} - the model tester {model_tester_class.__name__} lacks"
            " necessary method to create config."
        )
        raise ValueError(error)


def convert_tokenizer(tokenizer_fast: PreTrainedTokenizerFast):

    new_tokenizer = tokenizer_fast.train_new_from_iterator(training_ds["text"], TARGET_VOCAB_SIZE, show_progress=False)

    # Make sure it at least runs
    if not isinstance(new_tokenizer, LayoutLMv3TokenizerFast):
        new_tokenizer(testing_ds["text"])

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

    return feature_extractor


def convert_processors(processors, tiny_config, output_folder, result):
    """Change a processor to work with smaller inputs.

    For tokenizers, we try to reduce their vocabulary size.

    For feature extractor, we use smaller image size or change
    other attributes using the values from `tiny_config`. See `convert_feature_extractor`.

    This method should not fail: we catch the errors and put them in `result["warnings"]` with descriptive messages.
    """

    tokenizers = []
    feature_extractors = []
    for processor in processors:
        if isinstance(processor, PreTrainedTokenizerBase):
            tokenizers.append(processor)
        elif isinstance(processor, BaseImageProcessor):
            feature_extractors.append(processor)
        elif isinstance(processor, FeatureExtractionMixin):
            feature_extractors.append(processor)
        elif isinstance(processor, ProcessorMixin):
            # Currently, we only have these 2 possibilities
            tokenizers.append(processor.tokenizer)
            feature_extractors.append(processor.feature_extractor)

    # check the built processors have the unique type
    num_types = len(set([x.__class__.__name__ for x in feature_extractors]))
    if num_types >= 2:
        raise ValueError(f"`feature_extractors` should contain at most 1 type, but it contains {num_types} types!")
    num_types = len(set([x.__class__.__name__.replace("Fast", "") for x in tokenizers]))
    if num_types >= 2:
        raise ValueError(f"`tokenizers` should contain at most 1 tokenizer type, but it contains {num_types} types!")

    fast_tokenizer = None
    slow_tokenizer = None
    for tokenizer in tokenizers:
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            if fast_tokenizer is None:
                fast_tokenizer = tokenizer
                try:
                    # Wav2Vec2ForCTC , ByT5Tokenizer etc. all are already small enough and have no fast version that can
                    # be retrained
                    if fast_tokenizer.vocab_size > TARGET_VOCAB_SIZE:
                        fast_tokenizer = convert_tokenizer(tokenizer)
                except Exception as e:
                    result["warnings"].append(
                        f"Failed to convert the fast tokenizer for {fast_tokenizer.__class__.__name__}: {e}"
                    )
                    continue
        elif slow_tokenizer is None:
            slow_tokenizer = tokenizer

    # Make sure the fast tokenizer can be saved
    if fast_tokenizer:
        try:
            fast_tokenizer.save_pretrained(output_folder)
        except Exception as e:
            result["warnings"].append(
                f"Failed to save the fast tokenizer for {fast_tokenizer.__class__.__name__}: {e}"
            )
            fast_tokenizer = None

    # Make sure the slow tokenizer (if any) corresponds to the fast version (as it might be converted above)
    if fast_tokenizer:
        try:
            slow_tokenizer = AutoTokenizer.from_pretrained(output_folder, use_fast=False)
        except Exception as e:
            result["warnings"].append(
                f"Failed to load the slow tokenizer saved from {fast_tokenizer.__class__.__name__}: {e}"
            )
            # Let's just keep the fast version
            slow_tokenizer = None

    # If the fast version can't be created and saved, let's use the slow version
    if not fast_tokenizer and slow_tokenizer:
        try:
            slow_tokenizer.save_pretrained(output_folder)
        except Exception as e:
            result["warnings"].append(
                f"Failed to save the slow tokenizer for {slow_tokenizer.__class__.__name__}: {e}"
            )
            slow_tokenizer = None

    # update feature extractors using the tiny config
    try:
        feature_extractors = [convert_feature_extractor(p, tiny_config) for p in feature_extractors]
    except Exception as e:
        result["warnings"].append(f"Failed to convert feature extractors: {e}")
        feature_extractors = []

    processors = [fast_tokenizer, slow_tokenizer] + feature_extractors
    processors = [p for p in processors if p is not None]
    for p in processors:
        p.save_pretrained(output_folder)

    return processors


def get_checkpoint_dir(output_dir, model_arch):
    """Get framework-agnostic architecture name. Used to save all PT/TF/Flax models into the same directory."""

    arch_name = model_arch.__name__
    if arch_name.startswith("TF"):
        arch_name = arch_name[2:]
    elif arch_name.startswith("Flax"):
        arch_name = arch_name[4:]

    return os.path.join(output_dir, arch_name)


def build_model(model_arch, tiny_config, output_dir):
    """Create and save a model for `model_arch`.

    Also copy the set of processors to each model (under the same model type) output folder.
    """

    checkpoint_dir = get_checkpoint_dir(output_dir, model_arch)

    processor_output_dir = os.path.join(output_dir, "processors")
    # copy the (same set of) processors (for a model type) to the model arch. specific folder
    if os.path.isdir(processor_output_dir):
        shutil.copytree(processor_output_dir, checkpoint_dir, dirs_exist_ok=True)

    tiny_config = copy.deepcopy(tiny_config)

    # TODO: check `FlaubertWithLMHeadModel` and `XLMWithLMHeadModel`
    if any([model_arch.__name__.endswith(x) for x in ["ForCausalLM", "LMHeadModel"]]):
        tiny_config.is_encoder_decoder = False
        tiny_config.is_decoder = True

    model = model_arch(config=tiny_config)
    model.save_pretrained(checkpoint_dir)
    model.from_pretrained(checkpoint_dir)

    return model


def fill_result_with_error(result, error, models_to_create):
    """Fill `result` with errors for all target model arch if we can't build processor"""

    result["error"] = error
    for framework in FRAMEWORKS:
        if framework in models_to_create:
            result[framework] = {}
            for model_arch in models_to_create[framework]:
                result[framework][model_arch.__name__] = {"model": None, "checkpoint": None, "error": error}

    result["processor"] = {type(p).__name__: p.__class__.__name__ for p in result["processor"]}


def upload_model(model_dir, organization):
    """Upload the tiny models"""

    arch_name = model_dir.split(os.path.sep)[-1]
    repo_name = f"tiny-random-{arch_name}"

    repo_exist = False
    error = None
    try:
        create_repo(repo_id=repo_name, organization=organization, exist_ok=False, repo_type="model")
    except Exception as e:
        error = e
        if "You already created" in str(e):
            error = None
            logger.warning("Remote repository exists and will be cloned.")
            repo_exist = True
            try:
                create_repo(repo_id=repo_name, organization=organization, exist_ok=True, repo_type="model")
            except Exception as e:
                error = e
    if error is not None:
        raise ValueError(error)

    with tempfile.TemporaryDirectory() as tmpdir:

        repo = Repository(local_dir=tmpdir, clone_from=f"{organization}/{repo_name}")
        repo.git_pull()
        shutil.copytree(model_dir, tmpdir, dirs_exist_ok=True)

        if repo_exist:
            # Open a PR on the existing Hub repo.
            hub_pr_url = upload_folder(
                folder_path=model_dir,
                repo_id=f"{organization}/{repo_name}",
                repo_type="model",
                commit_message=f"Update tiny models for {arch_name}",
                commit_description=f"Upload tiny models for {arch_name}",
                create_pr=True,
            )
            logger.warning(f"PR open in {hub_pr_url}")
        else:
            # Push to Hub repo directly
            repo.git_add(auto_lfs_track=True)
            repo.git_commit(f"Upload tiny models for {arch_name}")
            repo.git_push(blocking=True)  # this prints a progress bar with the upload
            logger.warning(f"Tiny models {arch_name} pushed to {organization}/{repo_name}")


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
        TFEncoderDecoderModel,
        TFVisionEncoderDecoderModel,
        VisionEncoderDecoderModel,
        VisionTextDualEncoderModel,
        ViTConfig,
        ViTFeatureExtractor,
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
        tf_model_class = TFEncoderDecoderModel
    elif config_class.model_type == "vision-encoder-decoder":
        encoder_config_class = ViTConfig
        decoder_config_class = GPT2Config
        encoder_processor = (ViTFeatureExtractor,)
        decoder_processor = (GPT2TokenizerFast, GPT2Tokenizer)
        encoder_class = ViTModel
        decoder_class = GPT2LMHeadModel
        model_class = VisionEncoderDecoderModel
        tf_model_class = TFVisionEncoderDecoderModel
    elif config_class.model_type == "speech-encoder-decoder":
        encoder_config_class = Wav2Vec2Config
        decoder_config_class = BertConfig
        encoder_processor = (Wav2Vec2Processor,)
        decoder_processor = (BertTokenizerFast, BertTokenizer)
        encoder_class = Wav2Vec2Model
        decoder_class = BertLMHeadModel
        model_class = SpeechEncoderDecoderModel
        tf_model_class = None
    elif config_class.model_type == "vision-text-dual-encoder":
        # Not encoder-decoder, but encoder-encoder. We just keep the same name as above to make code easier
        encoder_config_class = ViTConfig
        decoder_config_class = BertConfig
        encoder_processor = (ViTFeatureExtractor,)
        decoder_processor = (BertTokenizerFast, BertTokenizer)
        encoder_class = ViTModel
        decoder_class = BertModel
        model_class = VisionTextDualEncoderModel
        tf_model_class = None

    with tempfile.TemporaryDirectory() as tmpdir:

        try:
            # build encoder
            models_to_create = {"processor": encoder_processor, "pytorch": (encoder_class,), "tensorflow": []}
            encoder_output_dir = os.path.join(tmpdir, "encoder")
            build(encoder_config_class, models_to_create, encoder_output_dir)

            # build decoder
            models_to_create = {"processor": decoder_processor, "pytorch": (decoder_class,), "tensorflow": []}
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

            if tf_model_class is not None:
                model = tf_model_class.from_pretrained(model_path, from_pt=True)
                model.save_pretrained(model_path)

            # copy the processors
            encoder_processor_path = os.path.join(encoder_output_dir, "processors")
            decoder_processor_path = os.path.join(decoder_output_dir, "processors")
            if os.path.isdir(encoder_processor_path):
                shutil.copytree(encoder_processor_path, model_path, dirs_exist_ok=True)
            if os.path.isdir(decoder_processor_path):
                shutil.copytree(decoder_processor_path, model_path, dirs_exist_ok=True)

            # fill `result`
            result["processor"] = tuple(set([x.__name__ for x in encoder_processor + decoder_processor]))

            result["pytorch"] = {model_class.__name__: {"model": model_class.__name__, "checkpoint": model_path}}

            result["tensorflow"] = {}
            if tf_model_class is not None:
                result["tensorflow"] = {
                    tf_model_class.__name__: {"model": tf_model_class.__name__, "checkpoint": model_path}
                }

        except Exception as e:
            result["error"] = f"Failed to build models for {config_class.__name__}: {e}"

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

    config_overrides = {}

    # Check if there is any tokenizer (prefer fast version if any)
    tokenizer = None
    for processor in processors:
        if isinstance(processor, PreTrainedTokenizerFast):
            tokenizer = processor
            break
        elif isinstance(processor, PreTrainedTokenizer):
            tokenizer = processor

    if tokenizer is None:
        return config_overrides

    # Get some properties of the (already converted) tokenizer (smaller vocab size, special token ids, etc.)
    vocab_size = tokenizer.vocab_size
    config_overrides["vocab_size"] = vocab_size

    # Used to create a new model tester with `tokenizer.vocab_size` in order to get the (updated) special token ids.
    model_tester_kwargs = {"vocab_size": vocab_size}
    # CLIP-like models have `text_model_tester` and `vision_model_tester`, and we need to pass `vocab_size` to
    # `text_model_tester` via `text_kwargs`. The same trick is also necessary for `Flava`.
    if config_class.__name__ in ["CLIPConfig", "GroupViTConfig", "OwlViTConfig", "XCLIPConfig", "FlavaConfig"]:
        del model_tester_kwargs["vocab_size"]
        model_tester_kwargs["text_kwargs"] = {"vocab_size": vocab_size}
    # `FSMTModelTester` accepts `src_vocab_size` and `tgt_vocab_size` but not `vocab_size`.
    elif config_class.__name__ == "FSMTConfig":
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

    return config_overrides


def build(config_class, models_to_create, output_dir):
    """Create all models for a certain model type.

    Args:
        config_class (`PretrainedConfig`):
            A subclass of `PretrainedConfig` that is used to determine `models_to_create`.
        models_to_create (`dict`):
            A dictionary containing the processor/model classes that we want to create the instances. These models are
            of the same model type which is associated to `config_class`.
        output_dir (`str`):
            The directory to save all the checkpoints. Each model architecture will be saved in a subdirectory under
            it. Models in different frameworks with the same architecture will be saved in the same subdirectory.
    """

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

    if len(processor_classes) == 0:
        error = f"No processor class could be found in {config_class.__name__}."
        fill_result_with_error(result, error, models_to_create)
        logger.error(result["error"])
        return result

    for processor_class in processor_classes:
        try:
            processor = build_processor(config_class, processor_class)
            if processor is not None:
                result["processor"][processor_class] = processor
        except Exception as e:
            error = f"Failed to build processor for {processor_class.__name__}: {e}"
            fill_result_with_error(result, error, models_to_create)
            logger.error(result["error"])
            return result

    if len(result["processor"]) == 0:
        error = f"No processor could be built for {config_class.__name__}."
        fill_result_with_error(result, error, models_to_create)
        logger.error(result["error"])
        return result

    try:
        tiny_config = get_tiny_config(config_class)
    except Exception as e:
        error = f"Failed to get tiny config for {config_class.__name__}: {e}"
        fill_result_with_error(result, error, models_to_create)
        logger.error(result["error"])
        return result

    # Convert the processors (reduce vocabulary size, smaller image size, etc.)
    processors = list(result["processor"].values())
    processor_output_folder = os.path.join(output_dir, "processors")
    try:
        processors = convert_processors(processors, tiny_config, processor_output_folder, result)
    except Exception as e:
        error = f"Failed to convert the processors: {e}"
        result["warnings"].append(error)

    if len(processors) == 0:
        error = f"No processor is returned by `convert_processors` for {config_class.__name__}."
        fill_result_with_error(result, error, models_to_create)
        logger.error(result["error"])
        return result

    try:
        config_overrides = get_config_overrides(config_class, processors)
    except Exception as e:
        error = f"Failure occurs while calling `get_config_overrides`: {e}"
        fill_result_with_error(result, error, models_to_create)
        logger.error(result["error"])
        return result

    # Just for us to see this easily in the report
    if "vocab_size" in config_overrides:
        result["vocab_size"] = config_overrides["vocab_size"]

    # Update attributes that `vocab_size` involves
    for k, v in config_overrides.items():
        if hasattr(tiny_config, k):
            setattr(tiny_config, k, v)
        # So far, we only have to deal with `text_config`, as `config_overrides` contains text-related attributes only.
        elif (
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
        logger.warning(result["warnings"])

    # update `result["processor"]`
    result["processor"] = {type(p).__name__: p.__class__.__name__ for p in processors}

    for pytorch_arch in models_to_create["pytorch"]:
        result["pytorch"][pytorch_arch.__name__] = {}
        error = None
        try:
            model = build_model(pytorch_arch, tiny_config, output_dir=output_dir)
        except Exception as e:
            model = None
            error = f"Failed to create the pytorch model for {pytorch_arch}: {e}"

        result["pytorch"][pytorch_arch.__name__]["model"] = model.__class__.__name__ if model is not None else None
        result["pytorch"][pytorch_arch.__name__]["checkpoint"] = (
            get_checkpoint_dir(output_dir, pytorch_arch) if model is not None else None
        )
        if error is not None:
            result["pytorch"][pytorch_arch.__name__]["error"] = error
            logger.error(f"{pytorch_arch.__name__}: {error}")

    for tensorflow_arch in models_to_create["tensorflow"]:
        # Make PT/TF weights compatible
        pt_arch_name = tensorflow_arch.__name__[2:]  # Remove `TF`
        pt_arch = getattr(transformers_module, pt_arch_name)

        result["tensorflow"][tensorflow_arch.__name__] = {}
        error = None
        if pt_arch.__name__ in result["pytorch"] and result["pytorch"][pt_arch.__name__]["checkpoint"] is not None:
            ckpt = get_checkpoint_dir(output_dir, pt_arch)
            # Use the same weights from PyTorch.
            try:
                model = tensorflow_arch.from_pretrained(ckpt, from_pt=True)
                model.save_pretrained(ckpt)
            except Exception as e:
                # Conversion may fail. Let's not create a model with different weights to avoid confusion (for now).
                model = None
                error = f"Failed to convert the pytorch model to the tensorflow model for {pt_arch}: {e}"
        else:
            try:
                model = build_model(tensorflow_arch, tiny_config, output_dir=output_dir)
            except Exception as e:
                model = None
                error = f"Failed to create the tensorflow model for {tensorflow_arch}: {e}"

        result["tensorflow"][tensorflow_arch.__name__]["model"] = (
            model.__class__.__name__ if model is not None else None
        )
        result["tensorflow"][tensorflow_arch.__name__]["checkpoint"] = (
            get_checkpoint_dir(output_dir, tensorflow_arch) if model is not None else None
        )
        if error is not None:
            result["tensorflow"][tensorflow_arch.__name__]["error"] = error
            logger.error(f"{tensorflow_arch.__name__}: {error}")

    if not result["error"]:
        del result["error"]
    if not result["warnings"]:
        del result["warnings"]

    return result


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

        for framework in FRAMEWORKS:
            if framework not in results[config_name]:
                continue
            for arch_name in results[config_name][framework]:
                if "error" in results[config_name][framework][arch_name]:
                    if config_name not in failed_results:
                        failed_results[config_name] = {}
                    if framework not in failed_results[config_name]:
                        failed_results[config_name][framework] = {}
                    if arch_name not in failed_results[config_name][framework]:
                        failed_results[config_name][framework][arch_name] = {}
                    error = results[config_name][framework][arch_name]["error"]
                    failed_results[config_name][framework][arch_name]["error"] = error

    return failed_results


def build_simple_report(results):

    text = ""
    failed_text = ""
    for config_name in results:
        for framework in FRAMEWORKS:
            if framework not in results[config_name]:
                continue
            for arch_name in results[config_name][framework]:
                if "error" in results[config_name][framework][arch_name]:
                    result = results[config_name][framework][arch_name]["error"]
                    failed_text += f"{arch_name}: {result}\n"
                else:
                    result = "OK"
                text += f"{arch_name}: {result}\n"

    return text, failed_text


if __name__ == "__main__":

    clone_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if os.getcwd() != clone_path:
        raise ValueError(f"This script should be run from the root of the clone of `transformers` {clone_path}")

    _pytorch_arch_mappings = [
        x
        for x in dir(transformers_module)
        if x.startswith("MODEL_") and x.endswith("_MAPPING") and x != "MODEL_NAMES_MAPPING"
    ]
    _tensorflow_arch_mappings = [
        x for x in dir(transformers_module) if x.startswith("TF_MODEL_") and x.endswith("_MAPPING")
    ]
    # _flax_arch_mappings = [x for x in dir(transformers_module) if x.startswith("FLAX_MODEL_") and x.endswith("_MAPPING")]

    pytorch_arch_mappings = [getattr(transformers_module, x) for x in _pytorch_arch_mappings]
    tensorflow_arch_mappings = [getattr(transformers_module, x) for x in _tensorflow_arch_mappings]
    # flax_arch_mappings = [getattr(transformers_module, x) for x in _flax_arch_mappings]

    unexportable_model_architectures = []

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    training_ds = ds["train"]
    testing_ds = ds["test"]

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
    parser.add_argument("--upload", action="store_true", help="If to upload the created tiny models to the Hub.")
    parser.add_argument(
        "--organization",
        default=None,
        type=str,
        help="The organization on the Hub to which the tiny models will be uploaded.",
    )
    parser.add_argument("output_path", type=Path, help="Path indicating where to store generated model.")

    args = parser.parse_args()

    if not args.all and not args.model_types:
        raise ValueError("Please provide at least one model type or pass `--all` to export all architectures.")

    config_classes = CONFIG_MAPPING.values()
    if not args.all:
        config_classes = [CONFIG_MAPPING[model_type] for model_type in args.model_types]

    # A map from config classes to tuples of processors (tokenizer, feature extractor, processor) classes
    processor_type_map = {c: get_processor_types_from_config_class(c) for c in config_classes}

    to_create = {
        c: {
            "processor": processor_type_map[c],
            "pytorch": get_architectures_from_config_class(c, pytorch_arch_mappings),
            "tensorflow": get_architectures_from_config_class(c, tensorflow_arch_mappings),
            # "flax": get_architectures_from_config_class(c, flax_arch_mappings),
        }
        for c in config_classes
    }

    results = {}
    for c, models_to_create in list(to_create.items()):
        print(f"Create models for {c.__name__} ...")
        result = build(c, models_to_create, output_dir=os.path.join(args.output_path, c.model_type))
        results[c.__name__] = result
        print("=" * 40)

    with open("tiny_model_creation_report.json", "w") as fp:
        json.dump(results, fp, indent=4)

    # Build the failure report
    failed_results = build_failed_report(results)
    with open("failed_report.json", "w") as fp:
        json.dump(failed_results, fp, indent=4)

    # Build the failure report
    simple_report, failed_report = build_simple_report(results)
    with open("simple_report.txt", "w") as fp:
        fp.write(simple_report)

    with open("simple_failed_report.txt", "w") as fp:
        fp.write(failed_report)

    if args.upload:
        if args.organization is None:
            raise ValueError("The argument `organization` could not be `None`. No model is uploaded")

        to_upload = []
        for model_type in os.listdir(args.output_path):
            for arch in os.listdir(os.path.join(args.output_path, model_type)):
                if arch == "processors":
                    continue
                to_upload.append(os.path.join(args.output_path, model_type, arch))
        to_upload = sorted(to_upload)

        upload_results = {}
        if len(to_upload) > 0:
            for model_dir in to_upload:
                try:
                    upload_model(model_dir, args.organization)
                except Exception as e:
                    error = f"Failed to upload {model_dir}: {e}"
                    logger.error(error)
                    upload_results[model_dir] = error

        with open("failed_uploads.json", "w") as fp:
            json.dump(upload_results, fp, indent=4)
