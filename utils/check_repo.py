# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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

import importlib
import inspect
import os
import re
import warnings
from pathlib import Path

from transformers import is_flax_available, is_tf_available, is_torch_available
from transformers.file_utils import ENV_VARS_TRUE_VALUES
from transformers.models.auto import get_values


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_repo.py
PATH_TO_TRANSFORMERS = "src/transformers"
PATH_TO_TESTS = "tests"
PATH_TO_DOC = "docs/source"

# Update this list for models that are not tested with a comment explaining the reason it should not be.
# Being in this list is an exception and should **not** be the rule.
IGNORE_NON_TESTED = [
    # models to ignore for not tested
    "BigBirdPegasusEncoder",  # Building part of bigger (tested) model.
    "BigBirdPegasusDecoder",  # Building part of bigger (tested) model.
    "BigBirdPegasusDecoderWrapper",  # Building part of bigger (tested) model.
    "M2M100Encoder",  # Building part of bigger (tested) model.
    "M2M100Decoder",  # Building part of bigger (tested) model.
    "Speech2TextEncoder",  # Building part of bigger (tested) model.
    "Speech2TextDecoder",  # Building part of bigger (tested) model.
    "LEDEncoder",  # Building part of bigger (tested) model.
    "LEDDecoder",  # Building part of bigger (tested) model.
    "BartDecoderWrapper",  # Building part of bigger (tested) model.
    "BartEncoder",  # Building part of bigger (tested) model.
    "BertLMHeadModel",  # Needs to be setup as decoder.
    "BlenderbotSmallEncoder",  # Building part of bigger (tested) model.
    "BlenderbotSmallDecoderWrapper",  # Building part of bigger (tested) model.
    "BlenderbotEncoder",  # Building part of bigger (tested) model.
    "BlenderbotDecoderWrapper",  # Building part of bigger (tested) model.
    "MBartEncoder",  # Building part of bigger (tested) model.
    "MBartDecoderWrapper",  # Building part of bigger (tested) model.
    "MegatronBertLMHeadModel",  # Building part of bigger (tested) model.
    "MegatronBertEncoder",  # Building part of bigger (tested) model.
    "MegatronBertDecoder",  # Building part of bigger (tested) model.
    "MegatronBertDecoderWrapper",  # Building part of bigger (tested) model.
    "PegasusEncoder",  # Building part of bigger (tested) model.
    "PegasusDecoderWrapper",  # Building part of bigger (tested) model.
    "DPREncoder",  # Building part of bigger (tested) model.
    "DPRSpanPredictor",  # Building part of bigger (tested) model.
    "ProphetNetDecoderWrapper",  # Building part of bigger (tested) model.
    "ReformerForMaskedLM",  # Needs to be setup as decoder.
    "T5Stack",  # Building part of bigger (tested) model.
    "TFDPREncoder",  # Building part of bigger (tested) model.
    "TFDPRSpanPredictor",  # Building part of bigger (tested) model.
    "TFElectraMainLayer",  # Building part of bigger (tested) model (should it be a TFPreTrainedModel ?)
    "TFRobertaForMultipleChoice",  # TODO: fix
    "SeparableConv1D",  # Building part of bigger (tested) model.
]

# Update this list with test files that don't have a tester with a `all_model_classes` variable and which don't
# trigger the common tests.
TEST_FILES_WITH_NO_COMMON_TESTS = [
    "test_modeling_camembert.py",
    "test_modeling_flax_bert.py",
    "test_modeling_flax_roberta.py",
    "test_modeling_mbart.py",
    "test_modeling_mt5.py",
    "test_modeling_pegasus.py",
    "test_modeling_tf_camembert.py",
    "test_modeling_tf_mt5.py",
    "test_modeling_tf_xlm_roberta.py",
    "test_modeling_xlm_prophetnet.py",
    "test_modeling_xlm_roberta.py",
]

# Update this list for models that are not in any of the auto MODEL_XXX_MAPPING. Being in this list is an exception and
# should **not** be the rule.
IGNORE_NON_AUTO_CONFIGURED = [
    # models to ignore for model xxx mapping
    "CLIPTextModel",
    "CLIPVisionModel",
    "FlaxCLIPTextModel",
    "FlaxCLIPVisionModel",
    "DPRReader",
    "DPRSpanPredictor",
    "FlaubertForQuestionAnswering",
    "GPT2DoubleHeadsModel",
    "LukeForEntityClassification",
    "LukeForEntityPairClassification",
    "LukeForEntitySpanClassification",
    "OpenAIGPTDoubleHeadsModel",
    "RagModel",
    "RagSequenceForGeneration",
    "RagTokenForGeneration",
    "T5Stack",
    "TFDPRReader",
    "TFDPRSpanPredictor",
    "TFGPT2DoubleHeadsModel",
    "TFOpenAIGPTDoubleHeadsModel",
    "TFRagModel",
    "TFRagSequenceForGeneration",
    "TFRagTokenForGeneration",
    "Wav2Vec2ForCTC",
    "XLMForQuestionAnswering",
    "XLNetForQuestionAnswering",
    "SeparableConv1D",
]

# This is to make sure the transformers module imported is the one in the repo.
spec = importlib.util.spec_from_file_location(
    "transformers",
    os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"),
    submodule_search_locations=[PATH_TO_TRANSFORMERS],
)
transformers = spec.loader.load_module()


# If some modeling modules should be ignored for all checks, they should be added in the nested list
# _ignore_modules of this function.
def get_model_modules():
    """Get the model modules inside the transformers library."""
    _ignore_modules = [
        "modeling_auto",
        "modeling_encoder_decoder",
        "modeling_marian",
        "modeling_mmbt",
        "modeling_outputs",
        "modeling_retribert",
        "modeling_utils",
        "modeling_flax_auto",
        "modeling_flax_utils",
        "modeling_transfo_xl_utilities",
        "modeling_tf_auto",
        "modeling_tf_outputs",
        "modeling_tf_pytorch_utils",
        "modeling_tf_utils",
        "modeling_tf_transfo_xl_utilities",
    ]
    modules = []
    for model in dir(transformers.models):
        # There are some magic dunder attributes in the dir, we ignore them
        if not model.startswith("__"):
            model_module = getattr(transformers.models, model)
            for submodule in dir(model_module):
                if submodule.startswith("modeling") and submodule not in _ignore_modules:
                    modeling_module = getattr(model_module, submodule)
                    if inspect.ismodule(modeling_module):
                        modules.append(modeling_module)
    return modules


def get_models(module):
    """Get the objects in module that are models."""
    models = []
    model_classes = (transformers.PreTrainedModel, transformers.TFPreTrainedModel, transformers.FlaxPreTrainedModel)
    for attr_name in dir(module):
        if "Pretrained" in attr_name or "PreTrained" in attr_name:
            continue
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, model_classes) and attr.__module__ == module.__name__:
            models.append((attr_name, attr))
    return models


# If some test_modeling files should be ignored when checking models are all tested, they should be added in the
# nested list _ignore_files of this function.
def get_model_test_files():
    """Get the model test files."""
    _ignore_files = [
        "test_modeling_common",
        "test_modeling_encoder_decoder",
        "test_modeling_marian",
        "test_modeling_tf_common",
    ]
    test_files = []
    for filename in os.listdir(PATH_TO_TESTS):
        if (
            os.path.isfile(f"{PATH_TO_TESTS}/{filename}")
            and filename.startswith("test_modeling")
            and not os.path.splitext(filename)[0] in _ignore_files
        ):
            test_files.append(filename)
    return test_files


# This is a bit hacky but I didn't find a way to import the test_file as a module and read inside the tester class
# for the all_model_classes variable.
def find_tested_models(test_file):
    """Parse the content of test_file to detect what's in all_model_classes"""
    # This is a bit hacky but I didn't find a way to import the test_file as a module and read inside the class
    with open(os.path.join(PATH_TO_TESTS, test_file), "r", encoding="utf-8", newline="\n") as f:
        content = f.read()
    all_models = re.findall(r"all_model_classes\s+=\s+\(\s*\(([^\)]*)\)", content)
    # Check with one less parenthesis as well
    all_models += re.findall(r"all_model_classes\s+=\s+\(([^\)]*)\)", content)
    if len(all_models) > 0:
        model_tested = []
        for entry in all_models:
            for line in entry.split(","):
                name = line.strip()
                if len(name) > 0:
                    model_tested.append(name)
        return model_tested


def check_models_are_tested(module, test_file):
    """Check models defined in module are tested in test_file."""
    defined_models = get_models(module)
    tested_models = find_tested_models(test_file)
    if tested_models is None:
        if test_file in TEST_FILES_WITH_NO_COMMON_TESTS:
            return
        return [
            f"{test_file} should define `all_model_classes` to apply common tests to the models it tests. "
            + "If this intentional, add the test filename to `TEST_FILES_WITH_NO_COMMON_TESTS` in the file "
            + "`utils/check_repo.py`."
        ]
    failures = []
    for model_name, _ in defined_models:
        if model_name not in tested_models and model_name not in IGNORE_NON_TESTED:
            failures.append(
                f"{model_name} is defined in {module.__name__} but is not tested in "
                + f"{os.path.join(PATH_TO_TESTS, test_file)}. Add it to the all_model_classes in that file."
                + "If common tests should not applied to that model, add its name to `IGNORE_NON_TESTED`"
                + "in the file `utils/check_repo.py`."
            )
    return failures


def check_all_models_are_tested():
    """Check all models are properly tested."""
    modules = get_model_modules()
    test_files = get_model_test_files()
    failures = []
    for module in modules:
        test_file = f"test_{module.__name__.split('.')[-1]}.py"
        if test_file not in test_files:
            failures.append(f"{module.__name__} does not have its corresponding test file {test_file}.")
        new_failures = check_models_are_tested(module, test_file)
        if new_failures is not None:
            failures += new_failures
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))


def get_all_auto_configured_models():
    """Return the list of all models in at least one auto class."""
    result = set()  # To avoid duplicates we concatenate all model classes in a set.
    if is_torch_available():
        for attr_name in dir(transformers.models.auto.modeling_auto):
            if attr_name.startswith("MODEL_") and attr_name.endswith("MAPPING"):
                result = result | set(get_values(getattr(transformers.models.auto.modeling_auto, attr_name)))
    if is_tf_available():
        for attr_name in dir(transformers.models.auto.modeling_tf_auto):
            if attr_name.startswith("TF_MODEL_") and attr_name.endswith("MAPPING"):
                result = result | set(get_values(getattr(transformers.models.auto.modeling_tf_auto, attr_name)))
    if is_flax_available():
        for attr_name in dir(transformers.models.auto.modeling_flax_auto):
            if attr_name.startswith("FLAX_MODEL_") and attr_name.endswith("MAPPING"):
                result = result | set(get_values(getattr(transformers.models.auto.modeling_flax_auto, attr_name)))
    return [cls.__name__ for cls in result]


def ignore_unautoclassed(model_name):
    """Rules to determine if `name` should be in an auto class."""
    # Special white list
    if model_name in IGNORE_NON_AUTO_CONFIGURED:
        return True
    # Encoder and Decoder should be ignored
    if "Encoder" in model_name or "Decoder" in model_name:
        return True
    return False


def check_models_are_auto_configured(module, all_auto_models):
    """Check models defined in module are each in an auto class."""
    defined_models = get_models(module)
    failures = []
    for model_name, _ in defined_models:
        if model_name not in all_auto_models and not ignore_unautoclassed(model_name):
            failures.append(
                f"{model_name} is defined in {module.__name__} but is not present in any of the auto mapping. "
                "If that is intended behavior, add its name to `IGNORE_NON_AUTO_CONFIGURED` in the file "
                "`utils/check_repo.py`."
            )
    return failures


def check_all_models_are_auto_configured():
    """Check all models are each in an auto class."""
    missing_backends = []
    if not is_torch_available():
        missing_backends.append("PyTorch")
    if not is_tf_available():
        missing_backends.append("TensorFlow")
    if not is_flax_available():
        missing_backends.append("Flax")
    if len(missing_backends) > 0:
        missing = ", ".join(missing_backends)
        if os.getenv("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
            raise Exception(
                "Full quality checks require all backends to be installed (with `pip install -e .[dev]` in the "
                f"Transformers repo, the following are missing: {missing}."
            )
        else:
            warnings.warn(
                "Full quality checks require all backends to be installed (with `pip install -e .[dev]` in the "
                f"Transformers repo, the following are missing: {missing}. While it's probably fine as long as you "
                "didn't make any change in one of those backends modeling files, you should probably execute the "
                "command above to be on the safe side."
            )
    modules = get_model_modules()
    all_auto_models = get_all_auto_configured_models()
    failures = []
    for module in modules:
        new_failures = check_models_are_auto_configured(module, all_auto_models)
        if new_failures is not None:
            failures += new_failures
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))


_re_decorator = re.compile(r"^\s*@(\S+)\s+$")


def check_decorator_order(filename):
    """Check that in the test file `filename` the slow decorator is always last."""
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    decorator_before = None
    errors = []
    for i, line in enumerate(lines):
        search = _re_decorator.search(line)
        if search is not None:
            decorator_name = search.groups()[0]
            if decorator_before is not None and decorator_name.startswith("parameterized"):
                errors.append(i)
            decorator_before = decorator_name
        elif decorator_before is not None:
            decorator_before = None
    return errors


def check_all_decorator_order():
    """Check that in all test files, the slow decorator is always last."""
    errors = []
    for fname in os.listdir(PATH_TO_TESTS):
        if fname.endswith(".py"):
            filename = os.path.join(PATH_TO_TESTS, fname)
            new_errors = check_decorator_order(filename)
            errors += [f"- {filename}, line {i}" for i in new_errors]
    if len(errors) > 0:
        msg = "\n".join(errors)
        raise ValueError(
            f"The parameterized decorator (and its variants) should always be first, but this is not the case in the following files:\n{msg}"
        )


def find_all_documented_objects():
    """Parse the content of all doc files to detect which classes and functions it documents"""
    documented_obj = []
    for doc_file in Path(PATH_TO_DOC).glob("**/*.rst"):
        with open(doc_file, "r", encoding="utf-8", newline="\n") as f:
            content = f.read()
        raw_doc_objs = re.findall(r"(?:autoclass|autofunction):: transformers.(\S+)\s+", content)
        documented_obj += [obj.split(".")[-1] for obj in raw_doc_objs]
    return documented_obj


# One good reason for not being documented is to be deprecated. Put in this list deprecated objects.
DEPRECATED_OBJECTS = [
    "AutoModelWithLMHead",
    "BartPretrainedModel",
    "DataCollator",
    "DataCollatorForSOP",
    "GlueDataset",
    "GlueDataTrainingArguments",
    "LineByLineTextDataset",
    "LineByLineWithRefDataset",
    "LineByLineWithSOPTextDataset",
    "PretrainedBartModel",
    "PretrainedFSMTModel",
    "SingleSentenceClassificationProcessor",
    "SquadDataTrainingArguments",
    "SquadDataset",
    "SquadExample",
    "SquadFeatures",
    "SquadV1Processor",
    "SquadV2Processor",
    "TFAutoModelWithLMHead",
    "TFBartPretrainedModel",
    "TextDataset",
    "TextDatasetForNextSentencePrediction",
    "Wav2Vec2ForMaskedLM",
    "Wav2Vec2Tokenizer",
    "glue_compute_metrics",
    "glue_convert_examples_to_features",
    "glue_output_modes",
    "glue_processors",
    "glue_tasks_num_labels",
    "squad_convert_examples_to_features",
    "xnli_compute_metrics",
    "xnli_output_modes",
    "xnli_processors",
    "xnli_tasks_num_labels",
]

# Exceptionally, some objects should not be documented after all rules passed.
# ONLY PUT SOMETHING IN THIS LIST AS A LAST RESORT!
UNDOCUMENTED_OBJECTS = [
    "AddedToken",  # This is a tokenizers class.
    "BasicTokenizer",  # Internal, should never have been in the main init.
    "CharacterTokenizer",  # Internal, should never have been in the main init.
    "DPRPretrainedReader",  # Like an Encoder.
    "MecabTokenizer",  # Internal, should never have been in the main init.
    "ModelCard",  # Internal type.
    "SqueezeBertModule",  # Internal building block (should have been called SqueezeBertLayer)
    "TFDPRPretrainedReader",  # Like an Encoder.
    "TransfoXLCorpus",  # Internal type.
    "WordpieceTokenizer",  # Internal, should never have been in the main init.
    "absl",  # External module
    "add_end_docstrings",  # Internal, should never have been in the main init.
    "add_start_docstrings",  # Internal, should never have been in the main init.
    "cached_path",  # Internal used for downloading models.
    "convert_tf_weight_name_to_pt_weight_name",  # Internal used to convert model weights
    "logger",  # Internal logger
    "logging",  # External module
    "requires_backends",  # Internal function
]

# This list should be empty. Objects in it should get their own doc page.
SHOULD_HAVE_THEIR_OWN_PAGE = [
    # Benchmarks
    "PyTorchBenchmark",
    "PyTorchBenchmarkArguments",
    "TensorFlowBenchmark",
    "TensorFlowBenchmarkArguments",
]


def ignore_undocumented(name):
    """Rules to determine if `name` should be undocumented."""
    # NOT DOCUMENTED ON PURPOSE.
    # Constants uppercase are not documented.
    if name.isupper():
        return True
    # PreTrainedModels / Encoders / Decoders / Layers / Embeddings / Attention are not documented.
    if (
        name.endswith("PreTrainedModel")
        or name.endswith("Decoder")
        or name.endswith("Encoder")
        or name.endswith("Layer")
        or name.endswith("Embeddings")
        or name.endswith("Attention")
    ):
        return True
    # Submodules are not documented.
    if os.path.isdir(os.path.join(PATH_TO_TRANSFORMERS, name)) or os.path.isfile(
        os.path.join(PATH_TO_TRANSFORMERS, f"{name}.py")
    ):
        return True
    # All load functions are not documented.
    if name.startswith("load_tf") or name.startswith("load_pytorch"):
        return True
    # is_xxx_available functions are not documented.
    if name.startswith("is_") and name.endswith("_available"):
        return True
    # Deprecated objects are not documented.
    if name in DEPRECATED_OBJECTS or name in UNDOCUMENTED_OBJECTS:
        return True
    # MMBT model does not really work.
    if name.startswith("MMBT"):
        return True
    if name in SHOULD_HAVE_THEIR_OWN_PAGE:
        return True
    return False


def check_all_objects_are_documented():
    """Check all models are properly documented."""
    documented_objs = find_all_documented_objects()
    modules = transformers._modules
    objects = [c for c in dir(transformers) if c not in modules and not c.startswith("_")]
    undocumented_objs = [c for c in objects if c not in documented_objs and not ignore_undocumented(c)]
    if len(undocumented_objs) > 0:
        raise Exception(
            "The following objects are in the public init so should be documented:\n - "
            + "\n - ".join(undocumented_objs)
        )


def check_repo_quality():
    """Check all models are properly tested and documented."""
    print("Checking all models are properly tested.")
    check_all_decorator_order()
    check_all_models_are_tested()
    print("Checking all objects are properly documented.")
    check_all_objects_are_documented()
    print("Checking all models are in at least one auto class.")
    check_all_models_are_auto_configured()


if __name__ == "__main__":
    check_repo_quality()
