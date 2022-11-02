# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union

import transformers.models.auto as auto_module
from transformers.models.auto.configuration_auto import model_type_to_module_name

from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


CURRENT_YEAR = date.today().year
TRANSFORMERS_PATH = Path(__file__).parent.parent
REPO_PATH = TRANSFORMERS_PATH.parent.parent


@dataclass
class ModelPatterns:
    """
    Holds the basic information about a new model for the add-new-model-like command.

    Args:
        model_name (`str`): The model name.
        checkpoint (`str`): The checkpoint to use for doc examples.
        model_type (`str`, *optional*):
            The model type, the identifier used internally in the library like `bert` or `xlm-roberta`. Will default to
            `model_name` lowercased with spaces replaced with minuses (-).
        model_lower_cased (`str`, *optional*):
            The lowercased version of the model name, to use for the module name or function names. Will default to
            `model_name` lowercased with spaces and minuses replaced with underscores.
        model_camel_cased (`str`, *optional*):
            The camel-cased version of the model name, to use for the class names. Will default to `model_name`
            camel-cased (with spaces and minuses both considered as word separators.
        model_upper_cased (`str`, *optional*):
            The uppercased version of the model name, to use for the constant names. Will default to `model_name`
            uppercased with spaces and minuses replaced with underscores.
        config_class (`str`, *optional*):
            The tokenizer class associated with this model. Will default to `"{model_camel_cased}Config"`.
        tokenizer_class (`str`, *optional*):
            The tokenizer class associated with this model (leave to `None` for models that don't use a tokenizer).
        feature_extractor_class (`str`, *optional*):
            The feature extractor class associated with this model (leave to `None` for models that don't use a feature
            extractor).
        processor_class (`str`, *optional*):
            The processor class associated with this model (leave to `None` for models that don't use a processor).
    """

    model_name: str
    checkpoint: str
    model_type: Optional[str] = None
    model_lower_cased: Optional[str] = None
    model_camel_cased: Optional[str] = None
    model_upper_cased: Optional[str] = None
    config_class: Optional[str] = None
    tokenizer_class: Optional[str] = None
    feature_extractor_class: Optional[str] = None
    processor_class: Optional[str] = None

    def __post_init__(self):
        if self.model_type is None:
            self.model_type = self.model_name.lower().replace(" ", "-")
        if self.model_lower_cased is None:
            self.model_lower_cased = self.model_name.lower().replace(" ", "_").replace("-", "_")
        if self.model_camel_cased is None:
            # Split the model name on - and space
            words = self.model_name.split(" ")
            words = list(chain(*[w.split("-") for w in words]))
            # Make sure each word is capitalized
            words = [w[0].upper() + w[1:] for w in words]
            self.model_camel_cased = "".join(words)
        if self.model_upper_cased is None:
            self.model_upper_cased = self.model_name.upper().replace(" ", "_").replace("-", "_")
        if self.config_class is None:
            self.config_class = f"{self.model_camel_cased}Config"


ATTRIBUTE_TO_PLACEHOLDER = {
    "config_class": "[CONFIG_CLASS]",
    "tokenizer_class": "[TOKENIZER_CLASS]",
    "feature_extractor_class": "[FEATURE_EXTRACTOR_CLASS]",
    "processor_class": "[PROCESSOR_CLASS]",
    "checkpoint": "[CHECKPOINT]",
    "model_type": "[MODEL_TYPE]",
    "model_upper_cased": "[MODEL_UPPER_CASED]",
    "model_camel_cased": "[MODEL_CAMELCASED]",
    "model_lower_cased": "[MODEL_LOWER_CASED]",
    "model_name": "[MODEL_NAME]",
}


def is_empty_line(line: str) -> bool:
    """
    Determines whether a line is empty or not.
    """
    return len(line) == 0 or line.isspace()


def find_indent(line: str) -> int:
    """
    Returns the number of spaces that start a line indent.
    """
    search = re.search("^(\s*)(?:\S|$)", line)
    if search is None:
        return 0
    return len(search.groups()[0])


def parse_module_content(content: str) -> List[str]:
    """
    Parse the content of a module in the list of objects it defines.

    Args:
        content (`str`): The content to parse

    Returns:
        `List[str]`: The list of objects defined in the module.
    """
    objects = []
    current_object = []
    lines = content.split("\n")
    # Doc-styler takes everything between two triple quotes in docstrings, so we need a fake """ here to go with this.
    end_markers = [")", "]", "}", '"""']

    for line in lines:
        # End of an object
        is_valid_object = len(current_object) > 0
        if is_valid_object and len(current_object) == 1:
            is_valid_object = not current_object[0].startswith("# Copied from")
        if not is_empty_line(line) and find_indent(line) == 0 and is_valid_object:
            # Closing parts should be included in current object
            if line in end_markers:
                current_object.append(line)
                objects.append("\n".join(current_object))
                current_object = []
            else:
                objects.append("\n".join(current_object))
                current_object = [line]
        else:
            current_object.append(line)

    # Add last object
    if len(current_object) > 0:
        objects.append("\n".join(current_object))

    return objects


def add_content_to_text(
    text: str,
    content: str,
    add_after: Optional[Union[str, Pattern]] = None,
    add_before: Optional[Union[str, Pattern]] = None,
    exact_match: bool = False,
) -> str:
    """
    A utility to add some content inside a given text.

    Args:
       text (`str`): The text in which we want to insert some content.
       content (`str`): The content to add.
       add_after (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added after the first instance matching it.
       add_before (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added before the first instance matching it.
       exact_match (`bool`, *optional*, defaults to `False`):
           A line is considered a match with `add_after` or `add_before` if it matches exactly when `exact_match=True`,
           otherwise, if `add_after`/`add_before` is present in the line.

    <Tip warning={true}>

    The arguments `add_after` and `add_before` are mutually exclusive, and one exactly needs to be provided.

    </Tip>

    Returns:
        `str`: The text with the new content added if a match was found.
    """
    if add_after is None and add_before is None:
        raise ValueError("You need to pass either `add_after` or `add_before`")
    if add_after is not None and add_before is not None:
        raise ValueError("You can't pass both `add_after` or `add_before`")
    pattern = add_after if add_before is None else add_before

    def this_is_the_line(line):
        if isinstance(pattern, Pattern):
            return pattern.search(line) is not None
        elif exact_match:
            return pattern == line
        else:
            return pattern in line

    new_lines = []
    for line in text.split("\n"):
        if this_is_the_line(line):
            if add_before is not None:
                new_lines.append(content)
            new_lines.append(line)
            if add_after is not None:
                new_lines.append(content)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def add_content_to_file(
    file_name: Union[str, os.PathLike],
    content: str,
    add_after: Optional[Union[str, Pattern]] = None,
    add_before: Optional[Union[str, Pattern]] = None,
    exact_match: bool = False,
):
    """
    A utility to add some content inside a given file.

    Args:
       file_name (`str` or `os.PathLike`): The name of the file in which we want to insert some content.
       content (`str`): The content to add.
       add_after (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added after the first instance matching it.
       add_before (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added before the first instance matching it.
       exact_match (`bool`, *optional*, defaults to `False`):
           A line is considered a match with `add_after` or `add_before` if it matches exactly when `exact_match=True`,
           otherwise, if `add_after`/`add_before` is present in the line.

    <Tip warning={true}>

    The arguments `add_after` and `add_before` are mutually exclusive, and one exactly needs to be provided.

    </Tip>
    """
    with open(file_name, "r", encoding="utf-8") as f:
        old_content = f.read()

    new_content = add_content_to_text(
        old_content, content, add_after=add_after, add_before=add_before, exact_match=exact_match
    )

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(new_content)


def replace_model_patterns(
    text: str, old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns
) -> Tuple[str, str]:
    """
    Replace all patterns present in a given text.

    Args:
        text (`str`): The text to treat.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.

    Returns:
        `Tuple(str, str)`: A tuple of with the treated text and the replacement actually done in it.
    """
    # The order is crucially important as we will check and replace in that order. For instance the config probably
    # contains the camel-cased named, but will be treated before.
    attributes_to_check = ["config_class"]
    # Add relevant preprocessing classes
    for attr in ["tokenizer_class", "feature_extractor_class", "processor_class"]:
        if getattr(old_model_patterns, attr) is not None and getattr(new_model_patterns, attr) is not None:
            attributes_to_check.append(attr)

    # Special cases for checkpoint and model_type
    if old_model_patterns.checkpoint not in [old_model_patterns.model_type, old_model_patterns.model_lower_cased]:
        attributes_to_check.append("checkpoint")
    if old_model_patterns.model_type != old_model_patterns.model_lower_cased:
        attributes_to_check.append("model_type")
    else:
        text = re.sub(
            rf'(\s*)model_type = "{old_model_patterns.model_type}"',
            r'\1model_type = "[MODEL_TYPE]"',
            text,
        )

    # Special case when the model camel cased and upper cased names are the same for the old model (like for GPT2) but
    # not the new one. We can't just do a replace in all the text and will need a special regex
    if old_model_patterns.model_upper_cased == old_model_patterns.model_camel_cased:
        old_model_value = old_model_patterns.model_upper_cased
        if re.search(rf"{old_model_value}_[A-Z_]*[^A-Z_]", text) is not None:
            text = re.sub(rf"{old_model_value}([A-Z_]*)([^a-zA-Z_])", r"[MODEL_UPPER_CASED]\1\2", text)
    else:
        attributes_to_check.append("model_upper_cased")

    attributes_to_check.extend(["model_camel_cased", "model_lower_cased", "model_name"])

    # Now let's replace every other attribute by their placeholder
    for attr in attributes_to_check:
        text = text.replace(getattr(old_model_patterns, attr), ATTRIBUTE_TO_PLACEHOLDER[attr])

    # Finally we can replace the placeholder byt the new values.
    replacements = []
    for attr, placeholder in ATTRIBUTE_TO_PLACEHOLDER.items():
        if placeholder in text:
            replacements.append((getattr(old_model_patterns, attr), getattr(new_model_patterns, attr)))
            text = text.replace(placeholder, getattr(new_model_patterns, attr))

    # If we have two inconsistent replacements, we don't return anything (ex: GPT2->GPT_NEW and GPT2->GPTNew)
    old_replacement_values = [old for old, new in replacements]
    if len(set(old_replacement_values)) != len(old_replacement_values):
        return text, ""

    replacements = simplify_replacements(replacements)
    replacements = [f"{old}->{new}" for old, new in replacements]
    return text, ",".join(replacements)


def simplify_replacements(replacements):
    """
    Simplify a list of replacement patterns to make sure there are no needless ones.

    For instance in the sequence "Bert->BertNew, BertConfig->BertNewConfig, bert->bert_new", the replacement
    "BertConfig->BertNewConfig" is implied by "Bert->BertNew" so not needed.

    Args:
        replacements (`List[Tuple[str, str]]`): List of patterns (old, new)

    Returns:
        `List[Tuple[str, str]]`: The list of patterns simplified.
    """
    if len(replacements) <= 1:
        # Nothing to simplify
        return replacements

    # Next let's sort replacements by length as a replacement can only "imply" another replacement if it's shorter.
    replacements.sort(key=lambda x: len(x[0]))

    idx = 0
    while idx < len(replacements):
        old, new = replacements[idx]
        # Loop through all replacements after
        j = idx + 1
        while j < len(replacements):
            old_2, new_2 = replacements[j]
            # If the replacement is implied by the current one, we can drop it.
            if old_2.replace(old, new) == new_2:
                replacements.pop(j)
            else:
                j += 1
        idx += 1

    return replacements


def get_module_from_file(module_file: Union[str, os.PathLike]) -> str:
    """
    Returns the module name corresponding to a module file.
    """
    full_module_path = Path(module_file).absolute()
    module_parts = full_module_path.with_suffix("").parts

    # Find the first part named transformers, starting from the end.
    idx = len(module_parts) - 1
    while idx >= 0 and module_parts[idx] != "transformers":
        idx -= 1
    if idx < 0:
        raise ValueError(f"{module_file} is not a transformers module.")

    return ".".join(module_parts[idx:])


SPECIAL_PATTERNS = {
    "_CHECKPOINT_FOR_DOC =": "checkpoint",
    "_CONFIG_FOR_DOC =": "config_class",
    "_TOKENIZER_FOR_DOC =": "tokenizer_class",
    "_FEAT_EXTRACTOR_FOR_DOC =": "feature_extractor_class",
    "_PROCESSOR_FOR_DOC =": "processor_class",
}


_re_class_func = re.compile(r"^(?:class|def)\s+([^\s:\(]+)\s*(?:\(|\:)", flags=re.MULTILINE)


def duplicate_module(
    module_file: Union[str, os.PathLike],
    old_model_patterns: ModelPatterns,
    new_model_patterns: ModelPatterns,
    dest_file: Optional[str] = None,
    add_copied_from: bool = True,
):
    """
    Create a new module from an existing one and adapting all function and classes names from old patterns to new ones.

    Args:
        module_file (`str` or `os.PathLike`): Path to the module to duplicate.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        dest_file (`str` or `os.PathLike`, *optional*): Path to the new module.
        add_copied_from (`bool`, *optional*, defaults to `True`):
            Whether or not to add `# Copied from` statements in the duplicated module.
    """
    if dest_file is None:
        dest_file = str(module_file).replace(
            old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased
        )

    with open(module_file, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub("# Copyright (\d+)\s", f"# Copyright {CURRENT_YEAR} ", content)
    objects = parse_module_content(content)

    # Loop and treat all objects
    new_objects = []
    for obj in objects:
        # Special cases
        if "PRETRAINED_CONFIG_ARCHIVE_MAP = {" in obj:
            # docstyle-ignore
            obj = (
                f"{new_model_patterns.model_upper_cased}_PRETRAINED_CONFIG_ARCHIVE_MAP = "
                + "{"
                + f"""
    "{new_model_patterns.checkpoint}": "https://huggingface.co/{new_model_patterns.checkpoint}/resolve/main/config.json",
"""
                + "}\n"
            )
            new_objects.append(obj)
            continue
        elif "PRETRAINED_MODEL_ARCHIVE_LIST = [" in obj:
            if obj.startswith("TF_"):
                prefix = "TF_"
            elif obj.startswith("FLAX_"):
                prefix = "FLAX_"
            else:
                prefix = ""
            # docstyle-ignore
            obj = f"""{prefix}{new_model_patterns.model_upper_cased}_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "{new_model_patterns.checkpoint}",
    # See all {new_model_patterns.model_name} models at https://huggingface.co/models?filter={new_model_patterns.model_type}
]
"""
            new_objects.append(obj)
            continue

        special_pattern = False
        for pattern, attr in SPECIAL_PATTERNS.items():
            if pattern in obj:
                obj = obj.replace(getattr(old_model_patterns, attr), getattr(new_model_patterns, attr))
                new_objects.append(obj)
                special_pattern = True
                break

        if special_pattern:
            continue

        # Regular classes functions
        old_obj = obj
        obj, replacement = replace_model_patterns(obj, old_model_patterns, new_model_patterns)
        has_copied_from = re.search("^#\s+Copied from", obj, flags=re.MULTILINE) is not None
        if add_copied_from and not has_copied_from and _re_class_func.search(obj) is not None and len(replacement) > 0:
            # Copied from statement must be added just before the class/function definition, which may not be the
            # first line because of decorators.
            module_name = get_module_from_file(module_file)
            old_object_name = _re_class_func.search(old_obj).groups()[0]
            obj = add_content_to_text(
                obj, f"# Copied from {module_name}.{old_object_name} with {replacement}", add_before=_re_class_func
            )
        # In all cases, we remove Copied from statement with indent on methods.
        obj = re.sub("\n[ ]+# Copied from [^\n]*\n", "\n", obj)

        new_objects.append(obj)

    with open(dest_file, "w", encoding="utf-8") as f:
        content = f.write("\n".join(new_objects))


def filter_framework_files(
    files: List[Union[str, os.PathLike]], frameworks: Optional[List[str]] = None
) -> List[Union[str, os.PathLike]]:
    """
    Filter a list of files to only keep the ones corresponding to a list of frameworks.

    Args:
        files (`List[Union[str, os.PathLike]]`): The list of files to filter.
        frameworks (`List[str]`, *optional*): The list of allowed frameworks.

    Returns:
        `List[Union[str, os.PathLike]]`: The list of filtered files.
    """
    if frameworks is None:
        frameworks = get_default_frameworks()

    framework_to_file = {}
    others = []
    for f in files:
        parts = Path(f).name.split("_")
        if "modeling" not in parts:
            others.append(f)
            continue
        if "tf" in parts:
            framework_to_file["tf"] = f
        elif "flax" in parts:
            framework_to_file["flax"] = f
        else:
            framework_to_file["pt"] = f

    return [framework_to_file[f] for f in frameworks if f in framework_to_file] + others


def get_model_files(model_type: str, frameworks: Optional[List[str]] = None) -> Dict[str, Union[Path, List[Path]]]:
    """
    Retrieves all the files associated to a model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        frameworks (`List[str]`, *optional*):
            If passed, will only keep the model files corresponding to the passed frameworks.

    Returns:
        `Dict[str, Union[Path, List[Path]]]`: A dictionary with the following keys:
        - **doc_file** -- The documentation file for the model.
        - **model_files** -- All the files in the model module.
        - **test_files** -- The test files for the model.
    """
    module_name = model_type_to_module_name(model_type)

    model_module = TRANSFORMERS_PATH / "models" / module_name
    model_files = list(model_module.glob("*.py"))
    model_files = filter_framework_files(model_files, frameworks=frameworks)

    doc_file = REPO_PATH / "docs" / "source" / "en" / "model_doc" / f"{model_type}.mdx"

    # Basic pattern for test files
    test_files = [
        f"test_modeling_{module_name}.py",
        f"test_modeling_tf_{module_name}.py",
        f"test_modeling_flax_{module_name}.py",
        f"test_tokenization_{module_name}.py",
        f"test_feature_extraction_{module_name}.py",
        f"test_processor_{module_name}.py",
    ]
    test_files = filter_framework_files(test_files, frameworks=frameworks)
    # Add the test directory
    test_files = [REPO_PATH / "tests" / "models" / module_name / f for f in test_files]
    # Filter by existing files
    test_files = [f for f in test_files if f.exists()]

    return {"doc_file": doc_file, "model_files": model_files, "module_name": module_name, "test_files": test_files}


_re_checkpoint_for_doc = re.compile("^_CHECKPOINT_FOR_DOC\s+=\s+(\S*)\s*$", flags=re.MULTILINE)


def find_base_model_checkpoint(
    model_type: str, model_files: Optional[Dict[str, Union[Path, List[Path]]]] = None
) -> str:
    """
    Finds the model checkpoint used in the docstrings for a given model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        model_files (`Dict[str, Union[Path, List[Path]]`, *optional*):
            The files associated to `model_type`. Can be passed to speed up the function, otherwise will be computed.

    Returns:
        `str`: The checkpoint used.
    """
    if model_files is None:
        model_files = get_model_files(model_type)
    module_files = model_files["model_files"]
    for fname in module_files:
        if "modeling" not in str(fname):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
            if _re_checkpoint_for_doc.search(content) is not None:
                checkpoint = _re_checkpoint_for_doc.search(content).groups()[0]
                # Remove quotes
                checkpoint = checkpoint.replace('"', "")
                checkpoint = checkpoint.replace("'", "")
                return checkpoint

    # TODO: Find some kind of fallback if there is no _CHECKPOINT_FOR_DOC in any of the modeling file.
    return ""


def get_default_frameworks():
    """
    Returns the list of frameworks (PyTorch, TensorFlow, Flax) that are installed in the environment.
    """
    frameworks = []
    if is_torch_available():
        frameworks.append("pt")
    if is_tf_available():
        frameworks.append("tf")
    if is_flax_available():
        frameworks.append("flax")
    return frameworks


_re_model_mapping = re.compile("MODEL_([A-Z_]*)MAPPING_NAMES")


def retrieve_model_classes(model_type: str, frameworks: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Retrieve the model classes associated to a given model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        frameworks (`List[str]`, *optional*):
            The frameworks to look for. Will default to `["pt", "tf", "flax"]`, passing a smaller list will restrict
            the classes returned.

    Returns:
        `Dict[str, List[str]]`: A dictionary with one key per framework and the list of model classes associated to
        that framework as values.
    """
    if frameworks is None:
        frameworks = get_default_frameworks()

    modules = {
        "pt": auto_module.modeling_auto if is_torch_available() else None,
        "tf": auto_module.modeling_tf_auto if is_tf_available() else None,
        "flax": auto_module.modeling_flax_auto if is_flax_available() else None,
    }

    model_classes = {}
    for framework in frameworks:
        new_model_classes = []
        if modules[framework] is None:
            raise ValueError(f"You selected {framework} in the frameworks, but it is not installed.")
        model_mappings = [attr for attr in dir(modules[framework]) if _re_model_mapping.search(attr) is not None]
        for model_mapping_name in model_mappings:
            model_mapping = getattr(modules[framework], model_mapping_name)
            if model_type in model_mapping:
                new_model_classes.append(model_mapping[model_type])

        if len(new_model_classes) > 0:
            # Remove duplicates
            model_classes[framework] = list(set(new_model_classes))

    return model_classes


def retrieve_info_for_model(model_type, frameworks: Optional[List[str]] = None):
    """
    Retrieves all the information from a given model_type.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        frameworks (`List[str]`, *optional*):
            If passed, will only keep the info corresponding to the passed frameworks.

    Returns:
        `Dict`: A dictionary with the following keys:
        - **frameworks** (`List[str]`): The list of frameworks that back this model type.
        - **model_classes** (`Dict[str, List[str]]`): The model classes implemented for that model type.
        - **model_files** (`Dict[str, Union[Path, List[Path]]]`): The files associated with that model type.
        - **model_patterns** (`ModelPatterns`): The various patterns for the model.
    """
    if model_type not in auto_module.MODEL_NAMES_MAPPING:
        raise ValueError(f"{model_type} is not a valid model type.")

    model_name = auto_module.MODEL_NAMES_MAPPING[model_type]
    config_class = auto_module.configuration_auto.CONFIG_MAPPING_NAMES[model_type]
    archive_map = auto_module.configuration_auto.CONFIG_ARCHIVE_MAP_MAPPING_NAMES.get(model_type, None)
    if model_type in auto_module.tokenization_auto.TOKENIZER_MAPPING_NAMES:
        tokenizer_classes = auto_module.tokenization_auto.TOKENIZER_MAPPING_NAMES[model_type]
        tokenizer_class = tokenizer_classes[0] if tokenizer_classes[0] is not None else tokenizer_classes[1]
    else:
        tokenizer_class = None
    feature_extractor_class = auto_module.feature_extraction_auto.FEATURE_EXTRACTOR_MAPPING_NAMES.get(model_type, None)
    processor_class = auto_module.processing_auto.PROCESSOR_MAPPING_NAMES.get(model_type, None)

    model_files = get_model_files(model_type, frameworks=frameworks)
    model_camel_cased = config_class.replace("Config", "")

    available_frameworks = []
    for fname in model_files["model_files"]:
        if "modeling_tf" in str(fname):
            available_frameworks.append("tf")
        elif "modeling_flax" in str(fname):
            available_frameworks.append("flax")
        elif "modeling" in str(fname):
            available_frameworks.append("pt")

    if frameworks is None:
        frameworks = get_default_frameworks()

    frameworks = [f for f in frameworks if f in available_frameworks]

    model_classes = retrieve_model_classes(model_type, frameworks=frameworks)

    # Retrieve model upper-cased name from the constant name of the pretrained archive map.
    if archive_map is None:
        model_upper_cased = model_camel_cased.upper()
    else:
        parts = archive_map.split("_")
        idx = 0
        while idx < len(parts) and parts[idx] != "PRETRAINED":
            idx += 1
        if idx < len(parts):
            model_upper_cased = "_".join(parts[:idx])
        else:
            model_upper_cased = model_camel_cased.upper()

    model_patterns = ModelPatterns(
        model_name,
        checkpoint=find_base_model_checkpoint(model_type, model_files=model_files),
        model_type=model_type,
        model_camel_cased=model_camel_cased,
        model_lower_cased=model_files["module_name"],
        model_upper_cased=model_upper_cased,
        config_class=config_class,
        tokenizer_class=tokenizer_class,
        feature_extractor_class=feature_extractor_class,
        processor_class=processor_class,
    )

    return {
        "frameworks": frameworks,
        "model_classes": model_classes,
        "model_files": model_files,
        "model_patterns": model_patterns,
    }


def clean_frameworks_in_init(
    init_file: Union[str, os.PathLike], frameworks: Optional[List[str]] = None, keep_processing: bool = True
):
    """
    Removes all the import lines that don't belong to a given list of frameworks or concern tokenizers/feature
    extractors/processors in an init.

    Args:
        init_file (`str` or `os.PathLike`): The path to the init to treat.
        frameworks (`List[str]`, *optional*):
           If passed, this will remove all imports that are subject to a framework not in frameworks
        keep_processing (`bool`, *optional*, defaults to `True`):
            Whether or not to keep the preprocessing (tokenizer, feature extractor, processor) imports in the init.
    """
    if frameworks is None:
        frameworks = get_default_frameworks()

    names = {"pt": "torch"}
    to_remove = [names.get(f, f) for f in ["pt", "tf", "flax"] if f not in frameworks]
    if not keep_processing:
        to_remove.extend(["sentencepiece", "tokenizers", "vision"])

    if len(to_remove) == 0:
        # Nothing to do
        return

    remove_pattern = "|".join(to_remove)
    re_conditional_imports = re.compile(rf"^\s*if not is_({remove_pattern})_available\(\):\s*$")
    re_try = re.compile(r"\s*try:")
    re_else = re.compile(r"\s*else:")
    re_is_xxx_available = re.compile(rf"is_({remove_pattern})_available")

    with open(init_file, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []
    idx = 0
    while idx < len(lines):
        # Conditional imports in try-except-else blocks
        if (re_conditional_imports.search(lines[idx]) is not None) and (re_try.search(lines[idx - 1]) is not None):
            # Remove the preceding `try:`
            new_lines.pop()
            idx += 1
            # Iterate until `else:`
            while is_empty_line(lines[idx]) or re_else.search(lines[idx]) is None:
                idx += 1
            idx += 1
            indent = find_indent(lines[idx])
            while find_indent(lines[idx]) >= indent or is_empty_line(lines[idx]):
                idx += 1
        # Remove the import from utils
        elif re_is_xxx_available.search(lines[idx]) is not None:
            line = lines[idx]
            for framework in to_remove:
                line = line.replace(f", is_{framework}_available", "")
                line = line.replace(f"is_{framework}_available, ", "")
                line = line.replace(f"is_{framework}_available,", "")
                line = line.replace(f"is_{framework}_available", "")

            if len(line.strip()) > 0:
                new_lines.append(line)
            idx += 1
        # Otherwise we keep the line, except if it's a tokenizer import and we don't want to keep it.
        elif keep_processing or (
            re.search('^\s*"(tokenization|processing|feature_extraction)', lines[idx]) is None
            and re.search("^\s*from .(tokenization|processing|feature_extraction)", lines[idx]) is None
        ):
            new_lines.append(lines[idx])
            idx += 1
        else:
            idx += 1

    with open(init_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def add_model_to_main_init(
    old_model_patterns: ModelPatterns,
    new_model_patterns: ModelPatterns,
    frameworks: Optional[List[str]] = None,
    with_processing: bool = True,
):
    """
    Add a model to the main init of Transformers.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        frameworks (`List[str]`, *optional*):
            If specified, only the models implemented in those frameworks will be added.
        with_processsing (`bool`, *optional*, defaults to `True`):
            Whether the tokenizer/feature extractor/processor of the model should also be added to the init or not.
    """
    with open(TRANSFORMERS_PATH / "__init__.py", "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    idx = 0
    new_lines = []
    framework = None
    while idx < len(lines):
        new_framework = False
        if not is_empty_line(lines[idx]) and find_indent(lines[idx]) == 0:
            framework = None
        elif lines[idx].lstrip().startswith("if not is_torch_available"):
            framework = "pt"
            new_framework = True
        elif lines[idx].lstrip().startswith("if not is_tf_available"):
            framework = "tf"
            new_framework = True
        elif lines[idx].lstrip().startswith("if not is_flax_available"):
            framework = "flax"
            new_framework = True

        if new_framework:
            # For a new framework, we need to skip until the else: block to get where the imports are.
            while lines[idx].strip() != "else:":
                new_lines.append(lines[idx])
                idx += 1

        # Skip if we are in a framework not wanted.
        if framework is not None and frameworks is not None and framework not in frameworks:
            new_lines.append(lines[idx])
            idx += 1
        elif re.search(rf'models.{old_model_patterns.model_lower_cased}( |")', lines[idx]) is not None:
            block = [lines[idx]]
            indent = find_indent(lines[idx])
            idx += 1
            while find_indent(lines[idx]) > indent:
                block.append(lines[idx])
                idx += 1
            if lines[idx].strip() in [")", "]", "],"]:
                block.append(lines[idx])
                idx += 1
            block = "\n".join(block)
            new_lines.append(block)

            add_block = True
            if not with_processing:
                processing_classes = [
                    old_model_patterns.tokenizer_class,
                    old_model_patterns.feature_extractor_class,
                    old_model_patterns.processor_class,
                ]
                # Only keep the ones that are not None
                processing_classes = [c for c in processing_classes if c is not None]
                for processing_class in processing_classes:
                    block = block.replace(f' "{processing_class}",', "")
                    block = block.replace(f', "{processing_class}"', "")
                    block = block.replace(f" {processing_class},", "")
                    block = block.replace(f", {processing_class}", "")

                    if processing_class in block:
                        add_block = False
            if add_block:
                new_lines.append(replace_model_patterns(block, old_model_patterns, new_model_patterns)[0])
        else:
            new_lines.append(lines[idx])
            idx += 1

    with open(TRANSFORMERS_PATH / "__init__.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def insert_tokenizer_in_auto_module(old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns):
    """
    Add a tokenizer to the relevant mappings in the auto module.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
    """
    if old_model_patterns.tokenizer_class is None or new_model_patterns.tokenizer_class is None:
        return

    with open(TRANSFORMERS_PATH / "models" / "auto" / "tokenization_auto.py", "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    idx = 0
    # First we get to the TOKENIZER_MAPPING_NAMES block.
    while not lines[idx].startswith("    TOKENIZER_MAPPING_NAMES = OrderedDict("):
        idx += 1
    idx += 1

    # That block will end at this prompt:
    while not lines[idx].startswith("TOKENIZER_MAPPING = _LazyAutoMapping"):
        # Either all the tokenizer block is defined on one line, in which case, it ends with "),"
        if lines[idx].endswith(","):
            block = lines[idx]
        # Otherwise it takes several lines until we get to a "),"
        else:
            block = []
            while not lines[idx].startswith("            ),"):
                block.append(lines[idx])
                idx += 1
            block = "\n".join(block)
        idx += 1

        # If we find the model type and tokenizer class in that block, we have the old model tokenizer block
        if f'"{old_model_patterns.model_type}"' in block and old_model_patterns.tokenizer_class in block:
            break

    new_block = block.replace(old_model_patterns.model_type, new_model_patterns.model_type)
    new_block = new_block.replace(old_model_patterns.tokenizer_class, new_model_patterns.tokenizer_class)

    new_lines = lines[:idx] + [new_block] + lines[idx:]
    with open(TRANSFORMERS_PATH / "models" / "auto" / "tokenization_auto.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


AUTO_CLASSES_PATTERNS = {
    "configuration_auto.py": [
        '        ("{model_type}", "{model_name}"),',
        '        ("{model_type}", "{config_class}"),',
        '        ("{model_type}", "{pretrained_archive_map}"),',
    ],
    "feature_extraction_auto.py": ['        ("{model_type}", "{feature_extractor_class}"),'],
    "modeling_auto.py": ['        ("{model_type}", "{any_pt_class}"),'],
    "modeling_tf_auto.py": ['        ("{model_type}", "{any_tf_class}"),'],
    "modeling_flax_auto.py": ['        ("{model_type}", "{any_flax_class}"),'],
    "processing_auto.py": ['        ("{model_type}", "{processor_class}"),'],
}


def add_model_to_auto_classes(
    old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns, model_classes: Dict[str, List[str]]
):
    """
    Add a model to the relevant mappings in the auto module.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        model_classes (`Dict[str, List[str]]`): A dictionary framework to list of model classes implemented.
    """
    for filename in AUTO_CLASSES_PATTERNS:
        # Extend patterns with all model classes if necessary
        new_patterns = []
        for pattern in AUTO_CLASSES_PATTERNS[filename]:
            if re.search("any_([a-z]*)_class", pattern) is not None:
                framework = re.search("any_([a-z]*)_class", pattern).groups()[0]
                if framework in model_classes:
                    new_patterns.extend(
                        [
                            pattern.replace("{" + f"any_{framework}_class" + "}", cls)
                            for cls in model_classes[framework]
                        ]
                    )
            elif "{config_class}" in pattern:
                new_patterns.append(pattern.replace("{config_class}", old_model_patterns.config_class))
            elif "{feature_extractor_class}" in pattern:
                if (
                    old_model_patterns.feature_extractor_class is not None
                    and new_model_patterns.feature_extractor_class is not None
                ):
                    new_patterns.append(
                        pattern.replace("{feature_extractor_class}", old_model_patterns.feature_extractor_class)
                    )
            elif "{processor_class}" in pattern:
                if old_model_patterns.processor_class is not None and new_model_patterns.processor_class is not None:
                    new_patterns.append(pattern.replace("{processor_class}", old_model_patterns.processor_class))
            else:
                new_patterns.append(pattern)

        # Loop through all patterns.
        for pattern in new_patterns:
            full_name = TRANSFORMERS_PATH / "models" / "auto" / filename
            old_model_line = pattern
            new_model_line = pattern
            for attr in ["model_type", "model_name"]:
                old_model_line = old_model_line.replace("{" + attr + "}", getattr(old_model_patterns, attr))
                new_model_line = new_model_line.replace("{" + attr + "}", getattr(new_model_patterns, attr))
            if "pretrained_archive_map" in pattern:
                old_model_line = old_model_line.replace(
                    "{pretrained_archive_map}", f"{old_model_patterns.model_upper_cased}_PRETRAINED_CONFIG_ARCHIVE_MAP"
                )
                new_model_line = new_model_line.replace(
                    "{pretrained_archive_map}", f"{new_model_patterns.model_upper_cased}_PRETRAINED_CONFIG_ARCHIVE_MAP"
                )

            new_model_line = new_model_line.replace(
                old_model_patterns.model_camel_cased, new_model_patterns.model_camel_cased
            )

            add_content_to_file(full_name, new_model_line, add_after=old_model_line)

    # Tokenizers require special handling
    insert_tokenizer_in_auto_module(old_model_patterns, new_model_patterns)


DOC_OVERVIEW_TEMPLATE = """## Overview

The {model_name} model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

"""


def duplicate_doc_file(
    doc_file: Union[str, os.PathLike],
    old_model_patterns: ModelPatterns,
    new_model_patterns: ModelPatterns,
    dest_file: Optional[Union[str, os.PathLike]] = None,
    frameworks: Optional[List[str]] = None,
):
    """
    Duplicate a documentation file and adapts it for a new model.

    Args:
        module_file (`str` or `os.PathLike`): Path to the doc file to duplicate.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        dest_file (`str` or `os.PathLike`, *optional*): Path to the new doc file.
            Will default to the a file named `{new_model_patterns.model_type}.mdx` in the same folder as `module_file`.
        frameworks (`List[str]`, *optional*):
            If passed, will only keep the model classes corresponding to this list of frameworks in the new doc file.
    """
    with open(doc_file, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub("<!--\s*Copyright (\d+)\s", f"<!--Copyright {CURRENT_YEAR} ", content)
    if frameworks is None:
        frameworks = get_default_frameworks()
    if dest_file is None:
        dest_file = Path(doc_file).parent / f"{new_model_patterns.model_type}.mdx"

    # Parse the doc file in blocks. One block per section/header
    lines = content.split("\n")
    blocks = []
    current_block = []

    for line in lines:
        if line.startswith("#"):
            blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)
    blocks.append("\n".join(current_block))

    new_blocks = []
    in_classes = False
    for block in blocks:
        # Copyright
        if not block.startswith("#"):
            new_blocks.append(block)
        # Main title
        elif re.search("^#\s+\S+", block) is not None:
            new_blocks.append(f"# {new_model_patterns.model_name}\n")
        # The config starts the part of the doc with the classes.
        elif not in_classes and old_model_patterns.config_class in block.split("\n")[0]:
            in_classes = True
            new_blocks.append(DOC_OVERVIEW_TEMPLATE.format(model_name=new_model_patterns.model_name))
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)
            new_blocks.append(new_block)
        # In classes
        elif in_classes:
            in_classes = True
            block_title = block.split("\n")[0]
            block_class = re.search("^#+\s+(\S.*)$", block_title).groups()[0]
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)

            if "Tokenizer" in block_class:
                # We only add the tokenizer if necessary
                if old_model_patterns.tokenizer_class != new_model_patterns.tokenizer_class:
                    new_blocks.append(new_block)
            elif "FeatureExtractor" in block_class:
                # We only add the feature extractor if necessary
                if old_model_patterns.feature_extractor_class != new_model_patterns.feature_extractor_class:
                    new_blocks.append(new_block)
            elif "Processor" in block_class:
                # We only add the processor if necessary
                if old_model_patterns.processor_class != new_model_patterns.processor_class:
                    new_blocks.append(new_block)
            elif block_class.startswith("Flax"):
                # We only add Flax models if in the selected frameworks
                if "flax" in frameworks:
                    new_blocks.append(new_block)
            elif block_class.startswith("TF"):
                # We only add TF models if in the selected frameworks
                if "tf" in frameworks:
                    new_blocks.append(new_block)
            elif len(block_class.split(" ")) == 1:
                # We only add PyTorch models if in the selected frameworks
                if "pt" in frameworks:
                    new_blocks.append(new_block)
            else:
                new_blocks.append(new_block)

    with open(dest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_blocks))


def create_new_model_like(
    model_type: str,
    new_model_patterns: ModelPatterns,
    add_copied_from: bool = True,
    frameworks: Optional[List[str]] = None,
    old_checkpoint: Optional[str] = None,
):
    """
    Creates a new model module like a given model of the Transformers library.

    Args:
        model_type (`str`): The model type to duplicate (like "bert" or "gpt2")
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        add_copied_from (`bool`, *optional*, defaults to `True`):
            Whether or not to add "Copied from" statements to all classes in the new model modeling files.
        frameworks (`List[str]`, *optional*):
            If passed, will limit the duplicate to the frameworks specified.
        old_checkpoint (`str`, *optional*):
            The name of the base checkpoint for the old model. Should be passed along when it can't be automatically
            recovered from the `model_type`.
    """
    # Retrieve all the old model info.
    model_info = retrieve_info_for_model(model_type, frameworks=frameworks)
    model_files = model_info["model_files"]
    old_model_patterns = model_info["model_patterns"]
    if old_checkpoint is not None:
        old_model_patterns.checkpoint = old_checkpoint
    if len(old_model_patterns.checkpoint) == 0:
        raise ValueError(
            "The old model checkpoint could not be recovered from the model type. Please pass it to the "
            "`old_checkpoint` argument."
        )

    keep_old_processing = True
    for processing_attr in ["feature_extractor_class", "processor_class", "tokenizer_class"]:
        if getattr(old_model_patterns, processing_attr) != getattr(new_model_patterns, processing_attr):
            keep_old_processing = False

    model_classes = model_info["model_classes"]

    # 1. We create the module for our new model.
    old_module_name = model_files["module_name"]
    module_folder = TRANSFORMERS_PATH / "models" / new_model_patterns.model_lower_cased
    os.makedirs(module_folder, exist_ok=True)

    files_to_adapt = model_files["model_files"]
    if keep_old_processing:
        files_to_adapt = [
            f
            for f in files_to_adapt
            if "tokenization" not in str(f) and "processing" not in str(f) and "feature_extraction" not in str(f)
        ]

    os.makedirs(module_folder, exist_ok=True)
    for module_file in files_to_adapt:
        new_module_name = module_file.name.replace(
            old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased
        )
        dest_file = module_folder / new_module_name
        duplicate_module(
            module_file,
            old_model_patterns,
            new_model_patterns,
            dest_file=dest_file,
            add_copied_from=add_copied_from and "modeling" in new_module_name,
        )

    clean_frameworks_in_init(
        module_folder / "__init__.py", frameworks=frameworks, keep_processing=not keep_old_processing
    )

    # 2. We add our new model to the models init and the main init
    add_content_to_file(
        TRANSFORMERS_PATH / "models" / "__init__.py",
        f"    {new_model_patterns.model_lower_cased},",
        add_after=f"    {old_module_name},",
        exact_match=True,
    )
    add_model_to_main_init(
        old_model_patterns, new_model_patterns, frameworks=frameworks, with_processing=not keep_old_processing
    )

    # 3. Add test files
    files_to_adapt = model_files["test_files"]
    if keep_old_processing:
        files_to_adapt = [
            f
            for f in files_to_adapt
            if "tokenization" not in str(f) and "processor" not in str(f) and "feature_extraction" not in str(f)
        ]

    def disable_fx_test(filename: Path) -> bool:
        with open(filename) as fp:
            content = fp.read()
        new_content = re.sub(r"fx_compatible\s*=\s*True", "fx_compatible = False", content)
        with open(filename, "w") as fp:
            fp.write(new_content)
        return content != new_content

    disabled_fx_test = False

    tests_folder = REPO_PATH / "tests" / "models" / new_model_patterns.model_lower_cased
    os.makedirs(tests_folder, exist_ok=True)
    with open(tests_folder / "__init__.py", "w"):
        pass

    for test_file in files_to_adapt:
        new_test_file_name = test_file.name.replace(
            old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased
        )
        dest_file = test_file.parent.parent / new_model_patterns.model_lower_cased / new_test_file_name
        duplicate_module(
            test_file,
            old_model_patterns,
            new_model_patterns,
            dest_file=dest_file,
            add_copied_from=False,
        )
        disabled_fx_test = disabled_fx_test | disable_fx_test(dest_file)

    if disabled_fx_test:
        print(
            "The tests for symbolic tracing with torch.fx were disabled, you can add those once symbolic tracing works"
            " for your new model."
        )

    # 4. Add model to auto classes
    add_model_to_auto_classes(old_model_patterns, new_model_patterns, model_classes)

    # 5. Add doc file
    doc_file = REPO_PATH / "docs" / "source" / "en" / "model_doc" / f"{old_model_patterns.model_type}.mdx"
    duplicate_doc_file(doc_file, old_model_patterns, new_model_patterns, frameworks=frameworks)

    # 6. Warn the user for duplicate patterns
    if old_model_patterns.model_type == old_model_patterns.checkpoint:
        print(
            "The model you picked has the same name for the model type and the checkpoint name "
            f"({old_model_patterns.model_type}). As a result, it's possible some places where the new checkpoint "
            f"should be, you have {new_model_patterns.model_type} instead. You should search for all instances of "
            f"{new_model_patterns.model_type} in the new files and check they're not badly used as checkpoints."
        )
    elif old_model_patterns.model_lower_cased == old_model_patterns.checkpoint:
        print(
            "The model you picked has the same name for the model type and the checkpoint name "
            f"({old_model_patterns.model_lower_cased}). As a result, it's possible some places where the new "
            f"checkpoint should be, you have {new_model_patterns.model_lower_cased} instead. You should search for "
            f"all instances of {new_model_patterns.model_lower_cased} in the new files and check they're not badly "
            "used as checkpoints."
        )
    if (
        old_model_patterns.model_type == old_model_patterns.model_lower_cased
        and new_model_patterns.model_type != new_model_patterns.model_lower_cased
    ):
        print(
            "The model you picked has the same name for the model type and the lowercased model name "
            f"({old_model_patterns.model_lower_cased}). As a result, it's possible some places where the new "
            f"model type should be, you have {new_model_patterns.model_lower_cased} instead. You should search for "
            f"all instances of {new_model_patterns.model_lower_cased} in the new files and check they're not badly "
            "used as the model type."
        )

    if not keep_old_processing and old_model_patterns.tokenizer_class is not None:
        print(
            "The constants at the start of the new tokenizer file created needs to be manually fixed. If your new "
            "model has a tokenizer fast, you will also need to manually add the converter in the "
            "`SLOW_TO_FAST_CONVERTERS` constant of `convert_slow_tokenizer.py`."
        )


def add_new_model_like_command_factory(args: Namespace):
    return AddNewModelLikeCommand(config_file=args.config_file, path_to_repo=args.path_to_repo)


class AddNewModelLikeCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_new_model_like_parser = parser.add_parser("add-new-model-like")
        add_new_model_like_parser.add_argument(
            "--config_file", type=str, help="A file with all the information for this model creation."
        )
        add_new_model_like_parser.add_argument(
            "--path_to_repo", type=str, help="When not using an editable install, the path to the Transformers repo."
        )
        add_new_model_like_parser.set_defaults(func=add_new_model_like_command_factory)

    def __init__(self, config_file=None, path_to_repo=None, *args):
        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.old_model_type = config["old_model_type"]
            self.model_patterns = ModelPatterns(**config["new_model_patterns"])
            self.add_copied_from = config.get("add_copied_from", True)
            self.frameworks = config.get("frameworks", get_default_frameworks())
            self.old_checkpoint = config.get("old_checkpoint", None)
        else:
            (
                self.old_model_type,
                self.model_patterns,
                self.add_copied_from,
                self.frameworks,
                self.old_checkpoint,
            ) = get_user_input()

        self.path_to_repo = path_to_repo

    def run(self):
        if self.path_to_repo is not None:
            # Adapt constants
            global TRANSFORMERS_PATH
            global REPO_PATH

            REPO_PATH = Path(self.path_to_repo)
            TRANSFORMERS_PATH = REPO_PATH / "src" / "transformers"

        create_new_model_like(
            model_type=self.old_model_type,
            new_model_patterns=self.model_patterns,
            add_copied_from=self.add_copied_from,
            frameworks=self.frameworks,
            old_checkpoint=self.old_checkpoint,
        )


def get_user_field(
    question: str,
    default_value: Optional[str] = None,
    is_valid_answer: Optional[Callable] = None,
    convert_to: Optional[Callable] = None,
    fallback_message: Optional[str] = None,
) -> Any:
    """
    A utility function that asks a question to the user to get an answer, potentially looping until it gets a valid
    answer.

    Args:
        question (`str`): The question to ask the user.
        default_value (`str`, *optional*): A potential default value that will be used when the answer is empty.
        is_valid_answer (`Callable`, *optional*):
            If set, the question will be asked until this function returns `True` on the provided answer.
        convert_to (`Callable`, *optional*):
            If set, the answer will be passed to this function. If this function raises an error on the procided
            answer, the question will be asked again.
        fallback_message (`str`, *optional*):
            A message that will be displayed each time the question is asked again to the user.

    Returns:
        `Any`: The answer provided by the user (or the default), passed through the potential conversion function.
    """
    if not question.endswith(" "):
        question = question + " "
    if default_value is not None:
        question = f"{question} [{default_value}] "

    valid_answer = False
    while not valid_answer:
        answer = input(question)
        if default_value is not None and len(answer) == 0:
            answer = default_value
        if is_valid_answer is not None:
            valid_answer = is_valid_answer(answer)
        elif convert_to is not None:
            try:
                answer = convert_to(answer)
                valid_answer = True
            except Exception:
                valid_answer = False
        else:
            valid_answer = True

        if not valid_answer:
            print(fallback_message)

    return answer


def convert_to_bool(x: str) -> bool:
    """
    Converts a string to a bool.
    """
    if x.lower() in ["1", "y", "yes", "true"]:
        return True
    if x.lower() in ["0", "n", "no", "false"]:
        return False
    raise ValueError(f"{x} is not a value that can be converted to a bool.")


def get_user_input():
    """
    Ask the user for the necessary inputs to add the new model.
    """
    model_types = list(auto_module.configuration_auto.MODEL_NAMES_MAPPING.keys())

    # Get old model type
    valid_model_type = False
    while not valid_model_type:
        old_model_type = input(
            "What is the model you would like to duplicate? Please provide the lowercase `model_type` (e.g. roberta): "
        )
        if old_model_type in model_types:
            valid_model_type = True
        else:
            print(f"{old_model_type} is not a valid model type.")
            near_choices = difflib.get_close_matches(old_model_type, model_types)
            if len(near_choices) >= 1:
                if len(near_choices) > 1:
                    near_choices = " or ".join(near_choices)
                print(f"Did you mean {near_choices}?")

    old_model_info = retrieve_info_for_model(old_model_type)
    old_tokenizer_class = old_model_info["model_patterns"].tokenizer_class
    old_feature_extractor_class = old_model_info["model_patterns"].feature_extractor_class
    old_processor_class = old_model_info["model_patterns"].processor_class
    old_frameworks = old_model_info["frameworks"]

    old_checkpoint = None
    if len(old_model_info["model_patterns"].checkpoint) == 0:
        old_checkpoint = get_user_field(
            "We couldn't find the name of the base checkpoint for that model, please enter it here."
        )

    model_name = get_user_field(
        "What is the name (with no special casing) for your new model in the paper (e.g. RoBERTa)? "
    )
    default_patterns = ModelPatterns(model_name, model_name)

    model_type = get_user_field(
        "What identifier would you like to use for the `model_type` of this model? ",
        default_value=default_patterns.model_type,
    )
    model_lower_cased = get_user_field(
        "What lowercase name would you like to use for the module (folder) of this model? ",
        default_value=default_patterns.model_lower_cased,
    )
    model_camel_cased = get_user_field(
        "What prefix (camel-cased) would you like to use for the model classes of this model (e.g. Roberta)? ",
        default_value=default_patterns.model_camel_cased,
    )
    model_upper_cased = get_user_field(
        "What prefix (upper-cased) would you like to use for the constants relative to this model? ",
        default_value=default_patterns.model_upper_cased,
    )
    config_class = get_user_field(
        "What will be the name of the config class for this model? ", default_value=f"{model_camel_cased}Config"
    )
    checkpoint = get_user_field(
        "Please give a checkpoint identifier (on the model Hub) for this new model (e.g. facebook/roberta-base): "
    )

    old_processing_classes = [
        c for c in [old_feature_extractor_class, old_tokenizer_class, old_processor_class] if c is not None
    ]
    old_processing_classes = ", ".join(old_processing_classes)
    keep_processing = get_user_field(
        f"Will your new model use the same processing class as {old_model_type} ({old_processing_classes}) (yes/no)? ",
        convert_to=convert_to_bool,
        fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
    )
    if keep_processing:
        feature_extractor_class = old_feature_extractor_class
        processor_class = old_processor_class
        tokenizer_class = old_tokenizer_class
    else:
        if old_tokenizer_class is not None:
            tokenizer_class = get_user_field(
                "What will be the name of the tokenizer class for this model? ",
                default_value=f"{model_camel_cased}Tokenizer",
            )
        else:
            tokenizer_class = None
        if old_feature_extractor_class is not None:
            feature_extractor_class = get_user_field(
                "What will be the name of the feature extractor class for this model? ",
                default_value=f"{model_camel_cased}FeatureExtractor",
            )
        else:
            feature_extractor_class = None
        if old_processor_class is not None:
            processor_class = get_user_field(
                "What will be the name of the processor class for this model? ",
                default_value=f"{model_camel_cased}Processor",
            )
        else:
            processor_class = None

    model_patterns = ModelPatterns(
        model_name,
        checkpoint,
        model_type=model_type,
        model_lower_cased=model_lower_cased,
        model_camel_cased=model_camel_cased,
        model_upper_cased=model_upper_cased,
        config_class=config_class,
        tokenizer_class=tokenizer_class,
        feature_extractor_class=feature_extractor_class,
        processor_class=processor_class,
    )

    add_copied_from = get_user_field(
        "Should we add # Copied from statements when creating the new modeling file (yes/no)? ",
        convert_to=convert_to_bool,
        default_value="yes",
        fallback_message="Please answer yes/no, y/n, true/false or 1/0.",
    )

    all_frameworks = get_user_field(
        "Should we add a version of your new model in all the frameworks implemented by"
        f" {old_model_type} ({old_frameworks}) (yes/no)? ",
        convert_to=convert_to_bool,
        default_value="yes",
        fallback_message="Please answer yes/no, y/n, true/false or 1/0.",
    )
    if all_frameworks:
        frameworks = None
    else:
        frameworks = get_user_field(
            "Please enter the list of framworks you want (pt, tf, flax) separated by spaces",
            is_valid_answer=lambda x: all(p in ["pt", "tf", "flax"] for p in x.split(" ")),
        )
        frameworks = list(set(frameworks.split(" ")))

    return (old_model_type, model_patterns, add_copied_from, frameworks, old_checkpoint)
