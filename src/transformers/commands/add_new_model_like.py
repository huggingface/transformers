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

# 1. Standard library
import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from re import Pattern
from typing import Any, Callable, Optional, Union

import yaml

from ..models.auto import modeling_auto
from ..models.auto.configuration_auto import model_type_to_module_name, CONFIG_MAPPING_NAMES, MODEL_NAMES_MAPPING
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from ..models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
from ..models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
from . import BaseTransformersCLICommand
from .add_fast_image_processor import add_fast_image_processor


CURRENT_YEAR = date.today().year
TRANSFORMERS_PATH = Path(__file__).parent.parent
REPO_PATH = TRANSFORMERS_PATH.parent.parent

COPYRIGHT = f"""
# coding=utf-8
# Copyright {CURRENT_YEAR} the HuggingFace team. All rights reserved.
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
""".strip()


def get_cased_name(lowercase_name: str) -> str:
    """From a model name in lowercase in the format `my_model`, return the cased name in the format `MyModel`."""
    alt_lowercase_name = lowercase_name.replace("_", "-")
    if lowercase_name in CONFIG_MAPPING_NAMES:
        return CONFIG_MAPPING_NAMES[lowercase_name].replace("Config", "")
    elif alt_lowercase_name in CONFIG_MAPPING_NAMES:
        return CONFIG_MAPPING_NAMES[alt_lowercase_name].replace("Config", "")
    else:
        return "".join(x.title() for x in lowercase_name.split("_"))


class ModelInfos(object):
    """
    Retrieve the basic informations about an existing model classes.
    """

    def __init__(self, lowercase_name: str):
        # Just to make sure it's indeed lowercase
        self.lowercase_name = lowercase_name.lower().replace(" ", "_").replace("-", "_")
        if self.lowercase_name not in CONFIG_MAPPING_NAMES:
            self.lowercase_name.replace("_", "-")
        if self.lowercase_name not in CONFIG_MAPPING_NAMES:
            raise ValueError(f"{lowercase_name} is not a valid model name")

        self.paper_name = MODEL_NAMES_MAPPING[self.lowercase_name]
        self.config_class = CONFIG_MAPPING_NAMES[self.lowercase_name]
        self.camelcase_name = self.config_class.replace("Config", "")

        # Get tokenizer class
        if self.lowercase_name in TOKENIZER_MAPPING_NAMES:
            self.tokenizer_class, self.fast_tokenizer_class = TOKENIZER_MAPPING_NAMES[model_type]
            self.fast_tokenizer_class = None if self.fast_tokenizer_class == "PreTrainedTokenizerFast" else self.fast_tokenizer_class
        else:
            self.tokenizer_class, self.fast_tokenizer_class = None, None

        # Get image processor classes
        image_processor_classes = IMAGE_PROCESSOR_MAPPING_NAMES.get(self.lowercase_name, None)
        if isinstance(image_processor_classes, tuple):
            if len(image_processor_classes) == 1:
                self.image_processor_class, self.fast_image_processor_class = image_processor_classes[0], None
            else:
                self.image_processor_class, self.fast_image_processor_class = image_processor_classes
        else:
            self.image_processor_class, self.fast_image_processor_class = image_processor_classes, None
        
        # Feature extractor and processor
        self.feature_extractor_class = FEATURE_EXTRACTOR_MAPPING_NAMES.get(self.lowercase_name, None)
        self.processor_class = PROCESSOR_MAPPING_NAMES.get(self.lowercase_name, None)



ATTRIBUTE_TO_PLACEHOLDER = {
    "config_class": "[CONFIG_CLASS]",
    "tokenizer_class": "[TOKENIZER_CLASS]",
    "image_processor_class": "[IMAGE_PROCESSOR_CLASS]",
    "image_processor_fast_class": "[IMAGE_PROCESSOR_FAST_CLASS]",
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
    search = re.search(r"^(\s*)(?:\S|$)", line)
    if search is None:
        return 0
    return len(search.groups()[0])


def parse_module_content(content: str) -> list[str]:
    """
    Parse the content of a module in the list of objects it defines.

    Args:
        content (`str`): The content to parse

    Returns:
        `list[str]`: The list of objects defined in the module.
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


def extract_block(content: str, indent_level: int = 0) -> str:
    """Return the first block in `content` with the indent level `indent_level`.

    The first line in `content` should be indented at `indent_level` level, otherwise an error will be thrown.

    This method will immediately stop the search when a (non-empty) line with indent level less than `indent_level` is
    encountered.

    Args:
        content (`str`): The content to parse
        indent_level (`int`, *optional*, default to 0): The indent level of the blocks to search for

    Returns:
        `str`: The first block in `content` with the indent level `indent_level`.
    """
    current_object = []
    lines = content.split("\n")
    # Doc-styler takes everything between two triple quotes in docstrings, so we need a fake """ here to go with this.
    end_markers = [")", "]", "}", '"""']

    for idx, line in enumerate(lines):
        if idx == 0 and indent_level > 0 and not is_empty_line(line) and find_indent(line) != indent_level:
            raise ValueError(
                f"When `indent_level > 0`, the first line in `content` should have indent level {indent_level}. Got "
                f"{find_indent(line)} instead."
            )

        if find_indent(line) < indent_level and not is_empty_line(line):
            break

        # End of an object
        is_valid_object = len(current_object) > 0
        if (
            not is_empty_line(line)
            and not line.endswith(":")
            and find_indent(line) == indent_level
            and is_valid_object
        ):
            # Closing parts should be included in current object
            if line.lstrip() in end_markers:
                current_object.append(line)
            return "\n".join(current_object)
        else:
            current_object.append(line)

    # Add last object
    if len(current_object) > 0:
        return "\n".join(current_object)


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
) -> tuple[str, str]:
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
    for attr in [
        "tokenizer_class",
        "image_processor_class",
        "image_processor_fast_class",
        "feature_extractor_class",
        "processor_class",
    ]:
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
        replacements (`list[tuple[str, str]]`): List of patterns (old, new)

    Returns:
        `list[tuple[str, str]]`: The list of patterns simplified.
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


_re_class_func = re.compile(r"^(?:class|def)\s+([^\s:\(]+)\s*(?:\(|\:)", flags=re.MULTILINE)


def remove_attributes(obj, target_attr):
    """Remove `target_attr` in `obj`."""
    lines = obj.split(os.linesep)

    target_idx = None
    for idx, line in enumerate(lines):
        # search for assignment
        if line.lstrip().startswith(f"{target_attr} = "):
            target_idx = idx
            break
        # search for function/method definition
        elif line.lstrip().startswith(f"def {target_attr}("):
            target_idx = idx
            break

    # target not found
    if target_idx is None:
        return obj

    line = lines[target_idx]
    indent_level = find_indent(line)
    # forward pass to find the ending of the block (including empty lines)
    parsed = extract_block("\n".join(lines[target_idx:]), indent_level)
    num_lines = len(parsed.split("\n"))
    for idx in range(num_lines):
        lines[target_idx + idx] = None

    # backward pass to find comments or decorator
    for idx in range(target_idx - 1, -1, -1):
        line = lines[idx]
        if (line.lstrip().startswith("#") or line.lstrip().startswith("@")) and find_indent(line) == indent_level:
            lines[idx] = None
        else:
            break

    new_obj = os.linesep.join([x for x in lines if x is not None])

    return new_obj


def duplicate_module(
    module_file: Union[str, os.PathLike],
    old_model_patterns: ModelPatterns,
    new_model_patterns: ModelPatterns,
    dest_file: Optional[str] = None,
    add_copied_from: bool = True,
    attrs_to_remove: Optional[list[str]] = None,
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

    content = re.sub(r"# Copyright (\d+)\s", f"# Copyright {CURRENT_YEAR} ", content)
    objects = parse_module_content(content)

    # Loop and treat all objects
    new_objects = []
    for obj in objects:
        # Regular classes functions
        old_obj = obj
        obj, replacement = replace_model_patterns(obj, old_model_patterns, new_model_patterns)
        has_copied_from = re.search(r"^#\s+Copied from", obj, flags=re.MULTILINE) is not None
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

    content = "\n".join(new_objects)
    # Remove some attributes that we don't want to copy to the new file(s)
    if attrs_to_remove is not None:
        for attr in attrs_to_remove:
            content = remove_attributes(content, target_attr=attr)

    with open(dest_file, "w", encoding="utf-8") as f:
        f.write(content)


def get_model_files(model_type: str) -> dict[str, Union[Path, list[Path]]]:
    """
    Retrieves all the files associated to a model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")

    Returns:
        `dict[str, Union[Path, list[Path]]]`: A dictionary with the following keys:
        - **doc_file** -- The documentation file for the model.
        - **model_files** -- All the files in the model module.
        - **test_files** -- The test files for the model.
    """
    module_name = model_type_to_module_name(model_type)

    model_module = TRANSFORMERS_PATH / "models" / module_name
    model_files = list(model_module.glob("*.py"))
    model_files = [f for f in model_files if not re.search(r"_(?:tf|flax)", Path(f).stem)]

    doc_file = REPO_PATH / "docs" / "source" / "en" / "model_doc" / f"{model_type}.md"

    # Basic pattern for test files
    test_files = [
        f"test_modeling_{module_name}.py",
        f"test_tokenization_{module_name}.py",
        f"test_image_processing_{module_name}.py",
        f"test_feature_extraction_{module_name}.py",
        f"test_processor_{module_name}.py",
    ]
    # Add the test directory
    test_files = [REPO_PATH / "tests" / "models" / module_name / f for f in test_files]
    # Filter by existing files
    test_files = [f for f in test_files if f.exists()]

    return {"doc_file": doc_file, "model_files": model_files, "module_name": module_name, "test_files": test_files}


_re_checkpoint_in_config = re.compile(r"\[(.+?)\]\((https://huggingface\.co/.+?)\)")


def find_base_model_checkpoint(
    model_type: str, model_files: Optional[dict[str, Union[Path, list[Path]]]] = None
) -> str:
    """
    Finds the model checkpoint used in the docstrings for a given model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        model_files (`dict[str, Union[Path, list[Path]]`, *optional*):
            The files associated to `model_type`. Can be passed to speed up the function, otherwise will be computed.

    Returns:
        `str`: The checkpoint used.
    """
    if model_files is None:
        model_files = get_model_files(model_type)
    module_files = model_files["model_files"]
    for fname in module_files:
        # After the @auto_docstring refactor, we expect the checkpoint to be in the configuration file's docstring
        if "configuration" not in str(fname):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
            if _re_checkpoint_in_config.search(content) is not None:
                checkpoint = _re_checkpoint_in_config.search(content).groups()[0]
                # Remove quotes
                checkpoint = checkpoint.replace('"', "")
                checkpoint = checkpoint.replace("'", "")
                return checkpoint

    # TODO: Find some kind of fallback if there is no _CHECKPOINT_FOR_DOC in any of the modeling file.
    return ""


_re_model_mapping = re.compile("MODEL_([A-Z_]*)MAPPING_NAMES")


def retrieve_model_classes(model_type: str) -> list[str]:
    """
    Retrieve the model classes associated to a given model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")

    Returns:
        `list[str]`: A list of model classes associated to the model.
    """
    new_model_classes = []
    model_mappings = [attr for attr in dir(modeling_auto) if _re_model_mapping.search(attr) is not None]
    for model_mapping_name in model_mappings:
        model_mapping = getattr(modeling_auto, model_mapping_name)
        if model_type in model_mapping:
            new_model_classes.append(model_mapping[model_type])

    if len(new_model_classes) > 0:
        # Remove duplicates
        model_classes = list(set(new_model_classes))

    return model_classes



def clean_init(init_file: Union[str, os.PathLike], keep_processing: bool = True):
    """
    Removes all the import lines that don't concern tokenizers/feature extractors/image processors/processors in an init.

    Args:
        init_file (`str` or `os.PathLike`): The path to the init to treat.
        keep_processing (`bool`, *optional*, defaults to `True`):
            Whether or not to keep the preprocessing (tokenizer, feature extractor, image processor, processor) imports
            in the init.
    """
    to_remove = ["tf", "flax"]
    if not keep_processing:
        to_remove.extend(["sentencepiece", "tokenizers", "vision"])

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
            re.search(r'^\s*"(tokenization|processing|feature_extraction|image_processing)', lines[idx]) is None
            and re.search(r"^\s*from .(tokenization|processing|feature_extraction|image_processing)", lines[idx])
            is None
        ):
            new_lines.append(lines[idx])
            idx += 1
        else:
            idx += 1

    with open(init_file, "w", encoding="utf-8") as f:
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

    pattern_tokenizer = re.compile(r"^\s*TOKENIZER_MAPPING_NAMES\s*=\s*OrderedDict\b")
    lines = content.split("\n")
    idx = 0
    # First we get to the TOKENIZER_MAPPING_NAMES block.
    while not pattern_tokenizer.search(lines[idx]):
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
            # should change to "        )," instead of "            ),"
            while not lines[idx].startswith("        ),"):
                block.append(lines[idx])
                idx += 1
            # if the lines[idx] does start with "        )," we still need it in our block
            block.append(lines[idx])
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
    "image_processing_auto.py": ['        ("{model_type}", "{image_processor_classes}"),'],
    "modeling_auto.py": ['        ("{model_type}", "{any_pt_class}"),'],
    "processing_auto.py": ['        ("{model_type}", "{processor_class}"),'],
}


def add_model_to_auto_classes(
    old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns, model_classes: list[str]
):
    """
    Add a model to the relevant mappings in the auto module.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        model_classes (`list[str]`): A list of model classes implemented.
    """
    for filename, patterns in AUTO_CLASSES_PATTERNS.items():
        # Extend patterns with all model classes if necessary
        new_patterns = []
        for pattern in patterns:
            if re.search("any_([a-z]*)_class", pattern) is not None:
                new_patterns.extend([pattern.replace("{" + "any_pt_class" + "}", cls) for cls in model_classes])
            elif "{config_class}" in pattern:
                new_patterns.append(pattern.replace("{config_class}", old_model_patterns.config_class))
            elif "{image_processor_classes}" in pattern:
                if (
                    old_model_patterns.image_processor_class is not None
                    and new_model_patterns.image_processor_class is not None
                ):
                    if (
                        old_model_patterns.image_processor_fast_class is not None
                        and new_model_patterns.image_processor_fast_class is not None
                    ):
                        new_patterns.append(
                            pattern.replace(
                                '"{image_processor_classes}"',
                                f'("{old_model_patterns.image_processor_class}", "{old_model_patterns.image_processor_fast_class}")',
                            )
                        )
                    else:
                        new_patterns.append(
                            pattern.replace(
                                '"{image_processor_classes}"', f'("{old_model_patterns.image_processor_class}",)'
                            )
                        )
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
):
    """
    Duplicate a documentation file and adapts it for a new model.

    Args:
        module_file (`str` or `os.PathLike`): Path to the doc file to duplicate.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        dest_file (`str` or `os.PathLike`, *optional*): Path to the new doc file.
            Will default to the a file named `{new_model_patterns.model_type}.md` in the same folder as `module_file`.
    """
    with open(doc_file, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"<!--\s*Copyright (\d+)\s", f"<!--Copyright {CURRENT_YEAR} ", content)
    if dest_file is None:
        dest_file = Path(doc_file).parent / f"{new_model_patterns.model_type}.md"

    # Parse the doc file in blocks. One block per section/header
    lines = content.split("\n")
    blocks = []
    current_block = []

    for line in lines:
        if line.startswith("#"):
            blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            if not re.search(r"</*(?:pt|tf|jax)>", line):
                current_block.append(line)
    blocks.append("\n".join(current_block))

    new_blocks = []
    in_classes = False
    for block in blocks:
        # Copyright
        if not block.startswith("#"):
            new_blocks.append(block)
        # Main title
        elif re.search(r"^#\s+\S+", block) is not None:
            new_blocks.append(f"# {new_model_patterns.model_name}\n")
        # The config starts the part of the doc with the classes.
        elif not in_classes and old_model_patterns.config_class in block.split("\n")[0]:
            in_classes = True
            new_blocks.append(DOC_OVERVIEW_TEMPLATE.format(model_name=new_model_patterns.model_name))
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)
            new_blocks.append(new_block)
        # In classes
        elif in_classes:
            block_title = block.split("\n")[0]
            block_class = re.search(r"^#+\s+(\S.*)$", block_title).groups()[0]
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)

            if "Tokenizer" in block_class:
                # We only add the tokenizer if necessary
                if old_model_patterns.tokenizer_class != new_model_patterns.tokenizer_class:
                    new_blocks.append(new_block)
            elif "ImageProcessor" in block_class:
                # We only add the image processor if necessary
                if old_model_patterns.image_processor_class != new_model_patterns.image_processor_class:
                    new_blocks.append(new_block)
            elif "ImageProcessorFast" in block_class:
                # We only add the image processor if necessary
                if old_model_patterns.image_processor_fast_class != new_model_patterns.image_processor_fast_class:
                    new_blocks.append(new_block)
            elif "FeatureExtractor" in block_class:
                # We only add the feature extractor if necessary
                if old_model_patterns.feature_extractor_class != new_model_patterns.feature_extractor_class:
                    new_blocks.append(new_block)
            elif "Processor" in block_class:
                # We only add the processor if necessary
                if old_model_patterns.processor_class != new_model_patterns.processor_class:
                    new_blocks.append(new_block)
            elif len(block_class.split(" ")) == 1:
                if not block_class.startswith("TF") and not block_class.startswith("Flax"):
                    new_blocks.append(new_block)
            else:
                new_blocks.append(new_block)

    with open(dest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_blocks))


def insert_model_in_doc_toc(old_model_patterns, new_model_patterns):
    """
    Insert the new model in the doc TOC, in the same section as the old model.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
    """
    toc_file = REPO_PATH / "docs" / "source" / "en" / "_toctree.yml"
    with open(toc_file, "r", encoding="utf8") as f:
        content = yaml.safe_load(f)

    # Get to the model API doc
    api_idx = 0
    while content[api_idx]["title"] != "API":
        api_idx += 1
    api_doc = content[api_idx]["sections"]

    model_idx = 0
    while api_doc[model_idx]["title"] != "Models":
        model_idx += 1
    model_doc = api_doc[model_idx]["sections"]

    # Find the base model in the Toc
    old_model_type = old_model_patterns.model_type
    section_idx = 0
    while section_idx < len(model_doc):
        sections = [entry["local"] for entry in model_doc[section_idx]["sections"]]
        if f"model_doc/{old_model_type}" in sections:
            break

        section_idx += 1

    if section_idx == len(model_doc):
        old_model = old_model_patterns.model_name
        new_model = new_model_patterns.model_name
        print(f"Did not find {old_model} in the table of content, so you will need to add {new_model} manually.")
        return

    # Add the new model in the same toc
    toc_entry = {"local": f"model_doc/{new_model_patterns.model_type}", "title": new_model_patterns.model_name}
    model_doc[section_idx]["sections"].append(toc_entry)
    model_doc[section_idx]["sections"] = sorted(model_doc[section_idx]["sections"], key=lambda s: s["title"].lower())
    api_doc[model_idx]["sections"] = model_doc
    content[api_idx]["sections"] = api_doc

    with open(toc_file, "w", encoding="utf-8") as f:
        f.write(yaml.dump(content, allow_unicode=True))


def create_modular_file(
    old_model_infos: ModelInfos,
    new_model_lowercase: str,
    add_tokenizer: bool,
    add_fast_tokenizer: bool,
    add_image_processor: bool,
    add_fast_image_processor: bool,
    add_feature_extractor: bool,
    add_processor: bool,
) -> str:
    old_model_files = get_model_files(old_model_infos.lowercase_name)
    old_model_classes = retrieve_model_classes(old_model_infos.lowercase_name)


def create_new_model_like(
    old_model_infos: ModelInfos,
    new_model_lowercase: str,
    new_model_paper_name: str,
    add_tokenizer: bool,
    add_fast_tokenizer: bool,
    add_image_processor: bool,
    add_fast_image_processor: bool,
    add_feature_extractor: bool,
    add_processor: bool,
    create_fast_image_processor: bool,
):
    """
    Creates a new model module like a given model of the Transformers library.

    Args:
        FILL
    """
    old_model_files = get_model_files(old_model_infos.lowercase_name)
    old_model_classes = retrieve_model_classes(old_model_infos.lowercase_name)

    # 1. We create the module for our new model
    new_module_folder = TRANSFORMERS_PATH / "models" / new_model_lowercase
    os.makedirs(new_module_folder, exist_ok=True)

    files_to_adapt = model_files["model_files"]
    if keep_old_processing:
        files_to_adapt = [
            f
            for f in files_to_adapt
            if "tokenization" not in str(f)
            and "processing" not in str(f)
            and "feature_extraction" not in str(f)
            and "image_processing" not in str(f)
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
        )

    clean_init(module_folder / "__init__.py", keep_processing=not keep_old_processing)

    # 2. We add our new model to the models init and the main init
    add_content_to_file(
        TRANSFORMERS_PATH / "models" / "__init__.py",
        f"    {new_model_patterns.model_lower_cased},",
        add_after=f"    {old_module_name},",
        exact_match=True,
    )

    # 3. Add test files
    files_to_adapt = model_files["test_files"]
    if keep_old_processing:
        files_to_adapt = [
            f
            for f in files_to_adapt
            if "tokenization" not in str(f)
            and "processor" not in str(f)
            and "feature_extraction" not in str(f)
            and "image_processing" not in str(f)
        ]

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
            attrs_to_remove=["pipeline_model_mapping", "is_pipeline_test_to_skip"],
        )

    # 4. Add model to auto classes
    add_model_to_auto_classes(old_model_patterns, new_model_patterns, model_classes)

    # 5. Add doc file
    doc_file = REPO_PATH / "docs" / "source" / "en" / "model_doc" / f"{old_model_patterns.model_type}.md"
    duplicate_doc_file(doc_file, old_model_patterns, new_model_patterns)
    insert_model_in_doc_toc(old_model_patterns, new_model_patterns)

    # 6. Add fast image processor if necessary
    if create_fast_image_processor:
        add_fast_image_processor(model_name=new_model_patterns.model_lower_cased)

    # 7. Warn the user for duplicate patterns
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
            "--path_to_repo", type=str, help="When not using an editable install, the path to the Transformers repo."
        )
        add_new_model_like_parser.set_defaults(func=add_new_model_like_command_factory)

    def __init__(self, config_file=None, path_to_repo=None, *args):
            (
                self.old_model_infos,
                self.new_model_lowercase,
                self.new_model_paper_name,
                self.add_tokenizer,
                self.add_fast_tokenizer,
                self.add_image_processor,
                self.add_fast_image_processor,
                self.add_feature_extractor,
                self.add_processor,
                self.create_fast_image_processor,
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
            old_model_info=self.old_model_infos,
            new_model_lowercase=self.new_model_lowercase,
            new_model_paper_name=self.new_model_paper_name,
            add_tokenizer=self.add_tokenizer,
            add_fast_tokenizer=self.add_fast_tokenizer,
            add_image_processor=self.add_image_processor,
            add_fast_image_processor=self.add_fast_image_processor,
            add_feature_extractor=self.add_feature_extractor,
            add_processor=self.add_processor,
            create_fast_image_processor=self.create_fast_image_processor,
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
            If set, the answer will be passed to this function. If this function raises an error on the provided
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
    model_types = list(MODEL_NAMES_MAPPING.keys())

    # Get old model type
    valid_model_type = False
    while not valid_model_type:
        old_model_type = input(
            "What is the model you would like to duplicate? Please provide the lowercase `model_name` (e.g. llama): "
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

    old_model_info = ModelInfos(old_model_type)

    # Ask for the new model name
    new_model_lowercase = get_user_field("What is the snake case name of the new model (e.g. `new_model`)? ")
    new_model_paper_name = get_user_field(
        "What is the full name (with no special casing) for your new model in the paper (e.g. `LlaMa`)? ",
        default_value="".join(x.title() for x in new_model_paper_name.split("_")
    )

    # Ask if we want to add individual processor classes as well
    add_tokenizer = False
    add_fast_tokenizer = False
    add_image_processor = False
    add_fast_image_processor = False
    add_feature_extractor = False
    add_processor = False
    if old_model_info.tokenizer_class is not None:
        add_tokenizer = not get_user_field(
        f"Will your new model use the same tokenizer class as {old_model_type} (yes/no)? ",
        convert_to=convert_to_bool,
        fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
    )
    if old_model_info.fast_tokenizer_class is not None:
        add_fast_tokenizer = not get_user_field(
        f"Will your new model use the same fast tokenizer class as {old_model_type} (yes/no)? ",
        convert_to=convert_to_bool,
        fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
    )
    if old_model_info.image_processor_class is not None:
        add_image_processor = not get_user_field(
        f"Will your new model use the same image processor class as {old_model_type} (yes/no)? ",
        convert_to=convert_to_bool,
        fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
    )
    if old_model_info.fast_image_processor_class is not None:
        add_fast_image_processor = not get_user_field(
        f"Will your new model use the same fast image processor class as {old_model_type} (yes/no)? ",
        convert_to=convert_to_bool,
        fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
    )
    if old_model_info.feature_extractor_class is not None:
        add_feature_extractor = not get_user_field(
        f"Will your new model use the same feature extractor class as {old_model_type} (yes/no)? ",
        convert_to=convert_to_bool,
        fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
    )
    if old_model_info.processor_class is not None:
        add_processor = not get_user_field(
        f"Will your new model use the same processor class as {old_model_type} (yes/no)? ",
        convert_to=convert_to_bool,
        fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
    )

    create_fast_image_processor = False
    if add_image_processor and not add_fast_image_processor:
        create_fast_image_processor = get_user_field(
                "A fast image processor can be created from the slow one, but modifications might be needed. "
                "Should we add a fast image processor class for this model (recommended) (yes/no)? ",
                convert_to=convert_to_bool,
                default_value="yes",
                fallback_message="Please answer yes/no, y/n, true/false or 1/0.",
            )


    return (
        old_model_info,
        new_model_lowercase,
        new_model_paper_name,
        add_tokenizer,
        add_fast_tokenizer,
        add_image_processor,
        add_fast_image_processor,
        add_feature_extractor,
        add_processor,
        create_fast_image_processor,
    )
