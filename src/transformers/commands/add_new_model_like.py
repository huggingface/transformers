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

import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional

import transformers.models.auto as auto_module
from transformers.models.auto.configuration_auto import model_type_to_module_name

from ..utils import logging
from . import BaseTransformersCLICommand


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
            The tokenizer class associated with this model. Will default to `"{model_camel_cased}Tokenizer"`.
    """

    model_name: str
    checkpoint: str
    model_type: Optional[str] = None
    model_lower_cased: Optional[str] = None
    model_camel_cased: Optional[str] = None
    model_upper_cased: Optional[str] = None
    config_class: Optional[str] = None
    tokenizer_class: Optional[str] = None

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
        if self.tokenizer_class is None:
            self.tokenizer_class = f"{self.model_camel_cased}Tokenizer"


def is_empty_line(line):
    return len(line) == 0 or line.isspace()


def find_indent(line):
    """
    Returns the number of spaces that start a line indent.
    """
    search = re.search("^(\s*)(?:\S|$)", line)
    if search is None:
        return 0
    return len(search.groups()[0])


def parse_module_content(content: str):
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
    # fmt: off
    # Black will try to restyle the \"\"\" in the next list into triple quotes and then the doc-styler is going to
    # believe every code sample is a docstring...

    end_markers = [")", "]", "}", '\"\"\"']
    # fmt: on

    for line in lines:
        # End of an object
        if not is_empty_line(line) and find_indent(line) == 0 and len(current_object) > 0:
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


def add_content_to_text(text, content, add_after=None, add_before=None, exact_match=False):
    """
    A utility to add some content inside a given text.
    
    Args:
       text (`str`): The text in which we want to insert some content.
       content (`str`): The content to add.
       add_after (`str` or `re.Pattern`):
           The pattern to test on a line of `text`, the new content is added after the first instance matching it.
       add_before (`str` or `re.Pattern`):
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
        if isinstance(pattern, re.Pattern):
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


def add_content_to_file(file_name, content, add_after=None, add_before=None, exact_match=False):
    """
    A utility to add some content inside a given file.
    
    Args:
       file_name (`str` or `os.PathLike`): The name of the file in which we want to insert some content.
       content (`str`): The content to add.
       add_after (`str` or `re.Pattern`):
           The pattern to test on a line of `text`, the new content is added after the first instance matching it.
       add_before (`str` or `re.Pattern`):
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


def replace_model_patterns(text: str, old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns):
    """
    Replace all patterns present in a given text.
    
    Args:
        text (`str`): The text to treat.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
    
    Returns:
        `Tuple(str, str)`: A tuple of with the treated text and the replacement actually done in it.
    """
    replacements = []
    # Special case when the model camel cased and upper cased names are the same for the old model (GPT2) but not the
    # new one. We can't just do a replace in all the text and will need to go word by word to see if they are
    # uppercased or not.
    if old_model_patterns.model_upper_cased == old_model_patterns.model_camel_cased and new_model_patterns.model_upper_cased != new_model_patterns.model_camel_cased:
        old_model_value = old_model_patterns.model_upper_cased
        new_model_upper = new_model_patterns.model_upper_cased
        
        if re.search(fr"{old_model_value}_[A-Z_]*[^A-Z_]", text) is not None:
            replacements.append((old_model_value, new_model_upper))
            text = re.sub(fr"{old_model_value}([A-Z_]*)([^a-zA-Z_])", fr"{new_model_upper}\1\2", text)
        
        # Now that we have done those, we can replace the camel cased ones normally.
        attributes = ["model_lower_cased", "model_camel_cased"]
    else:
        attributes = ["model_lower_cased", "model_camel_cased", "model_upper_cased"]
    
    for attribute in attributes:
        old_model_value = getattr(old_model_patterns, attribute)
        new_model_value = getattr(new_model_patterns, attribute)
        if old_model_value in text:
            replacements.append((old_model_value, new_model_value))
            text = text.replace(old_model_value, new_model_value)
    
    # We may have a config class that is different from NewModelConfig:
    if new_model_patterns.config_class != f"{new_model_patterns.model_camel_cased}Config":
        text = text.replace(f"{new_model_patterns.model_camel_cased}Config", old_model_patterns.config_class)
    
    # We may have a tokenizer class that is different from NewModelTokenizer:
    if new_model_patterns.tokenizer_class != f"{new_model_patterns.model_camel_cased}Tokenizer":
        text = text.replace(f"{new_model_patterns.model_camel_cased}Tokenizer", old_model_patterns.tokenizer_class)
    
    # If we have two inconsistent replacements, we don't return anything (ex: GPT2->GPT_NEW and GPT2->GPTNew)
    old_replacement_values = [old for old, new in replacements]
    if len(set(old_replacement_values)) != len(old_replacement_values):
        return text, ""
    
    if old_model_patterns.model_type == old_model_patterns.model_lower_cased:
        text = re.sub(fr'(\s*)model_type = "{new_model_patterns.model_lower_cased}"', fr'\1model_type = "{new_model_patterns.model_type}"', text)
    else:
        text = re.sub(fr'(\s*)model_type = "{old_model_patterns.model_type}"', fr'\1model_type = "{new_model_patterns.model_type}"', text)
    
    replacements = [f"{old}->{new}" for old, new in replacements]
    return text, ",".join(replacements)


def get_module_from_file(module_file):
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
}


_re_class_func = re.compile(r"^(?:class|def)\s+([^\s:\(]+)\s*(?:\(|\:)", flags=re.MULTILINE)


def duplicate_module(
    module_file: str,
    old_model_patterns: ModelPatterns,
    new_model_patterns: ModelPatterns,
    dest_file: Optional[str] = None,
    add_copied_from: bool = True,
):
    """
    Create a new module from an existing one and adapting all function and classes names from old patterns to new ones.
    """
    if dest_file is None:
        dest_file = str(module_file).replace(old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased)

    with open(module_file, "r", encoding="utf-8") as f:
        content = f.read()

    objects = parse_module_content(content)
    
    # Loop and treat all objects
    new_objects = []
    for obj in objects:
        # Special cases
        if "PRETRAINED_CONFIG_ARCHIVE_MAP = {" in obj:
            obj = f"{new_model_patterns.model_upper_cased}_PRETRAINED_CONFIG_ARCHIVE_MAP = " + "{" + f"""
    "{new_model_patterns.checkpoint}": "https://huggingface.co/{new_model_patterns.checkpoint}/resolve/main/config.json",
""" + "}\n"
            new_objects.append(obj)
            continue
        elif "PRETRAINED_MODEL_ARCHIVE_LIST = [" in obj:
            if obj.startswith("TF_"):
                prefix = "TF_"
            elif obj.startswith("FLAX_"):
                prefix = "FLAX_"
            else:
                prefix = ""
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
        obj, replacement = replace_model_patterns(obj, old_model_patterns, new_model_patterns)
        has_copied_from = re.search("^Copied from", obj, flags=re.MULTILINE)
        if add_copied_from and not has_copied_from and _re_class_func.search(content) is not None and len(replacement) > 0:
            # Copied from statement must be added just before the class/function definition, which may not be the
            # first line because of decorators.
            object_name = _re_class_func.search(content).groups()[0]
            module_name = get_module_from_file(module_file)
            obj = add_content_to_text(obj, f"# Copied from {module_name} with {replacement}", add_before=_re_class_func)
        elif not add_copied_from and has_copied_from:
            obj = re.sub("\n# Copied from [^\n]*\n", "\n", obj)
        # In all cases, we remove Copied from statement with indent on methods.
        obj = re.sub("\n[ ]+# Copied from [^\n]*\n", "\n", obj)
        
        new_objects.append(obj)
    
    with open(dest_file, "w", encoding="utf-8") as f:
        content = f.write("\n".join(new_objects))


def get_model_files(model_type):
    """
    Retrieves all the files associated to a model.
    
    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
    
    Returns:
        `Dict[str, [Path, List[Path]]]`: A dictionary with the following keys:
        - **doc_file** -- The documentation file for the model.
        - **model_files** -- All the files in the model module.
        - **test_files** -- The test files for the model.
    """
    module_name = auto_module.configuration_auto.model_type_to_module_name(model_type)
    
    transformers_path = Path(auto_module.__path__[0]).parent.parent
    repo_path = transformers_path.parent.parent
    
    model_module = transformers_path / "models" / module_name
    model_files = list(model_module.glob("*.py"))
    
    doc_file = repo_path / "models" / "docs" / "source" / f"{model_type}.mdx"
    
    # Basic pattern for test files
    test_files = [
        f"test_modeling_{module_name}.py",
        f"test_modeling_tf_{module_name}.py",
        f"test_modeling_flax_{module_name}.py",
        f"test_tokenization_{module_name}.py",
    ]
    # Add the test directory
    test_files = [repo_path / "tests" / f for f in test_files]
    # Filter by existing files
    test_files = [f for f in test_files if f.exists()]
    
    return {
        "doc_file": doc_file,
        "model_files": model_files,
        "module_name": module_name,
        "test_files": test_files
    }


_re_checkpoint_for_doc = re.compile("^_CHECKPOINT_FOR_DOC\s+=\s+(\S*)\s*$", flags=re.MULTILINE)


def find_base_model_checkpoint(model_type, model_files=None):
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


_re_model_mapping = re.compile("MODEL_([A-Z_]*)MAPPING_NAMES")


def retrieve_model_classes(model_type, frameworks=None):
    if frameworks is None:
        frameworks = ["pt", "tf", "flax"]
    
    modules = {
        "pt": auto_module.modeling_auto,
        "tf": auto_module.modeling_tf_auto,
        "flax": auto_module.modeling_flax_auto,
    }
    
    model_classes = {}
    for framework in frameworks:
        new_model_classes = []
        model_mappings = [attr for attr in dir(modules[framework]) if _re_model_mapping.search(attr) is not None]
        for model_mapping_name in model_mappings:
            model_mapping = getattr(modules[framework], model_mapping_name)
            if model_type in model_mapping:
                new_model_classes.append(model_mapping[model_type])
        
        
        if len(new_model_classes) > 0:
            # Remove duplicates
            model_classes[framework] = list(set(new_model_classes))
    
    return model_classes


def retrieve_info_for_model(model_type):
    """
    Retrieves all the information from a given model_type.
    
    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
    
    Returns:
        `Dict`: A dictionary with the following keys:
        - **model_patterns** (`ModelPatterns`): The various patterns for the model.
    """
    if model_type not in auto_module.MODEL_NAMES_MAPPING:
        raise ValueError(f"{model_type} is not a valid model type.")
    
    model_name = auto_module.MODEL_NAMES_MAPPING[model_type]
    config_class = auto_module.configuration_auto.CONFIG_MAPPING_NAMES[model_type]
    tokenizer_classes = auto_module.tokenization_auto.TOKENIZER_MAPPING_NAMES[model_type]
    tokenizer_class = tokenizer_classes[0] if tokenizer_classes[0] is not None else tokenizer_classes[1]
    
    model_files = get_model_files(model_type)
    model_camel_cased = config_class.replace("Config", "")
    
    frameworks = []
    for fname in model_files["model_files"]:
        if "modeling_tf" in str(fname):
            frameworks.append("tf")
        elif "modeling_flax" in str(fname):
            frameworks.append("flax")
        elif "modeling" in str(fname):
            frameworks.append("pt")
    
    model_classes = retrieve_model_classes(model_type, frameworks=frameworks)
    
    model_patterns = ModelPatterns(
        model_name,
        checkpoint=find_base_model_checkpoint(model_type, model_files=model_files),
        model_type=model_type,
        model_camel_cased=model_camel_cased,
        model_lower_cased=model_files["module_name"],
        model_upper_cased=model_camel_cased.upper(),
        config_class=config_class,
        tokenizer_class=tokenizer_class,
    )
    
    return {
        "frameworks": frameworks,
        "model_classes": model_classes,
        "model_files": model_files,
        "model_patterns":model_patterns,
    }


_re_sentencepiece_tokenizers = re.compile(r"^\s*if is_(sentencepiece|tokenizers)_available\(\):\s*$")


def clean_tokenization_in_init(init_file):
    """
    Removes all the import lines for tokenization in an init.
    """
    with open(init_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    new_lines = []
    idx = 0
    while idx < len(lines):
        # Conditional imports
        if _re_sentencepiece_tokenizers.search(lines[idx]) is not None:
            idx += 1
            while is_empty_line(lines[idx]):
                idx += 1
            indent = find_indent(lines[idx])
            while find_indent(lines[idx]) >= indent or is_empty_line(lines[idx]):
                idx += 1
        # Remove the import from file_utils
        elif "is_tokenizers_available" in lines[idx] or "is_sentencepiece_available" in lines[idx]:
            line = lines[idx].replace("is_tokenizers_available,", "").replace("is_sentencepiece_available,", "")
            # In case the , was not there
            line = line.replace("is_tokenizers_available,", "").replace("is_sentencepiece_available,", "")
            if len(line.strip()) > 0:
                new_lines.append(line)
            idx += 1
        elif re.search('^\s*"tokenization', lines[idx]) is None and re.search('^\s*from .tokenization', lines[idx]) is None:
            new_lines.append(lines[idx])
            idx += 1
        else:
            idx += 1
    
    with open(init_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def add_model_to_main_init(old_model_patterns, new_model_patterns, with_tokenizer=True):
    transformers_path = Path(auto_module.__path__[0]).parent.parent
    with open(transformers_path / "__init__.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    idx = 0
    new_lines = []
    while idx < len(lines):
        if f"models.{old_model_patterns.model_lower_cased}" in lines[idx]:
            block = [lines[idx]]
            indent = find_indent(lines[idx])
            idx += 1
            while find_indent(lines[idx]) > indent:
                block.append(lines[idx])
                idx += 1
            if lines[idx].strip() == ")":
                block.append(lines[idx])
                idx += 1
            block = "\n".join(block)
            new_lines.append(block)
            if not with_tokenizer:
                tokenizer_class = old_model_patterns.tokenizer_class
                block = block.replace(f' "{tokenizer_class},"', "")
                block = block.replace(f', "{tokenizer_class}"', "")
                block = block.replace(f" {tokenizer_class},", "")
                block = block.replace(f", {tokenizer_class}", "")
            if with_tokenizer or tokenizer_class not in block:
                new_lines.append(replace_model_patterns(block, old_model_patterns, new_model_patterns)[0])
        else:
            new_lines.append(lines[idx])
            idx += 1
    
    with open(transformers_path / "__init__.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def insert_tokenizer_in_auto_module(old_model_patterns, new_model_patterns):
    transformers_path = Path(auto_module.__path__[0]).parent.parent
    with open(transformers_path / "models" / "auto" / "tokenization_auto.py", "r", encoding="utf-8") as f:
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
        if old_model_patterns.model_type in block and old_model_patterns.tokenizer_class in block:
            break
    
    new_block = block.replace(old_model_patterns.model_type, new_model_patterns.model_type)
    new_block = new_block.replace(old_model_patterns.tokenizer_class, new_model_patterns.tokenizer_class)
    
    new_lines = lines[:idx] + [new_block] + lines[idx:]
    with open(transformers_path / "models" / "auto" / "tokenization_auto.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


AUTO_CLASSES_PATTERNS = {
    "configuration_auto.py": [
        '        ("{model_type}", "{model_name}"),',
        '        ("{model_type}", "{config_class}"),',
        '        ("{model_type}", "{pretrained_archive_map}"),',
    ],
    "modeling_auto.py": ['        ("{model_type}", "{any_pt_class}"),'],
    "modeling_tf_auto.py": ['        ("{model_type}", "{any_tf_class}"),'],
    "modeling_flax_auto.py": ['        ("{model_type}", "{any_flax_class}"),'],
}


def add_model_to_auto_classes(old_model_patterns, new_model_patterns, model_classes):
    transformers_path = Path(auto_module.__path__[0]).parent.parent

    for file in AUTO_CLASSES_PATTERNS:
        # Extend patterns with all model classes if necessary
        new_patterns = []
        for pattern in AUTO_CLASSES_PATTERNS[file]:
            if re.search("any_([a-z]*)_class", pattern) is not None:
                framework = re.search("any_([a-z]*)_class", pattern).groups()[0]
                if framework in model_classes:
                    new_patterns.extend(
                        [pattern.replace("{" + f"any_{framework}_class" + "}", cls) for cls in model_classes[framework]]
                    )
            else:
                new_patterns.append(pattern)

        # Loop through all patterns.
        for pattern in new_patterns:
            file_name = transformers_path / "models" / "auto" / file
            old_model_line = pattern
            new_model_line = pattern
            for attr in ["model_type", "model_name", "config_class"]:
                old_model_line = old_model_line.replace("{" + attr + "}", getattr(old_model_patterns, attr))
                new_model_line = new_model_line.replace("{" + attr + "}", getattr(new_model_patterns, attr))
            if "pretrained_archive_map" in pattern:
                old_model_line = old_model_line.replace("{pretrained_archive_map}", f"{old_model_patterns.model_upper_cased}_PRETRAINED_CONFIG_ARCHIVE_MAP")
                new_model_line = new_model_line.replace("{pretrained_archive_map}", f"{new_model_patterns.model_upper_cased}_PRETRAINED_CONFIG_ARCHIVE_MAP")

            new_model_line = new_model_line.replace(old_model_patterns.model_camel_cased, new_model_patterns.model_camel_cased)
            
            add_content_to_file(file_name, new_model_line, add_after=old_model_line)
    
    # Tokenizers require special handling
    insert_tokenizer_in_auto_module(old_model_patterns, new_model_patterns)


def create_new_model_like(model_type: str, new_model_patterns: ModelPatterns, add_copied_from: bool = True):
    """
    Creates a new model module like a given model of the Transformers library.
    
    Args:
        model_type (`str`): The model type to duplicate (like "bert" or "gpt2")
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        add_copied_from (`bool`, *optional*, defaults to `True`):
            Whether or not to add "Copied from" statements to all classes in the new model modeling files.
    """
    # Retrieve all the old model info.
    model_info = retrieve_info_for_model(model_type)
    model_files = model_info["model_files"]
    old_model_patterns = model_info["model_patterns"]
    keep_old_tokenizer = old_model_patterns.tokenizer_class == new_model_patterns.tokenizer_class
    model_classes = model_info["model_classes"]
    
    transformers_path = Path(auto_module.__path__[0]).parent.parent
    repo_path = transformers_path.parent.parent
    
    # 1. We create the module for our new model.
    old_module_name = model_files["module_name"]
    module_folder = transformers_path / "models" / new_model_patterns.model_lower_cased
    os.makedirs(module_folder, exist_ok=True)
    
    files_to_adapt = model_files["model_files"]
    if keep_old_tokenizer:
        files_to_adapt = [f for f in files_to_adapt if "tokenization" not in str(f)]
    
    os.makedirs(module_folder, exist_ok=True)
    for module_file in files_to_adapt:
        new_module_name = module_file.name.replace(old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased)
        dest_file = module_folder / new_module_name
        duplicate_module(
            module_file,
            old_model_patterns,
            new_model_patterns,
            dest_file=dest_file,
            add_copied_from=add_copied_from and "modeling" in new_module_name,
        )

    if keep_old_tokenizer:
        clean_tokenization_in_init(module_folder / "__init__.py")
    
    # 2. We add our new model to the models init and the main init
    add_content_to_file(
        transformers_path / "models" / "__init__.py",
        f"    {new_model_patterns.model_lower_cased},",
        add_after=f"    {old_module_name},",
        exact_match=True,
    )
    add_model_to_main_init(old_model_patterns, new_model_patterns, with_tokenizer=not keep_old_tokenizer)
    
    # 3. Add test files
    files_to_adapt = model_files["test_files"]
    if keep_old_tokenizer:
        files_to_adapt = [f for f in files_to_adapt if "tokenization" not in str(f)]
    
    for test_file in files_to_adapt:
        new_test_file_name = test_file.name.replace(old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased)
        dest_file = test_file.parent / new_test_file_name
        duplicate_module(
            test_file,
            old_model_patterns,
            new_model_patterns,
            dest_file=dest_file,
            add_copied_from=False,
        )
    
    # 4. Add model to auto classes
    add_model_to_auto_classes(old_model_patterns, new_model_patterns, model_classes)
    
    #5. Add doc file
    files_to_adapt = model_files["doc_files"]


def add_new_model_like_command_factory(args: Namespace):
    return AddNewModelLikeCommand(config_file=args.config_file)


class AddNewModelLikeCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_new_model_like_parser = parser.add_parser("add-new-model-like")
        add_new_model_like_parser.add_argument(
            "--config_file", type=str, help="A file with all the information for this model creation."
        )
        add_new_model_like_parser.set_defaults(func=add_new_model_like_command_factory)

    def __init__(self, config_file=None, *args):
        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.old_model_type = config["old_model_type"]
            self.model_patterns = ModelPatterns(**config["new_model_patterns"])
            self.add_copied_from = config.get("add_copied_from", True) 
            # Ignored for now
            self.frameworks = config.get("frameworks", ["pt", "tf", "flax"])

    def run(self):
        create_new_model_like(
            model_type=self.old_model_type,
            new_model_patterns=self.new_model_patterns,
            add_copied_from=self.add_copied_from,
        )
