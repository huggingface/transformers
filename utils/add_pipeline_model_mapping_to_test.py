# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""A script to add and/or update the attribute `pipeline_model_mapping` in model test files.

This script will be (mostly) used in the following 2 situations:

  - run within a (scheduled) CI job to:
    - check if model test files in the library have updated `pipeline_model_mapping`,
    - and/or update test files and (possibly) open a GitHub pull request automatically
  - being run by a `transformers` member to quickly check and update some particular test file(s)

This script is **NOT** intended to be run (manually) by community contributors.
"""

import argparse
import glob
import inspect
import os
import re
import unittest

from get_test_info import get_test_classes

from tests.test_pipeline_mixin import pipeline_test_mapping


PIPELINE_TEST_MAPPING = {}
for task, _ in pipeline_test_mapping.items():
    PIPELINE_TEST_MAPPING[task] = {"pt": None, "tf": None}


# DO **NOT** add item to this set (unless the reason is approved)
TEST_FILE_TO_IGNORE = {
    "tests/models/esm/test_modeling_esmfold.py",  # The pipeline test mapping is added to `test_modeling_esm.py`
}


def get_framework(test_class):
    """Infer the framework from the test class `test_class`."""

    if "ModelTesterMixin" in [x.__name__ for x in test_class.__bases__]:
        return "pt"
    elif "TFModelTesterMixin" in [x.__name__ for x in test_class.__bases__]:
        return "tf"
    elif "FlaxModelTesterMixin" in [x.__name__ for x in test_class.__bases__]:
        return "flax"
    else:
        return None


def get_mapping_for_task(task, framework):
    """Get mappings defined in `XXXPipelineTests` for the task `task`."""
    # Use the cached results
    if PIPELINE_TEST_MAPPING[task].get(framework, None) is not None:
        return PIPELINE_TEST_MAPPING[task][framework]

    pipeline_test_class = pipeline_test_mapping[task]["test"]
    mapping = None

    if framework == "pt":
        mapping = getattr(pipeline_test_class, "model_mapping", None)
    elif framework == "tf":
        mapping = getattr(pipeline_test_class, "tf_model_mapping", None)

    if mapping is not None:
        mapping = dict(mapping.items())

    # cache the results
    PIPELINE_TEST_MAPPING[task][framework] = mapping
    return mapping


def get_model_for_pipeline_test(test_class, task):
    """Get the model architecture(s) related to the test class `test_class` for a pipeline `task`."""
    framework = get_framework(test_class)
    if framework is None:
        return None
    mapping = get_mapping_for_task(task, framework)
    if mapping is None:
        return None

    config_classes = list({model_class.config_class for model_class in test_class.all_model_classes})
    if len(config_classes) != 1:
        raise ValueError("There should be exactly one configuration class from `test_class.all_model_classes`.")

    # This could be a list/tuple of model classes, but it's rare.
    model_class = mapping.get(config_classes[0], None)
    if isinstance(model_class, (tuple, list)):
        model_class = sorted(model_class, key=lambda x: x.__name__)

    return model_class


def get_pipeline_model_mapping(test_class):
    """Get `pipeline_model_mapping` for `test_class`."""
    mapping = [(task, get_model_for_pipeline_test(test_class, task)) for task in pipeline_test_mapping]
    mapping = sorted([(task, model) for task, model in mapping if model is not None], key=lambda x: x[0])

    return dict(mapping)


def get_pipeline_model_mapping_string(test_class):
    """Get `pipeline_model_mapping` for `test_class` as a string (to be added to the test file).

    This will be a 1-line string. After this is added to a test file, `make style` will format it beautifully.
    """
    framework = get_framework(test_class)
    if framework == "pt":
        framework = "torch"
    default_value = "{}"

    mapping = get_pipeline_model_mapping(test_class)
    if len(mapping) == 0:
        return ""

    texts = []
    for task, model_classes in mapping.items():
        if isinstance(model_classes, (tuple, list)):
            # A list/tuple of model classes
            value = "(" + ", ".join([x.__name__ for x in model_classes]) + ")"
        else:
            # A single model class
            value = model_classes.__name__
        texts.append(f'"{task}": {value}')
    text = "{" + ", ".join(texts) + "}"
    text = f"pipeline_model_mapping = {text} if is_{framework}_available() else {default_value}"

    return text


def is_valid_test_class(test_class):
    """Restrict to `XXXModelTesterMixin` and should be a subclass of `unittest.TestCase`."""
    base_class_names = {"ModelTesterMixin", "TFModelTesterMixin", "FlaxModelTesterMixin"}
    if not issubclass(test_class, unittest.TestCase):
        return False
    return len(base_class_names.intersection([x.__name__ for x in test_class.__bases__])) > 0


def find_test_class(test_file):
    """Find a test class in `test_file` to which we will add `pipeline_model_mapping`."""
    test_classes = [x for x in get_test_classes(test_file) if is_valid_test_class(x)]

    target_test_class = None
    for test_class in test_classes:
        # If a test class has defined `pipeline_model_mapping`, let's take it
        if getattr(test_class, "pipeline_model_mapping", None) is not None:
            target_test_class = test_class
            break
    # Take the test class with the shortest name (just a heuristic)
    if target_test_class is None and len(test_classes) > 0:
        target_test_class = sorted(test_classes, key=lambda x: (len(x.__name__), x.__name__))[0]

    return target_test_class


def find_block_ending(lines, start_idx, indent_level):
    end_idx = start_idx
    for idx, line in enumerate(lines[start_idx:]):
        indent = len(line) - len(line.lstrip())
        if idx == 0 or indent > indent_level or (indent == indent_level and line.strip() == ")"):
            end_idx = start_idx + idx
        elif idx > 0 and indent <= indent_level:
            # Outside the definition block of `pipeline_model_mapping`
            break

    return end_idx


def add_pipeline_model_mapping(test_class, overwrite=False):
    """Add `pipeline_model_mapping` to `test_class`."""
    if getattr(test_class, "pipeline_model_mapping", None) is not None:
        if not overwrite:
            return "", -1

    line_to_add = get_pipeline_model_mapping_string(test_class)
    if len(line_to_add) == 0:
        return "", -1
    line_to_add = line_to_add + "\n"

    # The code defined the class `test_class`
    class_lines, class_start_line_no = inspect.getsourcelines(test_class)
    # `inspect` gives the code for an object, including decorator(s) if any.
    # We (only) need the exact line of the class definition.
    for idx, line in enumerate(class_lines):
        if line.lstrip().startswith("class "):
            class_lines = class_lines[idx:]
            class_start_line_no += idx
            break
    class_end_line_no = class_start_line_no + len(class_lines) - 1

    # The index in `class_lines` that starts the definition of `all_model_classes`, `all_generative_model_classes` or
    # `pipeline_model_mapping`. This assumes they are defined in such order, and we take the start index of the last
    # block that appears in a `test_class`.
    start_idx = None
    # The indent level of the line at `class_lines[start_idx]` (if defined)
    indent_level = 0
    # To record if `pipeline_model_mapping` is found in `test_class`.
    def_line = None
    for idx, line in enumerate(class_lines):
        if line.strip().startswith("all_model_classes = "):
            indent_level = len(line) - len(line.lstrip())
            start_idx = idx
        elif line.strip().startswith("all_generative_model_classes = "):
            indent_level = len(line) - len(line.lstrip())
            start_idx = idx
        elif line.strip().startswith("pipeline_model_mapping = "):
            indent_level = len(line) - len(line.lstrip())
            start_idx = idx
            def_line = line
            break

    if start_idx is None:
        return "", -1
    # Find the ending index (inclusive) of the above found block.
    end_idx = find_block_ending(class_lines, start_idx, indent_level)

    # Extract `is_xxx_available()` from existing blocks: some models require specific libraries like `timm` and use
    # `is_timm_available()` instead of `is_torch_available()`.
    # Keep leading and trailing whitespaces
    r = re.compile(r"\s(is_\S+?_available\(\))\s")
    for line in class_lines[start_idx : end_idx + 1]:
        backend_condition = r.search(line)
        if backend_condition is not None:
            # replace the leading and trailing whitespaces to the space character " ".
            target = " " + backend_condition[0][1:-1] + " "
            line_to_add = r.sub(target, line_to_add)
            break

    if def_line is None:
        # `pipeline_model_mapping` is not defined. The target index is set to the ending index (inclusive) of
        # `all_model_classes` or `all_generative_model_classes`.
        target_idx = end_idx
    else:
        # `pipeline_model_mapping` is defined. The target index is set to be one **BEFORE** its start index.
        target_idx = start_idx - 1
        # mark the lines of the currently existing `pipeline_model_mapping` to be removed.
        for idx in range(start_idx, end_idx + 1):
            # These lines are going to be removed before writing to the test file.
            class_lines[idx] = None  # noqa

    # Make sure the test class is a subclass of `PipelineTesterMixin`.
    parent_classes = [x.__name__ for x in test_class.__bases__]
    if "PipelineTesterMixin" not in parent_classes:
        # Put `PipelineTesterMixin` just before `unittest.TestCase`
        _parent_classes = [x for x in parent_classes if x != "TestCase"] + ["PipelineTesterMixin"]
        if "TestCase" in parent_classes:
            # Here we **assume** the original string is always with `unittest.TestCase`.
            _parent_classes.append("unittest.TestCase")
        parent_classes = ", ".join(_parent_classes)
        for idx, line in enumerate(class_lines):
            # Find the ending of the declaration of `test_class`
            if line.strip().endswith("):"):
                # mark the lines of the declaration of `test_class` to be removed
                for _idx in range(idx + 1):
                    class_lines[_idx] = None  # noqa
                break
        # Add the new, one-line, class declaration for `test_class`
        class_lines[0] = f"class {test_class.__name__}({parent_classes}):\n"

    # Add indentation
    line_to_add = " " * indent_level + line_to_add
    # Insert `pipeline_model_mapping` to `class_lines`.
    # (The line at `target_idx` should be kept by definition!)
    class_lines = class_lines[: target_idx + 1] + [line_to_add] + class_lines[target_idx + 1 :]
    # Remove the lines that are marked to be removed
    class_lines = [x for x in class_lines if x is not None]

    # Move from test class to module (in order to write to the test file)
    module_lines = inspect.getsourcelines(inspect.getmodule(test_class))[0]
    # Be careful with the 1-off between line numbers and array indices
    module_lines = module_lines[: class_start_line_no - 1] + class_lines + module_lines[class_end_line_no:]
    code = "".join(module_lines)

    moddule_file = inspect.getsourcefile(test_class)
    with open(moddule_file, "w", encoding="UTF-8", newline="\n") as fp:
        fp.write(code)

    return line_to_add


def add_pipeline_model_mapping_to_test_file(test_file, overwrite=False):
    """Add `pipeline_model_mapping` to `test_file`."""
    test_class = find_test_class(test_file)
    if test_class:
        add_pipeline_model_mapping(test_class, overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_file", type=str, help="A path to the test file, starting with the repository's `tests` directory."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If to check and modify all test files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If to overwrite a test class if it has already defined `pipeline_model_mapping`.",
    )
    args = parser.parse_args()

    if not args.all and not args.test_file:
        raise ValueError("Please specify either `test_file` or pass `--all` to check/modify all test files.")
    elif args.all and args.test_file:
        raise ValueError("Only one of `--test_file` and `--all` could be specified.")

    test_files = []
    if args.test_file:
        test_files = [args.test_file]
    else:
        pattern = os.path.join("tests", "models", "**", "test_modeling_*.py")
        for test_file in glob.glob(pattern):
            # `Flax` is not concerned at this moment
            if not test_file.startswith("test_modeling_flax_"):
                test_files.append(test_file)

    for test_file in test_files:
        if test_file in TEST_FILE_TO_IGNORE:
            print(f"[SKIPPED] {test_file} is skipped as it is in `TEST_FILE_TO_IGNORE` in the file {__file__}.")
            continue
        add_pipeline_model_mapping_to_test_file(test_file, overwrite=args.overwrite)
