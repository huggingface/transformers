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

import importlib
import os
import sys


# This is required to make the module import works (when the python process is running from the root of the repo)
sys.path.append(".")


r"""
The argument `test_file` in this file refers to a model test file. This should be a string of the from
`tests/models/*/test_modeling_*.py`.
"""


def get_module_path(test_file):
    """Return the module path of a model test file."""
    components = test_file.split(os.path.sep)
    if components[0:2] != ["tests", "models"]:
        raise ValueError(
            "`test_file` should start with `tests/models/` (with `/` being the OS specific path separator). Got "
            f"{test_file} instead."
        )
    test_fn = components[-1]
    if not test_fn.endswith("py"):
        raise ValueError(f"`test_file` should be a python file. Got {test_fn} instead.")
    if not test_fn.startswith("test_modeling_"):
        raise ValueError(
            f"`test_file` should point to a file name of the form `test_modeling_*.py`. Got {test_fn} instead."
        )

    components = components[:-1] + [test_fn.replace(".py", "")]
    test_module_path = ".".join(components)

    return test_module_path


def get_test_module(test_file):
    """Get the module of a model test file."""
    test_module_path = get_module_path(test_file)
    try:
        test_module = importlib.import_module(test_module_path)
    except AttributeError as exc:
        # e.g. if you have a `tests` folder in `site-packages`, created by another package, when trying to import
        # `tests.models...`
        raise ValueError(
            f"Could not import module {test_module_path}. Confirm that you don't have a package with the same root "
            "name installed or in your environment's `site-packages`."
        ) from exc

    return test_module


def get_tester_classes(test_file):
    """Get all classes in a model test file whose names ends with `ModelTester`."""
    tester_classes = []
    test_module = get_test_module(test_file)
    for attr in dir(test_module):
        if attr.endswith("ModelTester"):
            tester_classes.append(getattr(test_module, attr))

    # sort with class names
    return sorted(tester_classes, key=lambda x: x.__name__)


def get_test_classes(test_file):
    """Get all [test] classes in a model test file with attribute `all_model_classes` that are non-empty.

    These are usually the (model) test classes containing the (non-slow) tests to run and are subclasses of one of the
    classes `ModelTesterMixin`, `TFModelTesterMixin` or `FlaxModelTesterMixin`, as well as a subclass of
    `unittest.TestCase`. Exceptions include `RagTestMixin` (and its subclasses).
    """
    test_classes = []
    test_module = get_test_module(test_file)
    for attr in dir(test_module):
        attr_value = getattr(test_module, attr)
        # (TF/Flax)ModelTesterMixin is also an attribute in specific model test module. Let's exclude them by checking
        # `all_model_classes` is not empty (which also excludes other special classes).
        model_classes = getattr(attr_value, "all_model_classes", [])
        if len(model_classes) > 0:
            test_classes.append(attr_value)

    # sort with class names
    return sorted(test_classes, key=lambda x: x.__name__)


def get_model_classes(test_file):
    """Get all model classes that appear in `all_model_classes` attributes in a model test file."""
    test_classes = get_test_classes(test_file)
    model_classes = set()
    for test_class in test_classes:
        model_classes.update(test_class.all_model_classes)

    # sort with class names
    return sorted(model_classes, key=lambda x: x.__name__)


def get_model_tester_from_test_class(test_class):
    """Get the model tester class of a model test class."""
    test = test_class()
    if hasattr(test, "setUp"):
        test.setUp()

    model_tester = None
    if hasattr(test, "model_tester"):
        # `(TF/Flax)ModelTesterMixin` has this attribute default to `None`. Let's skip this case.
        if test.model_tester is not None:
            model_tester = test.model_tester.__class__

    return model_tester


def get_test_classes_for_model(test_file, model_class):
    """Get all [test] classes in `test_file` that have `model_class` in their `all_model_classes`."""
    test_classes = get_test_classes(test_file)

    target_test_classes = []
    for test_class in test_classes:
        if model_class in test_class.all_model_classes:
            target_test_classes.append(test_class)

    # sort with class names
    return sorted(target_test_classes, key=lambda x: x.__name__)


def get_tester_classes_for_model(test_file, model_class):
    """Get all model tester classes in `test_file` that are associated to `model_class`."""
    test_classes = get_test_classes_for_model(test_file, model_class)

    tester_classes = []
    for test_class in test_classes:
        tester_class = get_model_tester_from_test_class(test_class)
        if tester_class is not None:
            tester_classes.append(tester_class)

    # sort with class names
    return sorted(tester_classes, key=lambda x: x.__name__)


def get_test_to_tester_mapping(test_file):
    """Get a mapping from [test] classes to model tester classes in `test_file`.

    This uses `get_test_classes` which may return classes that are NOT subclasses of `unittest.TestCase`.
    """
    test_classes = get_test_classes(test_file)
    test_tester_mapping = {test_class: get_model_tester_from_test_class(test_class) for test_class in test_classes}
    return test_tester_mapping


def get_model_to_test_mapping(test_file):
    """Get a mapping from model classes to test classes in `test_file`."""
    model_classes = get_model_classes(test_file)
    model_test_mapping = {
        model_class: get_test_classes_for_model(test_file, model_class) for model_class in model_classes
    }
    return model_test_mapping


def get_model_to_tester_mapping(test_file):
    """Get a mapping from model classes to model tester classes in `test_file`."""
    model_classes = get_model_classes(test_file)
    model_to_tester_mapping = {
        model_class: get_tester_classes_for_model(test_file, model_class) for model_class in model_classes
    }
    return model_to_tester_mapping


def to_json(o):
    """Make the information succinct and easy to read.

    Avoid the full class representation like `<class 'transformers.models.bert.modeling_bert.BertForMaskedLM'>` when
    displaying the results. Instead, we use class name (`BertForMaskedLM`) for the readability.
    """
    if isinstance(o, str):
        return o
    elif isinstance(o, type):
        return o.__name__
    elif isinstance(o, (list, tuple)):
        return [to_json(x) for x in o]
    elif isinstance(o, dict):
        return {to_json(k): to_json(v) for k, v in o.items()}
    else:
        return o
