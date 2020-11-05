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

import argparse
import os
import re


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_dummies.py
PATH_TO_TRANSFORMERS = "src/transformers"

_re_single_line_import = re.compile(r"\s+from\s+\S*\s+import\s+([^\(\s].*)\n")

DUMMY_CONSTANT = """
{0} = None
"""

DUMMY_PT_PRETRAINED_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)
"""

DUMMY_PT_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
"""

DUMMY_PT_FUNCTION = """
def {0}(*args, **kwargs):
    requires_pytorch({0})
"""


DUMMY_TF_PRETRAINED_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_tf(self)

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_tf(self)
"""

DUMMY_TF_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_tf(self)
"""

DUMMY_TF_FUNCTION = """
def {0}(*args, **kwargs):
    requires_tf({0})
"""


DUMMY_FLAX_PRETRAINED_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_flax(self)

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_flax(self)
"""

DUMMY_FLAX_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_flax(self)
"""

DUMMY_FLAX_FUNCTION = """
def {0}(*args, **kwargs):
    requires_flax({0})
"""


DUMMY_SENTENCEPIECE_PRETRAINED_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_sentencepiece(self)

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_sentencepiece(self)
"""

DUMMY_SENTENCEPIECE_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_sentencepiece(self)
"""

DUMMY_SENTENCEPIECE_FUNCTION = """
def {0}(*args, **kwargs):
    requires_sentencepiece({0})
"""


DUMMY_TOKENIZERS_PRETRAINED_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_tokenizers(self)

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_tokenizers(self)
"""

DUMMY_TOKENIZERS_CLASS = """
class {0}:
    def __init__(self, *args, **kwargs):
        requires_tokenizers(self)
"""

DUMMY_TOKENIZERS_FUNCTION = """
def {0}(*args, **kwargs):
    requires_tokenizers({0})
"""

# Map all these to dummy type

DUMMY_PRETRAINED_CLASS = {
    "pt": DUMMY_PT_PRETRAINED_CLASS,
    "tf": DUMMY_TF_PRETRAINED_CLASS,
    "flax": DUMMY_FLAX_PRETRAINED_CLASS,
    "sentencepiece": DUMMY_SENTENCEPIECE_PRETRAINED_CLASS,
    "tokenizers": DUMMY_TOKENIZERS_PRETRAINED_CLASS,
}

DUMMY_CLASS = {
    "pt": DUMMY_PT_CLASS,
    "tf": DUMMY_TF_CLASS,
    "flax": DUMMY_FLAX_CLASS,
    "sentencepiece": DUMMY_SENTENCEPIECE_CLASS,
    "tokenizers": DUMMY_TOKENIZERS_CLASS,
}

DUMMY_FUNCTION = {
    "pt": DUMMY_PT_FUNCTION,
    "tf": DUMMY_TF_FUNCTION,
    "flax": DUMMY_FLAX_FUNCTION,
    "sentencepiece": DUMMY_SENTENCEPIECE_FUNCTION,
    "tokenizers": DUMMY_TOKENIZERS_FUNCTION,
}


def read_init():
    """ Read the init and exctracts PyTorch, TensorFlow, SentencePiece and Tokenizers objects. """
    with open(os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"), "r", encoding="utf-8") as f:
        lines = f.readlines()

    line_index = 0
    # Find where the SentencePiece imports begin
    sentencepiece_objects = []
    while not lines[line_index].startswith("if is_sentencepiece_available():"):
        line_index += 1
    line_index += 1

    # Until we unindent, add SentencePiece objects to the list
    while len(lines[line_index]) <= 1 or lines[line_index].startswith("    "):
        line = lines[line_index]
        search = _re_single_line_import.search(line)
        if search is not None:
            sentencepiece_objects += search.groups()[0].split(", ")
        elif line.startswith("        "):
            sentencepiece_objects.append(line[8:-2])
        line_index += 1

    # Find where the Tokenizers imports begin
    tokenizers_objects = []
    while not lines[line_index].startswith("if is_tokenizers_available():"):
        line_index += 1
    line_index += 1

    # Until we unindent, add Tokenizers objects to the list
    while len(lines[line_index]) <= 1 or lines[line_index].startswith("    "):
        line = lines[line_index]
        search = _re_single_line_import.search(line)
        if search is not None:
            tokenizers_objects += search.groups()[0].split(", ")
        elif line.startswith("        "):
            tokenizers_objects.append(line[8:-2])
        line_index += 1

    # Find where the PyTorch imports begin
    pt_objects = []
    while not lines[line_index].startswith("if is_torch_available():"):
        line_index += 1
    line_index += 1

    # Until we unindent, add PyTorch objects to the list
    while len(lines[line_index]) <= 1 or lines[line_index].startswith("    "):
        line = lines[line_index]
        search = _re_single_line_import.search(line)
        if search is not None:
            pt_objects += search.groups()[0].split(", ")
        elif line.startswith("        "):
            pt_objects.append(line[8:-2])
        line_index += 1

    # Find where the TF imports begin
    tf_objects = []
    while not lines[line_index].startswith("if is_tf_available():"):
        line_index += 1
    line_index += 1

    # Until we unindent, add PyTorch objects to the list
    while len(lines[line_index]) <= 1 or lines[line_index].startswith("    "):
        line = lines[line_index]
        search = _re_single_line_import.search(line)
        if search is not None:
            tf_objects += search.groups()[0].split(", ")
        elif line.startswith("        "):
            tf_objects.append(line[8:-2])
        line_index += 1

    # Find where the FLAX imports begin
    flax_objects = []
    while not lines[line_index].startswith("if is_flax_available():"):
        line_index += 1
    line_index += 1

    # Until we unindent, add PyTorch objects to the list
    while len(lines[line_index]) <= 1 or lines[line_index].startswith("    "):
        line = lines[line_index]
        search = _re_single_line_import.search(line)
        if search is not None:
            flax_objects += search.groups()[0].split(", ")
        elif line.startswith("        "):
            flax_objects.append(line[8:-2])
        line_index += 1

    return sentencepiece_objects, tokenizers_objects, pt_objects, tf_objects, flax_objects


def create_dummy_object(name, type="pt"):
    """ Create the code for the dummy object corresponding to `name`."""
    _pretrained = [
        "Config" "ForCausalLM",
        "ForConditionalGeneration",
        "ForMaskedLM",
        "ForMultipleChoice",
        "ForQuestionAnswering",
        "ForSequenceClassification",
        "ForTokenClassification",
        "Model",
        "Tokenizer",
    ]
    assert type in ["pt", "tf", "sentencepiece", "tokenizers", "flax"]
    if name.isupper():
        return DUMMY_CONSTANT.format(name)
    elif name.islower():
        return (DUMMY_FUNCTION[type]).format(name)
    else:
        is_pretrained = False
        for part in _pretrained:
            if part in name:
                is_pretrained = True
                break
        if is_pretrained:
            template = DUMMY_PRETRAINED_CLASS[type]
        else:
            template = DUMMY_CLASS[type]
        return template.format(name)


def create_dummy_files():
    """ Create the content of the dummy files. """
    sentencepiece_objects, tokenizers_objects, pt_objects, tf_objects, flax_objects = read_init()

    sentencepiece_dummies = "# This file is autogenerated by the command `make fix-copies`, do not edit.\n"
    sentencepiece_dummies += "from ..file_utils import requires_sentencepiece\n\n"
    sentencepiece_dummies += "\n".join([create_dummy_object(o, type="sentencepiece") for o in sentencepiece_objects])

    tokenizers_dummies = "# This file is autogenerated by the command `make fix-copies`, do not edit.\n"
    tokenizers_dummies += "from ..file_utils import requires_tokenizers\n\n"
    tokenizers_dummies += "\n".join([create_dummy_object(o, type="tokenizers") for o in tokenizers_objects])

    pt_dummies = "# This file is autogenerated by the command `make fix-copies`, do not edit.\n"
    pt_dummies += "from ..file_utils import requires_pytorch\n\n"
    pt_dummies += "\n".join([create_dummy_object(o, type="pt") for o in pt_objects])

    tf_dummies = "# This file is autogenerated by the command `make fix-copies`, do not edit.\n"
    tf_dummies += "from ..file_utils import requires_tf\n\n"
    tf_dummies += "\n".join([create_dummy_object(o, type="tf") for o in tf_objects])

    flax_dummies = "# This file is autogenerated by the command `make fix-copies`, do not edit.\n"
    flax_dummies += "from ..file_utils import requires_flax\n\n"
    flax_dummies += "\n".join([create_dummy_object(o, type="flax") for o in flax_objects])

    return sentencepiece_dummies, tokenizers_dummies, pt_dummies, tf_dummies, flax_dummies


def check_dummies(overwrite=False):
    """ Check if the dummy files are up to date and maybe `overwrite` with the right content. """
    sentencepiece_dummies, tokenizers_dummies, pt_dummies, tf_dummies, flax_dummies = create_dummy_files()
    path = os.path.join(PATH_TO_TRANSFORMERS, "utils")
    sentencepiece_file = os.path.join(path, "dummy_sentencepiece_objects.py")
    tokenizers_file = os.path.join(path, "dummy_tokenizers_objects.py")
    pt_file = os.path.join(path, "dummy_pt_objects.py")
    tf_file = os.path.join(path, "dummy_tf_objects.py")
    flax_file = os.path.join(path, "dummy_flax_objects.py")

    with open(sentencepiece_file, "r", encoding="utf-8") as f:
        actual_sentencepiece_dummies = f.read()
    with open(tokenizers_file, "r", encoding="utf-8") as f:
        actual_tokenizers_dummies = f.read()
    with open(pt_file, "r", encoding="utf-8") as f:
        actual_pt_dummies = f.read()
    with open(tf_file, "r", encoding="utf-8") as f:
        actual_tf_dummies = f.read()
    with open(flax_file, "r", encoding="utf-8") as f:
        actual_flax_dummies = f.read()

    if sentencepiece_dummies != actual_sentencepiece_dummies:
        if overwrite:
            print("Updating transformers.utils.dummy_sentencepiece_objects.py as the main __init__ has new objects.")
            with open(sentencepiece_file, "w", encoding="utf-8") as f:
                f.write(sentencepiece_dummies)
        else:
            raise ValueError(
                "The main __init__ has objects that are not present in transformers.utils.dummy_sentencepiece_objects.py.",
                "Run `make fix-copies` to fix this.",
            )

    if tokenizers_dummies != actual_tokenizers_dummies:
        if overwrite:
            print("Updating transformers.utils.dummy_tokenizers_objects.py as the main __init__ has new objects.")
            with open(tokenizers_file, "w", encoding="utf-8") as f:
                f.write(tokenizers_dummies)
        else:
            raise ValueError(
                "The main __init__ has objects that are not present in transformers.utils.dummy_tokenizers_objects.py.",
                "Run `make fix-copies` to fix this.",
            )

    if pt_dummies != actual_pt_dummies:
        if overwrite:
            print("Updating transformers.utils.dummy_pt_objects.py as the main __init__ has new objects.")
            with open(pt_file, "w", encoding="utf-8") as f:
                f.write(pt_dummies)
        else:
            raise ValueError(
                "The main __init__ has objects that are not present in transformers.utils.dummy_pt_objects.py.",
                "Run `make fix-copies` to fix this.",
            )

    if tf_dummies != actual_tf_dummies:
        if overwrite:
            print("Updating transformers.utils.dummy_tf_objects.py as the main __init__ has new objects.")
            with open(tf_file, "w", encoding="utf-8") as f:
                f.write(tf_dummies)
        else:
            raise ValueError(
                "The main __init__ has objects that are not present in transformers.utils.dummy_pt_objects.py.",
                "Run `make fix-copies` to fix this.",
            )

    if flax_dummies != actual_flax_dummies:
        if overwrite:
            print("Updating transformers.utils.dummy_flax_objects.py as the main __init__ has new objects.")
            with open(flax_file, "w", encoding="utf-8") as f:
                f.write(flax_dummies)
        else:
            raise ValueError(
                "The main __init__ has objects that are not present in transformers.utils.dummy_flax_objects.py.",
                "Run `make fix-copies` to fix this.",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_dummies(args.fix_and_overwrite)
