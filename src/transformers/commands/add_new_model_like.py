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
import os
import re
import subprocess
import textwrap
from argparse import ArgumentParser, Namespace
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ..models.auto.configuration_auto import CONFIG_MAPPING_NAMES, MODEL_NAMES_MAPPING
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
from ..models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
from ..models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from ..models.auto.video_processing_auto import VIDEO_PROCESSOR_MAPPING_NAMES
from ..utils import is_libcst_available
from . import BaseTransformersCLICommand
from .add_fast_image_processor import add_fast_image_processor


# We protect this import to avoid requiring it for all `transformers` CLI commands - however it is actually
# strictly required for this one (we need it both for modular and for the following Visitor)
if is_libcst_available():
    import libcst as cst
    from libcst import CSTVisitor
    from libcst import matchers as m

    class ClassFinder(CSTVisitor):
        """
        A visitor to find all classes in a python module.
        """

        def __init__(self):
            self.classes: list = []
            self.public_classes: list = []
            self.is_in_class = False

        def visit_ClassDef(self, node: cst.ClassDef) -> None:
            """Record class names. We assume classes always only appear at top-level (i.e. no class definition in function or similar)"""
            self.classes.append(node.name.value)
            self.is_in_class = True

        def leave_ClassDef(self, node: cst.ClassDef):
            self.is_in_class = False

        def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine):
            """Record all public classes inside the `__all__` assignment."""
            simple_top_level_assign_structure = m.SimpleStatementLine(
                body=[m.Assign(targets=[m.AssignTarget(target=m.Name())])]
            )
            if not self.is_in_class and m.matches(node, simple_top_level_assign_structure):
                assigned_variable = node.body[0].targets[0].target.value
                if assigned_variable == "__all__":
                    elements = node.body[0].value.elements
                    self.public_classes = [element.value.value for element in elements]


CURRENT_YEAR = date.today().year
TRANSFORMERS_PATH = Path(__file__).parent.parent
REPO_PATH = TRANSFORMERS_PATH.parent.parent

COPYRIGHT = f"""
# coding=utf-8
# Copyright {CURRENT_YEAR} the HuggingFace Team. All rights reserved.
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
""".lstrip()


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
            self.tokenizer_class, self.fast_tokenizer_class = TOKENIZER_MAPPING_NAMES[self.lowercase_name]
            self.fast_tokenizer_class = (
                None if self.fast_tokenizer_class == "PreTrainedTokenizerFast" else self.fast_tokenizer_class
            )
        else:
            self.tokenizer_class, self.fast_tokenizer_class = None, None

        self.image_processor_class, self.fast_image_processor_class = IMAGE_PROCESSOR_MAPPING_NAMES.get(
            self.lowercase_name, (None, None)
        )
        self.video_processor_class = VIDEO_PROCESSOR_MAPPING_NAMES.get(self.lowercase_name, None)
        self.feature_extractor_class = FEATURE_EXTRACTOR_MAPPING_NAMES.get(self.lowercase_name, None)
        self.processor_class = PROCESSOR_MAPPING_NAMES.get(self.lowercase_name, None)


def add_content_to_file(file_name: Union[str, os.PathLike], new_content: str, add_after: str):
    """
    A utility to add some content inside a given file.

    Args:
        file_name (`str` or `os.PathLike`):
            The name of the file in which we want to insert some content.
        new_content (`str`):
            The content to add.
       add_after (`str`):
           The new content is added just after the first instance matching it.
    """
    with open(file_name, "r", encoding="utf-8") as f:
        old_content = f.read()

    before, after = old_content.split(add_after, 1)
    new_content = before + add_after + new_content + after

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(new_content)


def add_model_to_auto_mappings(
    old_model_infos: ModelInfos,
    new_lowercase_name: str,
    new_model_paper_name: str,
    filenames_to_add: list[tuple[str, bool]],
):
    """
    Add a model to all the relevant mappings in the auto module.

    Args:
        old_model_infos (`ModelInfos`):
            The structure containing the class informations of the old model.
        new_lowercase_name (`str`):
            The new lowercase model name.
        new_model_paper_name (`str`):
            The fully cased name (as in the official paper name) of the new model.
        filenames_to_add (`list[tuple[str, bool]]`):
            A list of tuples of all potential filenames to add for a new model, along a boolean flag describing if we
            should add this file or not. For example, [(`modeling_xxx.px`, True), (`configuration_xxx.py`, True), (`tokenization_xxx.py`, False),...]
    """
    new_cased_name = "".join(x.title() for x in new_lowercase_name.replace("-", "_").split("_"))
    old_lowercase_name = old_model_infos.lowercase_name
    old_cased_name = old_model_infos.camelcase_name
    filenames_to_add = [
        (filename.replace(old_lowercase_name, "auto"), to_add) for filename, to_add in filenames_to_add[1:]
    ]
    # fast tokenizer/image processor have the same auto mappings as normal ones
    corrected_filenames_to_add = []
    for file, to_add in filenames_to_add:
        if re.search(r"(?:tokenization)|(?:image_processing)_auto_fast.py", file):
            previous_file, previous_to_add = corrected_filenames_to_add[-1]
            corrected_filenames_to_add[-1] = (previous_file, previous_to_add or to_add)
        else:
            corrected_filenames_to_add.append((file, to_add))

    # Add the config mappings directly as the handling for config is a bit different
    add_content_to_file(
        TRANSFORMERS_PATH / "models" / "auto" / "configuration_auto.py",
        new_content=f'        ("{new_lowercase_name}", "{new_cased_name}Config"),\n',
        add_after="CONFIG_MAPPING_NAMES = OrderedDict[str, str](\n    [\n        # Add configs here\n",
    )
    add_content_to_file(
        TRANSFORMERS_PATH / "models" / "auto" / "configuration_auto.py",
        new_content=f'        ("{new_lowercase_name}", "{new_model_paper_name}"),\n',
        add_after="MODEL_NAMES_MAPPING = OrderedDict[str, str](\n    [\n        # Add full (and cased) model names here\n",
    )

    for filename, to_add in corrected_filenames_to_add:
        if to_add:
            # The auto mapping
            filename = filename.replace("_fast.py", ".py")
            with open(TRANSFORMERS_PATH / "models" / "auto" / filename) as f:
                file = f.read()
            # The regex has to be a bit complex like this as the tokenizer mapping has new lines everywhere
            matching_lines = re.findall(
                rf'( {{8,12}}\(\s*"{old_lowercase_name}",.*?\),\n)(?: {{4,12}}\(|\])', file, re.DOTALL
            )
            for match in matching_lines:
                add_content_to_file(
                    TRANSFORMERS_PATH / "models" / "auto" / filename,
                    new_content=match.replace(old_lowercase_name, new_lowercase_name).replace(
                        old_cased_name, new_cased_name
                    ),
                    add_after=match,
                )


def create_doc_file(new_paper_name: str, public_classes: list[str]):
    """
    Create a new doc file to fill for the new model.

    Args:
        new_paper_name (`str`):
            The fully cased name (as in the official paper name) of the new model.
        public_classes (`list[str]`):
            A list of all the public classes that the model will have in the library.
    """
    added_note = (
        "\n\n⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that "
        "may not be rendered properly in your Markdown viewer.\n\n-->\n\n"
    )
    copyright_for_markdown = re.sub(r"# ?", "", COPYRIGHT).replace("coding=utf-8\n", "<!--") + added_note

    doc_template = textwrap.dedent(
        f"""
        # {new_paper_name}

        ## Overview

        The {new_paper_name} model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
        <INSERT SHORT SUMMARY HERE>

        The abstract from the paper is the following:

        <INSERT PAPER ABSTRACT HERE>

        Tips:

        <INSERT TIPS ABOUT MODEL HERE>

        This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
        The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

        ## Usage examples

        <INSERT SOME NICE EXAMPLES HERE>

        """
    )

    # Add public classes doc
    doc_for_classes = []
    for class_ in public_classes:
        doc = f"## {class_}\n\n[[autodoc]] {class_}"
        if "Model" in class_:
            doc += "\n    - forward"
        doc_for_classes.append(doc)

    class_doc = "\n\n".join(doc_for_classes)

    return copyright_for_markdown + doc_template + class_doc


def insert_model_in_doc_toc(old_lowercase_name: str, new_lowercase_name: str, new_model_paper_name: str):
    """
    Insert the new model in the doc `_toctree.yaml`, in the same section as the old model.

    Args:
        old_lowercase_name (`str`):
            The old lowercase model name.
        new_lowercase_name (`str`):
            The old lowercase model name.
        new_model_paper_name (`str`):
            The fully cased name (as in the official paper name) of the new model.
    """
    toc_file = REPO_PATH / "docs" / "source" / "en" / "_toctree.yml"
    with open(toc_file, "r") as f:
        content = f.read()

    old_model_toc = re.search(rf"- local: model_doc/{old_lowercase_name}\n {{8}}title: .*?\n", content).group(0)
    new_toc = f"      - local: model_doc/{new_lowercase_name}\n        title: {new_model_paper_name}\n"
    add_content_to_file(
        REPO_PATH / "docs" / "source" / "en" / "_toctree.yml", new_content=new_toc, add_after=old_model_toc
    )


def create_init_file(old_lowercase_name: str, new_lowercase_name: str, filenames_to_add: list[tuple[str, bool]]):
    """
    Create the `__init__.py` file to add in the new model folder.

    Args:
        old_lowercase_name (`str`):
            The old lowercase model name.
        new_lowercase_name (`str`):
            The new lowercase model name.
        filenames_to_add (`list[tuple[str, bool]]`):
            A list of tuples of all potential filenames to add for a new model, along a boolean flag describing if we
            should add this file or not. For example, [(`modeling_xxx.px`, True), (`configuration_xxx.py`, True), (`tokenization_xxx.py`, False),...]
    """
    filenames_to_add = [
        (filename.replace(old_lowercase_name, new_lowercase_name).replace(".py", ""), to_add)
        for filename, to_add in filenames_to_add
    ]
    imports = "\n            ".join(f"from .{file} import *" for file, to_add in filenames_to_add if to_add)
    init_file = COPYRIGHT + textwrap.dedent(
        f"""
        from typing import TYPE_CHECKING

        from ...utils import _LazyModule
        from ...utils.import_utils import define_import_structure


        if TYPE_CHECKING:
            {imports}
        else:
            import sys

            _file = globals()["__file__"]
            sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
        """
    )
    return init_file


def find_all_classes_from_file(module_name: str) -> set:
    """
    Find the name of all classes defined in `module_name`, including public ones (defined in `__all__`).

    Args:
        module_name (`str`):
            The full path to the python module from which to extract classes.
    """
    with open(module_name, "r", encoding="utf-8") as file:
        source_code = file.read()
    module = cst.parse_module(source_code)
    visitor = ClassFinder()
    module.visit(visitor)
    return visitor.classes, visitor.public_classes


def find_modular_structure(
    module_name: str, old_model_infos: ModelInfos, new_cased_name: str
) -> tuple[str, str, list]:
    """
    Extract the modular structure that will be needed to copy a file `module_name` using modular.

    Args:
        module_name (`str`):
            The full path to the python module to copy with modular.
        old_model_infos (`ModelInfos`):
            The structure containing the class informations of the old model.
        new_cased_name (`str`):
            The new cased model name.
    """
    all_classes, public_classes = find_all_classes_from_file(module_name)
    import_location = ".".join(module_name.parts[-2:]).replace(".py", "")
    old_cased_name = old_model_infos.camelcase_name
    imports = f"from ..{import_location} import {', '.join(class_ for class_ in all_classes)}"
    modular_classes = "\n\n".join(
        f"class {class_.replace(old_cased_name, new_cased_name)}({class_}):\n    pass" for class_ in all_classes
    )
    public_classes = [class_.replace(old_cased_name, new_cased_name) for class_ in public_classes]
    return imports, modular_classes, public_classes


def create_modular_file(
    old_model_infos: ModelInfos,
    new_lowercase_name: str,
    filenames_to_add: list[tuple[str, bool]],
) -> str:
    """
    Create a new modular file which will copy the old model, based on the new name and the different filenames
    (modules) to add.

    Args:
        old_model_infos (`ModelInfos`):
            The structure containing the class informations of the old model.
        new_lowercase_name (`str`):
            The new lowercase model name.
        filenames_to_add (`list[tuple[str, bool]]`):
            A list of tuples of all potential filenames to add for a new model, along a boolean flag describing if we
            should add this file or not. For example, [(`modeling_xxx.px`, True), (`configuration_xxx.py`, True), (`tokenization_xxx.py`, False),...]
    """
    new_cased_name = "".join(x.title() for x in new_lowercase_name.replace("-", "_").split("_"))
    old_lowercase_name = old_model_infos.lowercase_name
    old_folder_root = TRANSFORMERS_PATH / "models" / old_lowercase_name

    # Construct the modular file from the original (old) model, by subclassing each class
    all_imports = ""
    all_bodies = ""
    all_public_classes = []
    for filename, to_add in filenames_to_add:
        if to_add:
            imports, body, public_classes = find_modular_structure(
                old_folder_root / filename, old_model_infos, new_cased_name
            )
            all_imports += f"\n{imports}"
            all_bodies += f"\n\n{body}"
            all_public_classes.extend(public_classes)

    # Create the __all__ assignment
    public_classes_formatted = "\n            ".join(f"{public_class}," for public_class in all_public_classes)
    all_statement = textwrap.dedent(
        f"""

        __all__ = [
            {public_classes_formatted}
        ]
        """
    )
    # Create the whole modular file
    modular_file = COPYRIGHT + all_imports + all_bodies + all_statement
    # Remove outer explicit quotes "" around the public class names before returning them
    all_public_classes = [public_class.replace('"', "") for public_class in all_public_classes]
    return modular_file, all_public_classes


def create_test_files(old_model_infos: ModelInfos, new_lowercase_name, filenames_to_add: list[tuple[str, bool]]):
    """
    Create the test files for the new model. It basically copies over the old test files and adjust the class names.

    Args:
        old_model_infos (`ModelInfos`):
            The structure containing the class informations of the old model.
        new_lowercase_name (`str`):
            The new lowercase model name.
        filenames_to_add (`list[tuple[str, bool]]`):
            A list of tuples of all potential filenames to add for a new model, along a boolean flag describing if we
            should add this file or not. For example, [(`modeling_xxx.px`, True), (`configuration_xxx.py`, True), (`tokenization_xxx.py`, False),...]
    """
    new_cased_name = "".join(x.title() for x in new_lowercase_name.replace("-", "_").split("_"))
    old_lowercase_name = old_model_infos.lowercase_name
    old_cased_name = old_model_infos.camelcase_name
    filenames_to_add = [
        ("test_" + filename.replace(old_lowercase_name, new_lowercase_name), to_add)
        for filename, to_add in filenames_to_add[1:]
    ]
    # fast tokenizer/image processor have the same test files as normal ones
    corrected_filenames_to_add = []
    for file, to_add in filenames_to_add:
        if re.search(rf"test_(?:tokenization)|(?:image_processing)_{new_lowercase_name}_fast.py", file):
            previous_file, previous_to_add = corrected_filenames_to_add[-1]
            corrected_filenames_to_add[-1] = (previous_file, previous_to_add or to_add)
        else:
            corrected_filenames_to_add.append((file, to_add))

    test_files = {}
    for new_file, to_add in corrected_filenames_to_add:
        if to_add:
            original_test_file = new_file.replace(new_lowercase_name, old_lowercase_name)
            original_test_path = REPO_PATH / "tests" / "models" / old_lowercase_name / original_test_file
            # Sometimes, tests may not exist
            if not original_test_path.is_file():
                continue
            with open(original_test_path, "r") as f:
                test_code = f.read()
            # Remove old copyright and add new one
            test_lines = test_code.split("\n")
            idx = 0
            while test_lines[idx].startswith("#"):
                idx += 1
            test_code = COPYRIGHT + "\n".join(test_lines[idx:])
            test_files[new_file] = test_code.replace(old_cased_name, new_cased_name)

    return test_files


def create_new_model_like(
    old_model_infos: ModelInfos,
    new_lowercase_name: str,
    new_model_paper_name: str,
    filenames_to_add: list[tuple[str, bool]],
    create_fast_image_processor: bool,
):
    """
    Creates a new model module like a given model of the Transformers library.

    Args:
        old_model_infos (`ModelInfos`):
            The structure containing the class informations of the old model.
        new_lowercase_name (`str`):
            The new lowercase model name.
        new_model_paper_name (`str`):
            The fully cased name (as in the official paper name) of the new model.
        filenames_to_add (`list[tuple[str, bool]]`):
            A list of tuples of all potential filenames to add for a new model, along a boolean flag describing if we
            should add this file or not. For example, [(`modeling_xxx.px`, True), (`configuration_xxx.py`, True), (`tokenization_xxx.py`, False),...]
        create_fast_image_processor (`bool`):
            If it makes sense, whether to add a fast processor as well, even if the old model does not have one.
    """
    # As the import was protected, raise if not present (as it's actually a hard dependency for this command)
    if not is_libcst_available():
        raise ValueError("You need to install `libcst` to run this command -> `pip install libcst`")

    old_lowercase_name = old_model_infos.lowercase_name

    # 1. We create the folder for our new model
    new_module_folder = TRANSFORMERS_PATH / "models" / new_lowercase_name
    os.makedirs(new_module_folder, exist_ok=True)

    # 2. Create and add the modular file
    modular_file, public_classes = create_modular_file(old_model_infos, new_lowercase_name, filenames_to_add)
    with open(new_module_folder / f"modular_{new_lowercase_name}.py", "w") as f:
        f.write(modular_file)

    # 3. Create and add the __init__.py
    init_file = create_init_file(old_lowercase_name, new_lowercase_name, filenames_to_add)
    with open(new_module_folder / "__init__.py", "w") as f:
        f.write(init_file)

    # 4. Add new model to the models init
    add_content_to_file(
        TRANSFORMERS_PATH / "models" / "__init__.py",
        new_content=f"    from .{new_lowercase_name} import *\n",
        add_after="if TYPE_CHECKING:\n",
    )

    # 5. Add model to auto mappings
    add_model_to_auto_mappings(old_model_infos, new_lowercase_name, new_model_paper_name, filenames_to_add)

    # 6. Add test files
    tests_folder = REPO_PATH / "tests" / "models" / new_lowercase_name
    os.makedirs(tests_folder, exist_ok=True)
    # Add empty __init__.py
    with open(tests_folder / "__init__.py", "w"):
        pass
    test_files = create_test_files(old_model_infos, new_lowercase_name, filenames_to_add)
    for filename, content in test_files.items():
        with open(tests_folder / filename, "w") as f:
            f.write(content)

    # 7. Add doc file
    doc_file = create_doc_file(new_model_paper_name, public_classes)
    with open(REPO_PATH / "docs" / "source" / "en" / "model_doc" / f"{new_lowercase_name}.md", "w") as f:
        f.write(doc_file)
    insert_model_in_doc_toc(old_lowercase_name, new_lowercase_name, new_model_paper_name)

    # 8. Add additional fast image processor if necessary
    if create_fast_image_processor:
        add_fast_image_processor(model_name=new_lowercase_name)

    # 9. Run linters
    model_init_file = TRANSFORMERS_PATH / "models" / "__init__.py"
    subprocess.run(
        ["ruff", "check", new_module_folder, tests_folder, model_init_file, "--fix"],
        cwd=REPO_PATH,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ruff", "format", new_module_folder, tests_folder, model_init_file],
        cwd=REPO_PATH,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["python", "utils/check_doc_toc.py", "--fix_and_overwrite"], cwd=REPO_PATH, stdout=subprocess.DEVNULL
    )
    subprocess.run(["python", "utils/sort_auto_mappings.py"], cwd=REPO_PATH, stdout=subprocess.DEVNULL)

    # 10. Run the modular conversion
    subprocess.run(
        ["python", "utils/modular_model_converter.py", new_lowercase_name], cwd=REPO_PATH, stdout=subprocess.DEVNULL
    )


def get_user_field(
    question: str,
    default_value: Optional[str] = None,
    convert_to: Optional[Callable] = None,
    fallback_message: Optional[str] = None,
) -> Any:
    """
    A utility function that asks a question to the user to get an answer, potentially looping until it gets a valid
    answer.

    Args:
        question (`str`):
            The question to ask the user.
        default_value (`str`, *optional*):
            A potential default value that will be used when the answer is empty.
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
        if convert_to is not None:
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
            "What model would you like to duplicate? Please provide it as lowercase, e.g. `llama`): "
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

    old_model_infos = ModelInfos(old_model_type)

    # Ask for the new model name
    new_lowercase_name = get_user_field(
        "What is the new model name? Please provide it as snake lowercase, e.g. `new_model`?"
    )
    new_model_paper_name = get_user_field(
        "What is the fully cased name you would like to appear in the doc (e.g. `NeW ModEl`)? ",
        default_value="".join(x.title() for x in new_lowercase_name.split("_")),
    )

    # Ask if we want to add individual processor classes as well
    add_tokenizer = False
    add_fast_tokenizer = False
    add_image_processor = False
    add_fast_image_processor = False
    add_video_processor = False
    add_feature_extractor = False
    add_processor = False
    if old_model_infos.tokenizer_class is not None:
        add_tokenizer = get_user_field(
            f"Do you want to create a new tokenizer? If `no`, it will use the same as {old_model_type} (y/n)?",
            convert_to=convert_to_bool,
            fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
        )
    if old_model_infos.fast_tokenizer_class is not None:
        add_fast_tokenizer = get_user_field(
            f"Do you want to create a new fast tokenizer? If `no`, it will use the same as {old_model_type} (y/n)?",
            convert_to=convert_to_bool,
            fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
        )
    if old_model_infos.image_processor_class is not None:
        add_image_processor = get_user_field(
            f"Do you want to create a new image processor? If `no`, it will use the same as {old_model_type} (y/n)?",
            convert_to=convert_to_bool,
            fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
        )
    if old_model_infos.fast_image_processor_class is not None:
        add_fast_image_processor = get_user_field(
            f"Do you want to create a new fast image processor? If `no`, it will use the same as {old_model_type} (y/n)?",
            convert_to=convert_to_bool,
            fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
        )
    if old_model_infos.video_processor_class is not None:
        add_video_processor = get_user_field(
            f"Do you want to create a new video processor? If `no`, it will use the same as {old_model_type} (y/n)?",
            convert_to=convert_to_bool,
            fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
        )
    if old_model_infos.feature_extractor_class is not None:
        add_feature_extractor = get_user_field(
            f"Do you want to create a new feature extractor? If `no`, it will use the same as {old_model_type} (y/n)?",
            convert_to=convert_to_bool,
            fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
        )
    if old_model_infos.processor_class is not None:
        add_processor = get_user_field(
            f"Do you want to create a new processor? If `no`, it will use the same as {old_model_type} (y/n)?",
            convert_to=convert_to_bool,
            fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
        )

    old_lowercase_name = old_model_infos.lowercase_name
    # A list of the old filenames, along whether we should copy them or not
    filenames_to_add = (
        (f"configuration_{old_lowercase_name}.py", True),
        (f"modeling_{old_lowercase_name}.py", True),
        (f"tokenization_{old_lowercase_name}.py", add_tokenizer),
        (f"tokenization_{old_lowercase_name}_fast.py", add_fast_tokenizer),
        (f"image_processing_{old_lowercase_name}.py", add_image_processor),
        (f"image_processing_{old_lowercase_name}_fast.py", add_fast_image_processor),
        (f"video_processing_{old_lowercase_name}.py", add_video_processor),
        (f"feature_extraction_{old_lowercase_name}.py", add_feature_extractor),
        (f"processing_{old_lowercase_name}.py", add_processor),
    )

    create_fast_image_processor = False
    if add_image_processor and not add_fast_image_processor:
        create_fast_image_processor = get_user_field(
            "A fast image processor can be created from the slow one, but modifications might be needed. "
            "Should we add a fast image processor class for this model (recommended) (y/n)? ",
            convert_to=convert_to_bool,
            default_value="y",
            fallback_message="Please answer yes/no, y/n, true/false or 1/0.",
        )

    return old_model_infos, new_lowercase_name, new_model_paper_name, filenames_to_add, create_fast_image_processor


def add_new_model_like_command_factory(args: Namespace):
    return AddNewModelLikeCommand(path_to_repo=args.path_to_repo)


class AddNewModelLikeCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_new_model_like_parser = parser.add_parser("add-new-model-like")
        add_new_model_like_parser.add_argument(
            "--path_to_repo", type=str, help="When not using an editable install, the path to the Transformers repo."
        )
        add_new_model_like_parser.set_defaults(func=add_new_model_like_command_factory)

    def __init__(self, path_to_repo=None, *args):
        (
            self.old_model_infos,
            self.new_lowercase_name,
            self.new_model_paper_name,
            self.filenames_to_add,
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
            old_model_infos=self.old_model_infos,
            new_lowercase_name=self.new_lowercase_name,
            new_model_paper_name=self.new_model_paper_name,
            filenames_to_add=self.filenames_to_add,
            create_fast_image_processor=self.create_fast_image_processor,
        )
