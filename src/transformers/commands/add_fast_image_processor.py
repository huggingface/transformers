# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import os
import re
from argparse import ArgumentParser, Namespace
from datetime import date
from pathlib import Path

from ..utils import logging
from . import BaseTransformersCLICommand


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


CURRENT_YEAR = date.today().year
TRANSFORMERS_PATH = Path(__file__).parent.parent
REPO_PATH = TRANSFORMERS_PATH.parent.parent


def add_import_structure_entry_init(content: str, fast_image_processor_name: str, model_name: str):
    # Step 1: Find the block
    block_regex = re.compile(
        r"if not is_torchvision_available\(\):.*?else:\s*(\n(?P<indent>\s+)_import_structure\[.*?\].*?\n(?:\s*(?P=indent)_import_structure\[.*?\].*?\n)*)",
        re.DOTALL,
    )
    match = block_regex.search(content)

    if not match:
        raise ValueError("Couldn't find the '_import_structure' block.")

    # Capture the block content and indentation
    block_content = match.group(1)
    indent = match.group("indent")

    # Step 2: Parse existing entries
    lines = block_content.strip().split("\n")
    entries = []

    import_structure_header = indent + lines[0]
    entries = lines[1:]

    # Add the new entry, maintaining alphabetical order
    new_entry = f'{indent}_import_structure["models.{model_name}"].append("{fast_image_processor_name}")'
    if new_entry not in entries:
        entries.append(new_entry)

    entries.sort()
    entries = [import_structure_header] + entries

    # Step 3: Reconstruct the block
    updated_block = "\n".join(entry for entry in entries)

    # Replace the original block in the content
    updated_content = content[: match.start(1)] + "\n" + updated_block + "\n" + content[match.end(1) :]

    return updated_content


def add_import_statement_init(content: str, fast_image_processor_name: str, model_name: str):
    # Step 1: Find the block
    block_regex = re.compile(
        r"if not is_torchvision_available\(\):\s+raise OptionalDependencyNotAvailable\(\)\s+except OptionalDependencyNotAvailable:\s+from \.utils\.dummy_torchvision_objects import \*\s+else:(?P<else_block>\s*(\n\s*from .+ import .*\n)+)(?=\s*# Modeling)",
        re.DOTALL,
    )
    match = block_regex.search(content)

    if match:
        block_content = match.group("else_block")  # The captured import block
    else:
        print("Couldn't find the import statement block.")

    # Step 2: Parse existing entries
    lines = block_content.strip().split("\n")
    entries = []

    indent = " " * (len(lines[1]) - len(lines[1].lstrip()))
    import_structure_header = indent + lines[0]
    entries = lines[1:]

    # Add the new entry, maintaining alphabetical order
    new_entry = f"{indent}from .models.{model_name} import {fast_image_processor_name}"
    if new_entry not in entries:
        entries.append(new_entry)

    entries.sort()
    entries = [import_structure_header] + entries

    # Step 3: Reconstruct the block
    updated_block = "\n".join(entry for entry in entries)

    # Replace the original block in the content
    updated_content = (
        content[: match.start("else_block")] + "\n" + updated_block + "\n\n" + content[match.end("else_block") :]
    )

    return updated_content


def add_fast_image_processor_to_main_init(fast_image_processor_name: str, model_name: str):
    with open(TRANSFORMERS_PATH / "__init__.py", "r", encoding="utf-8") as f:
        content = f.read()

    # add _import_structure entry
    content = add_import_structure_entry_init(content, fast_image_processor_name, model_name)
    # add import statement
    content = add_import_statement_init(content, fast_image_processor_name, model_name)

    # write the updated content
    with open(TRANSFORMERS_PATH / "__init__.py", "w", encoding="utf-8") as f:
        f.write(content)


def add_fast_image_processor_to_model_init(
    fast_image_processing_module_file: str, fast_image_processor_name, model_name: str
):
    with open(TRANSFORMERS_PATH / "models" / model_name / "__init__.py", "r", encoding="utf-8") as f:
        content = f.read()

    fast_image_processing_module_file = fast_image_processing_module_file.split(os.sep)[-1].replace(".py", "")

    if "import *" in content:
        # we have an init file in the updated format
        # get the indented block after if TYPE_CHECKING: and before else:, append the new import, sort the imports and write the updated content
        # Step 1: Find the block
        block_regex = re.compile(
            r"if TYPE_CHECKING:\n(?P<if_block>.*?)(?=\s*else:)",
            re.DOTALL,
        )
        match = block_regex.search(content)

        if not match:
            raise ValueError("Couldn't find the 'if TYPE_CHECKING' block.")

        block_content = match.group("if_block")  # The captured import block

        # Step 2: Parse existing entries
        entries = block_content.split("\n")
        indent = " " * (len(entries[0]) - len(entries[0].lstrip()))
        new_entry = f"{indent}from .{fast_image_processing_module_file} import *"
        if new_entry not in entries:
            entries.append(new_entry)
        entries.sort()
        updated_block = "\n".join(entry for entry in entries)

        # Replace the original block in the content
        updated_content = content[: match.start("if_block")] + updated_block + content[match.end("if_block") :]
    else:
        # we have an init file in the old format

        # add "is_torchvision_available" import to from ...utils import (
        # get import block
        import_block_regex = re.compile(r"from \.\.\.utils import \(\n(?P<import_block>.*?)(?=\n\))", re.DOTALL)
        match = import_block_regex.search(content)
        if not match:
            raise ValueError("Couldn't find the 'from ...utils import' block.")

        import_block = match.group("import_block")
        entries = import_block.split("\n")
        indent = " " * (len(entries[0]) - len(entries[0].lstrip()))
        new_entry = f"{indent}is_torchvision_available,"
        if new_entry not in entries:
            entries.append(new_entry)
        entries.sort()
        updated_block = "\n".join(entry for entry in entries)
        # Replace the original block in the content
        updated_content = content[: match.start("import_block")] + updated_block + content[match.end("import_block") :]

        vision_import_structure_block = f'    _import_structure["{fast_image_processing_module_file[:-5]}"] = ["{fast_image_processor_name[:-4]}"]\n'

        added_import_structure_block = (
            "try:\n    if not is_torchvision_available():\n"
            "        raise OptionalDependencyNotAvailable()\n"
            "except OptionalDependencyNotAvailable:\n"
            "    pass\n"
            "else:\n"
            f'    _import_structure["{fast_image_processing_module_file}"] = ["{fast_image_processor_name}"]\n'
        )

        if vision_import_structure_block not in updated_content:
            raise ValueError("Couldn't find the 'vision _import_structure block' block.")

        if added_import_structure_block not in updated_content:
            updated_content = updated_content.replace(
                vision_import_structure_block, vision_import_structure_block + "\n" + added_import_structure_block
            )

        vision_import_statement_block = (
            f"        from .{fast_image_processing_module_file[:-5]} import {fast_image_processor_name[:-4]}\n"
        )

        added_import_statement_block = (
            "    try:\n        if not is_torchvision_available():\n"
            "            raise OptionalDependencyNotAvailable()\n"
            "    except OptionalDependencyNotAvailable:\n"
            "        pass\n"
            "    else:\n"
            f"        from .{fast_image_processing_module_file} import {fast_image_processor_name}\n"
        )

        if vision_import_statement_block not in updated_content:
            raise ValueError("Couldn't find the 'vision _import_structure block' block.")

        if added_import_statement_block not in updated_content:
            updated_content = updated_content.replace(
                vision_import_statement_block, vision_import_statement_block + "\n" + added_import_statement_block
            )

    # write the updated content
    with open(TRANSFORMERS_PATH / "models" / model_name / "__init__.py", "w", encoding="utf-8") as f:
        f.write(updated_content)


def add_fast_image_processor_to_auto(image_processor_name: str, fast_image_processor_name: str):
    with open(TRANSFORMERS_PATH / "models" / "auto" / "image_processing_auto.py", "r", encoding="utf-8") as f:
        content = f.read()

    # get all lines containing the image processor name
    updated_content = content.replace(
        f'("{image_processor_name}",)', f'("{image_processor_name}", "{fast_image_processor_name}")'
    )

    # write the updated content
    with open(TRANSFORMERS_PATH / "models" / "auto" / "image_processing_auto.py", "w", encoding="utf-8") as f:
        f.write(updated_content)


def add_fast_image_processor_to_dummy(fast_image_processor_name: str):
    dummy_torchvision_objects_file = TRANSFORMERS_PATH / "utils" / "dummy_torchvision_objects.py"
    with open(dummy_torchvision_objects_file, "r", encoding="utf-8") as f:
        content = f.read()

    # regex to find objects starting with "class " and ending with "ImageProcessorFast", including "ImageProcessorFast" in the match
    image_processor_names = re.findall(r"class (\w*ImageProcessorFast)", content)
    image_processor_names.append(fast_image_processor_name)
    image_processor_names.sort()
    index_new = image_processor_names.index(fast_image_processor_name)

    new_dummy_object = (
        f"class {fast_image_processor_name}(metaclass=DummyObject):\n"
        '    _backends = ["torchvision"]\n\n'
        "    def __init__(self, *args, **kwargs):\n"
        '        requires_backends(self, ["torchvision"])\n'
    )
    if new_dummy_object not in content:
        if index_new != len(image_processor_names) - 1:
            # add the dummy object just before the next ImageProcessorFast
            first_line = f"class {image_processor_names[index_new+1]}(metaclass=DummyObject):"
            updated_content = content.replace(first_line, new_dummy_object + "\n\n" + first_line)
        else:
            # add the dummy object at the very end
            updated_content = content + "\n\n" + new_dummy_object

        # write the updated content
        with open(dummy_torchvision_objects_file, "w", encoding="utf-8") as f:
            f.write(updated_content)


def add_fast_image_processor_file(
    fast_image_processing_module_file: str, fast_image_processor_name: str, content_base_file: str
):
    # if the file already exists, do nothing
    if os.path.exists(fast_image_processing_module_file):
        print(f"{fast_image_processing_module_file} already exists. Skipping.")
        return

    imports = "\n\nfrom ...image_processing_utils_fast import BaseImageProcessorFast\n\n\n"
    content_header = get_fast_image_processing_content_header(content_base_file)
    content_base_file = (
        f"class {fast_image_processor_name}(BaseImageProcessorFast):\n"
        "    # To be implemented\n"
        "    resample = None\n"
        "    image_mean = None\n"
        "    image_std = None\n"
        '    size = {"height": None, "width": None}\n'
        '    crop_size = {"height": None, "width": None}\n'
        "    do_resize = None\n"
        "    do_center_crop = None\n"
        "    do_rescale = None\n"
        "    do_normalize = None\n"
        "    do_convert_rgb = None\n"
    )

    content = content_header + imports + content_base_file

    with open(fast_image_processing_module_file, "w", encoding="utf-8") as f:
        f.write(content)


def get_fast_image_processing_content_header(content: str) -> str:
    # get all lines before and including the line containing """Image processor
    content_header = re.search(r"^(.*?\n)*?\"\"\"Image processor.*", content)
    content_header = content_header.group(0)
    content_header = re.sub(r"# Copyright (\d+)\s", f"# Copyright {CURRENT_YEAR} ", content_header)
    content_header = content_header.replace("Image processor", "Fast Image processor")
    return content_header


def add_fast_image_processor(model_name: str):
    model_module = TRANSFORMERS_PATH / "models" / model_name
    image_processing_module_file = list(model_module.glob("image_processing*.py"))
    if not image_processing_module_file:
        raise ValueError(f"No image processing module found in {model_module}")
    elif len(image_processing_module_file) > 1:
        raise ValueError(f"Multiple image processing modules found in {model_module}")

    image_processing_module_file = str(image_processing_module_file[0])
    with open(image_processing_module_file, "r", encoding="utf-8") as f:
        content_base_file = f.read()

    # regex to find object starting with "class " and ending with "ImageProcessor", including "ImageProcessor" in the match
    image_processor_name = re.findall(r"class (\w*ImageProcessor)", content_base_file)
    if not image_processor_name:
        raise ValueError(f"No ImageProcessor class found in {image_processing_module_file}")
    elif len(image_processor_name) > 1:
        raise ValueError(f"Multiple ImageProcessor classes found in {image_processing_module_file}")

    image_processor_name = image_processor_name[0]
    fast_image_processor_name = image_processor_name + "Fast"
    fast_image_processing_module_file = image_processing_module_file.replace(".py", "_fast.py")

    print(f"Adding {fast_image_processor_name} to {fast_image_processing_module_file}")

    add_fast_image_processor_to_main_init(
        fast_image_processor_name=fast_image_processor_name,
        model_name=model_name,
    )

    add_fast_image_processor_to_model_init(
        fast_image_processing_module_file=fast_image_processing_module_file,
        fast_image_processor_name=fast_image_processor_name,
        model_name=model_name,
    )

    add_fast_image_processor_to_auto(
        image_processor_name=image_processor_name,
        fast_image_processor_name=fast_image_processor_name,
    )

    add_fast_image_processor_to_dummy(fast_image_processor_name=fast_image_processor_name)

    add_fast_image_processor_file(
        fast_image_processing_module_file=fast_image_processing_module_file,
        fast_image_processor_name=fast_image_processor_name,
        content_base_file=content_base_file,
    )


def add_new_model_like_command_factory(args: Namespace):
    return AddFastImageProcessorCommand(model_name=args.model_name)


class AddFastImageProcessorCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_fast_image_processor_parser = parser.add_parser("add-fast-image-processor")
        add_fast_image_processor_parser.add_argument(
            "--model-name",
            type=str,
            required=True,
            help="The name of the folder containing the model's implementation.",
        )
        add_fast_image_processor_parser.set_defaults(func=add_new_model_like_command_factory)

    def __init__(self, model_name: str, *args):
        self.model_name = model_name

    def run(self):
        add_fast_image_processor(model_name=self.model_name)
