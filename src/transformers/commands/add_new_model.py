# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

from ..utils import logging
from . import BaseTransformersCLICommand


try:
    from cookiecutter.main import cookiecutter

    _has_cookiecutter = True
except ImportError:
    _has_cookiecutter = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def add_new_model_command_factory(args: Namespace):
    return AddNewModelCommand(args.testing, args.testing_file, path=args.path)


class AddNewModelCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_new_model_parser = parser.add_parser("add-new-model")
        add_new_model_parser.add_argument("--testing", action="store_true", help="If in testing mode.")
        add_new_model_parser.add_argument("--testing_file", type=str, help="Configuration file on which to run.")
        add_new_model_parser.add_argument(
            "--path", type=str, help="Path to cookiecutter. Should only be used for testing purposes."
        )
        add_new_model_parser.set_defaults(func=add_new_model_command_factory)

    def __init__(self, testing: bool, testing_file: str, path=None, *args):
        self._testing = testing
        self._testing_file = testing_file
        self._path = path

    def run(self):
        if not _has_cookiecutter:
            raise ImportError(
                "Model creation dependencies are required to use the `add_new_model` command. Install them by running "
                "the following at the root of your `transformers` clone:\n\n\t$ pip install -e .[modelcreation]\n"
            )
        # Ensure that there is no other `cookiecutter-template-xxx` directory in the current working directory
        directories = [directory for directory in os.listdir() if "cookiecutter-template-" == directory[:22]]
        if len(directories) > 0:
            raise ValueError(
                "Several directories starting with `cookiecutter-template-` in current working directory. "
                "Please clean your directory by removing all folders starting with `cookiecutter-template-` or "
                "change your working directory."
            )

        path_to_transformer_root = (
            Path(__file__).parent.parent.parent.parent if self._path is None else Path(self._path).parent.parent
        )
        path_to_cookiecutter = path_to_transformer_root / "templates" / "adding_a_new_model"

        # Execute cookiecutter
        if not self._testing:
            cookiecutter(str(path_to_cookiecutter))
        else:
            with open(self._testing_file, "r") as configuration_file:
                testing_configuration = json.load(configuration_file)

            cookiecutter(
                str(path_to_cookiecutter if self._path is None else self._path),
                no_input=True,
                extra_context=testing_configuration,
            )

        directory = [directory for directory in os.listdir() if "cookiecutter-template-" in directory[:22]][0]

        # Retrieve configuration
        with open(directory + "/configuration.json", "r") as configuration_file:
            configuration = json.load(configuration_file)

        lowercase_model_name = configuration["lowercase_modelname"]
        pytorch_or_tensorflow = configuration["generate_tensorflow_and_pytorch"]
        os.remove(f"{directory}/configuration.json")

        output_pytorch = "PyTorch" in pytorch_or_tensorflow
        output_tensorflow = "TensorFlow" in pytorch_or_tensorflow

        model_dir = f"{path_to_transformer_root}/src/transformers/models/{lowercase_model_name}"
        os.makedirs(model_dir, exist_ok=True)

        shutil.move(
            f"{directory}/__init__.py",
            f"{model_dir}/__init__.py",
        )
        shutil.move(
            f"{directory}/configuration_{lowercase_model_name}.py",
            f"{model_dir}/configuration_{lowercase_model_name}.py",
        )

        def remove_copy_lines(path):
            with open(path, "r") as f:
                lines = f.readlines()
            with open(path, "w") as f:
                for line in lines:
                    if "# Copied from transformers." not in line:
                        f.write(line)

        if output_pytorch:
            if not self._testing:
                remove_copy_lines(f"{directory}/modeling_{lowercase_model_name}.py")

            shutil.move(
                f"{directory}/modeling_{lowercase_model_name}.py",
                f"{model_dir}/modeling_{lowercase_model_name}.py",
            )

            shutil.move(
                f"{directory}/test_modeling_{lowercase_model_name}.py",
                f"{path_to_transformer_root}/tests/test_modeling_{lowercase_model_name}.py",
            )
        else:
            os.remove(f"{directory}/modeling_{lowercase_model_name}.py")
            os.remove(f"{directory}/test_modeling_{lowercase_model_name}.py")

        if output_tensorflow:
            if not self._testing:
                remove_copy_lines(f"{directory}/modeling_tf_{lowercase_model_name}.py")

            shutil.move(
                f"{directory}/modeling_tf_{lowercase_model_name}.py",
                f"{model_dir}/modeling_tf_{lowercase_model_name}.py",
            )

            shutil.move(
                f"{directory}/test_modeling_tf_{lowercase_model_name}.py",
                f"{path_to_transformer_root}/tests/test_modeling_tf_{lowercase_model_name}.py",
            )
        else:
            os.remove(f"{directory}/modeling_tf_{lowercase_model_name}.py")
            os.remove(f"{directory}/test_modeling_tf_{lowercase_model_name}.py")

        shutil.move(
            f"{directory}/{lowercase_model_name}.rst",
            f"{path_to_transformer_root}/docs/source/model_doc/{lowercase_model_name}.rst",
        )

        shutil.move(
            f"{directory}/tokenization_{lowercase_model_name}.py",
            f"{model_dir}/tokenization_{lowercase_model_name}.py",
        )

        shutil.move(
            f"{directory}/tokenization_fast_{lowercase_model_name}.py",
            f"{model_dir}/tokenization_{lowercase_model_name}_fast.py",
        )

        from os import fdopen, remove
        from shutil import copymode, move
        from tempfile import mkstemp

        def replace(original_file: str, line_to_copy_below: str, lines_to_copy: List[str]):
            # Create temp file
            fh, abs_path = mkstemp()
            line_found = False
            with fdopen(fh, "w") as new_file:
                with open(original_file) as old_file:
                    for line in old_file:
                        new_file.write(line)
                        if line_to_copy_below in line:
                            line_found = True
                            for line_to_copy in lines_to_copy:
                                new_file.write(line_to_copy)

            if not line_found:
                raise ValueError(f"Line {line_to_copy_below} was not found in file.")

            # Copy the file permissions from the old file to the new file
            copymode(original_file, abs_path)
            # Remove original file
            remove(original_file)
            # Move new file
            move(abs_path, original_file)

        def skip_units(line):
            return ("generating PyTorch" in line and not output_pytorch) or (
                "generating TensorFlow" in line and not output_tensorflow
            )

        def replace_in_files(path_to_datafile):
            with open(path_to_datafile) as datafile:
                lines_to_copy = []
                skip_file = False
                skip_snippet = False
                for line in datafile:
                    if "# To replace in: " in line and "##" not in line:
                        file_to_replace_in = line.split('"')[1]
                        skip_file = skip_units(line)
                    elif "# Below: " in line and "##" not in line:
                        line_to_copy_below = line.split('"')[1]
                        skip_snippet = skip_units(line)
                    elif "# End." in line and "##" not in line:
                        if not skip_file and not skip_snippet:
                            replace(file_to_replace_in, line_to_copy_below, lines_to_copy)

                        lines_to_copy = []
                    elif "# Replace with" in line and "##" not in line:
                        lines_to_copy = []
                    elif "##" not in line:
                        lines_to_copy.append(line)

            remove(path_to_datafile)

        replace_in_files(f"{directory}/to_replace_{lowercase_model_name}.py")
        os.rmdir(directory)
