# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_config_docstrings.py
PATH_TO_TRANSFORMERS = "src/transformers"


# This is to make sure the transformers module imported is the one in the repo.
spec = importlib.util.spec_from_file_location(
    "transformers",
    os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"),
    submodule_search_locations=[PATH_TO_TRANSFORMERS],
)
transformers = spec.loader.load_module()

CONFIG_MAPPING = transformers.models.auto.configuration_auto.CONFIG_MAPPING

# Regex pattern used to find the checkpoint mentioned in the docstring of `config_class`.
# For example, `[bert-base-uncased](https://huggingface.co/bert-base-uncased)`
_re_checkpoint = re.compile("\[(.+?)\]\((https://huggingface\.co/.+?)\)")


CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK = {
    "CLIPConfig",
    "DecisionTransformerConfig",
    "EncoderDecoderConfig",
    "RagConfig",
    "SpeechEncoderDecoderConfig",
    "VisionEncoderDecoderConfig",
    "VisionTextDualEncoderConfig",
}


def check_config_docstrings_have_checkpoints():
    configs_without_checkpoint = []

    for config_class in list(CONFIG_MAPPING.values()):
        checkpoint_found = False

        # source code of `config_class`
        config_source = inspect.getsource(config_class)
        checkpoints = _re_checkpoint.findall(config_source)

        for checkpoint in checkpoints:
            # Each `checkpoint` is a tuple of a checkpoint name and a checkpoint link.
            # For example, `('bert-base-uncased', 'https://huggingface.co/bert-base-uncased')`
            ckpt_name, ckpt_link = checkpoint

            # verify the checkpoint name corresponds to the checkpoint link
            ckpt_link_from_name = f"https://huggingface.co/{ckpt_name}"
            if ckpt_link == ckpt_link_from_name:
                checkpoint_found = True
                break

        name = config_class.__name__
        if not checkpoint_found and name not in CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK:
            configs_without_checkpoint.append(name)

    if len(configs_without_checkpoint) > 0:
        message = "\n".join(sorted(configs_without_checkpoint))
        raise ValueError(f"The following configurations don't contain any valid checkpoint:\n{message}")


if __name__ == "__main__":
    check_config_docstrings_have_checkpoints()
