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

import argparse
import os

from transformers.utils import direct_transformers_import


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_task_guides.py
TRANSFORMERS_PATH = "src/transformers"
PATH_TO_TASK_GUIDES = "docs/source/en/tasks"


def _find_text_in_file(filename, start_prompt, end_prompt):
    """
    Find the text in `filename` between a line beginning with `start_prompt` and before `end_prompt`, removing empty
    lines.
    """
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    # Find the start prompt.
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1

    end_index = start_index
    while not lines[end_index].startswith(end_prompt):
        end_index += 1
    end_index -= 1

    while len(lines[start_index]) <= 1:
        start_index += 1
    while len(lines[end_index]) <= 1:
        end_index -= 1
    end_index += 1
    return "".join(lines[start_index:end_index]), start_index, end_index, lines


# This is to make sure the transformers module imported is the one in the repo.
transformers_module = direct_transformers_import(TRANSFORMERS_PATH)

TASK_GUIDE_TO_MODELS = {
    "asr.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_CTC_MAPPING_NAMES,
    "audio_classification.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    "language_modeling.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    "image_classification.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    "masked_language_modeling.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    "multiple_choice.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    "object_detection.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
    "question_answering.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    "semantic_segmentation.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    "sequence_classification.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    "summarization.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    "token_classification.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    "translation.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    "video_classification.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
    "document_question_answering.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    "monocular_depth_estimation.md": transformers_module.models.auto.modeling_auto.MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES,
}

# This list contains model types used in some task guides that are not in `CONFIG_MAPPING_NAMES` (therefore not in any
# `MODEL_MAPPING_NAMES` or any `MODEL_FOR_XXX_MAPPING_NAMES`).
SPECIAL_TASK_GUIDE_TO_MODEL_TYPES = {
    "summarization.md": ("nllb",),
    "translation.md": ("nllb",),
}


def get_model_list_for_task(task_guide):
    """
    Return the list of models supporting given task.
    """
    model_maping_names = TASK_GUIDE_TO_MODELS[task_guide]
    special_model_types = SPECIAL_TASK_GUIDE_TO_MODEL_TYPES.get(task_guide, set())
    model_names = {
        code: name
        for code, name in transformers_module.MODEL_NAMES_MAPPING.items()
        if (code in model_maping_names or code in special_model_types)
    }
    return ", ".join([f"[{name}](../model_doc/{code})" for code, name in model_names.items()]) + "\n"


def check_model_list_for_task(task_guide, overwrite=False):
    """For a given task guide, checks the model list in the generated tip for consistency with the state of the lib and overwrites if needed."""

    current_list, start_index, end_index, lines = _find_text_in_file(
        filename=os.path.join(PATH_TO_TASK_GUIDES, task_guide),
        start_prompt="<!--This tip is automatically generated by `make fix-copies`, do not fill manually!-->",
        end_prompt="<!--End of the generated tip-->",
    )

    new_list = get_model_list_for_task(task_guide)

    if current_list != new_list:
        if overwrite:
            with open(os.path.join(PATH_TO_TASK_GUIDES, task_guide), "w", encoding="utf-8", newline="\n") as f:
                f.writelines(lines[:start_index] + [new_list] + lines[end_index:])
        else:
            raise ValueError(
                f"The list of models that can be used in the {task_guide} guide needs an update. Run `make fix-copies`"
                " to fix this."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    for task_guide in TASK_GUIDE_TO_MODELS.keys():
        check_model_list_for_task(task_guide, args.fix_and_overwrite)
