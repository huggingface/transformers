"""
Script which deprecates a list of given models
"""

import argparse
import os
from collections import defaultdict
from packaging import version

from transformers import CONFIG_MAPPING
from transformers import __version__ as current_version

import json
import requests
import glob

from git import Repo

REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

repo = Repo(REPO_PATH)

print(REPO_PATH)


# # FIXME - more robust way of finding the model doc - find the model class form the model's init file and then find the model doc


def get_last_stable_minor_release():
    # Get the last stable release of transformers
    url = "https://pypi.org/pypi/transformers/json"
    release_data = requests.get(url).json()

    # Find the last stable release of of transformers (version below current version)
    major_version, minor_version, patch_version, _ = current_version.split(".")
    last_major_minor = f"{major_version}.{int(minor_version) - 1}"
    last_stable_minor_releases = [release for release in release_data["releases"] if release.startswith(last_major_minor)]
    last_stable_release = sorted(last_stable_minor_releases, key=version.parse)[-1]

    return last_stable_release


def build_tip_message(last_stable_release):
    return """
    <Tip warning={true}>

    This model is in maintenance mode only, we don't accept any new PRs changing its code.
    """ + f"""If you run into any issues running this model, please reinstall the last version that supported this model: v{last_stable_release}.
    You can do so by running the following command: `pip install -U transformers=={last_stable_release}`.

    </Tip>"""


def insert_tip_to_model_doc(model_doc_path, tip_message):
    tip_message_lines = tip_message.split("\n")

    with open(model_doc_path, "r") as f:
        model_doc = f.read()

    # Add the tip message to the model doc page directly underneath the title
    lines = model_doc.split("\n")

    new_model_lines = []
    for line in lines:
        if line.startswith("# "):
            new_model_lines.append(line)
            new_model_lines.extend(tip_message_lines)
        else:
            new_model_lines.append(line)

    with open(model_doc_path, "w") as f:
        f.write("\n".join(new_model_lines))



models = [
    # 'graphormer',
    # 'time_series_transformer',
    'conditional_detr',
    # 'xlm_prophetnet',
    # 'qdqbert',
    # 'nat',
    # 'data2vec',
    # 'ernie_m',
    # 'dinat',
    # 'tvlt',
    # 'nezha',
    # 'jukebox',
    'vit_hybrid',
    # 'decision_transformer',
    # 'x_clip',
    # 'deta',
    # 'speech_to_text_2',
    # 'efficientformer',
    # 'realm',
    # 'openai',
    # 'gptsan_japanese'
]


def get_model_doc_path(model):
    model_doc_path = f"docs/source/en/model_doc/{model}.md"

    if os.path.exists(model_doc_path):
        return model_doc_path

    # Try replacing _ with - in the model name
    model_doc_path = f"docs/source/en/model_doc/{model.replace('_', '-')}.md"

    if os.path.exists(model_doc_path):
        return model_doc_path

    # Try replacing _ with "" in the model name
    model_doc_path = f"docs/source/en/model_doc/{model.replace('_', '')}.md"

    if os.path.exists(model_doc_path):
        return model_doc_path

    return None


skipped_models = []
model_info = defaultdict(dict)
# For each model, find its model doc page e.g. docs/source/en/model_doc/conditional_detr.md for conditional_detr
for model in models:
    model_doc_path = get_model_doc_path(model)
    model_path = f"src/transformers/models/{model}"

    if model_doc_path is None:
        print(f"Model doc path does not exist for {model}")
        skipped_models.append(model)
        continue
    else:
        model_info[model]["model_doc_path"] = model_doc_path

    if not os.path.exists(model_path):
        print(f"Model path does not exist for {model}")
        skipped_models.append(model)
        continue
    else:
        model_info[model]["model_path"] = model_path


models = [model for model in models if model not in skipped_models]

# Comment out for the moment just to avoid having to revert when testing
# last_stable_release = get_last_stable_minor_release()

# tip_message = build_tip_message(last_stable_release)
# # Add the tip message to the model doc page directly underneath the title
# for model, info in model_info.items():
#     if not "model_doc_path" in info:
#         continue

#     insert_tip_to_model_doc(info["model_doc_path"], tip_message)



def move_model_files_to_deprecated(model):
    # FIXME - more robust path handling
    model_path = f"src/transformers/models/{model}"
    deprecated_model_path = f"src/transformers/models/deprecated/{model}"

    if not os.path.exists(deprecated_model_path):
        os.makedirs(deprecated_model_path)

    for file in os.listdir(model_path):
        if file == "__pycache__":
            continue
        repo.git.mv(f"{model_path}/{file}", f"{deprecated_model_path}/{file}")


# # For each model, we move all the files in the model directory to a deprecated directory
# # e.g. src/transformers/models/conditional_detr/* -> src/transformers/models/deprecated/conditional_detr/*
# for model, info in model_info.items():
#     if not "model_path" in info:
#         continue
#    move_model_files_to_deprecated(model)


def delete_model_tests(model):
    tests_path = f"tests/models/{model}"

    if os.path.exists(tests_path):
        repo.git.rm("-r", tests_path)


# for model, info in model_info.items():
#     if not "model_path" in info:
#         continue
#     delete_model_tests(model)



def update_alphabetic_ordering_of_imports():
    # For the direct import, they will be sorted by make fixup.
    # We need to sort the _import_structure lines

    def get_line_indent(s):
        return len(s) - len(s.lstrip())

    new_init_file_lines = []
    else_block = []
    # Sub-blocks are lines that are indented at the same level and are part. Blocks are separated by a blank line or comment
    # This is to ensure that we maintain the same structure as the original file

    # a = 1 - sub_block 1
    # b = 2 - sub_block 1
    # # Comment
    # c = 3 - sub_block 2

    # d = 4 - sub_block 3
    sub_block = []

    # Indented block is a block of lines have an indented section at the same level
    # a = 1          - sub_block 1
    # b = [          - sub_block 1, indented_block 1
    #     1, 2, 3    - sub_block 1, indented_block 1
    # ]              - sub_block 1, indented_block 1
    # c = 3          - sub_block 1
    indented_block = []

    # Maybe block is a block of lines that might be part of the else block, but we're not sure yet
    #     a = 1  - sub_block 1
    #            - maybe_else_block 1 (part of else block)
    #            - maybe_else_block 1 (part of else block)
    #     b = 2  - sub_block 2
    #            - maybe_else_block 2 (not part of else block)
    # c = 3
    maybe_else_block = []
    in_else_block = False

    # We iterate over each line in the init file to create a new init file
    for i, line in enumerate(init_file.split("\n")):
        # Next line is in the else block
        if line.startswith("else:"):
            new_init_file_lines.append(line)
            in_else_block = True

        # Not in the else block - just add the line directly
        elif not in_else_block:
            new_init_file_lines.append(line)

        elif in_else_block:
            indent = get_line_indent(line)

            # previous line(s) were a blank line but within the else block
            if indent and maybe_else_block:
                else_block.append(maybe_else_block)
                maybe_else_block = []
                # We then want to add the line according to the rules below

            # We might be outside of the else block, or it might just be a blank line
            if not indent and line == "":
                # End any existing sub_block and add it to the else block
                else_block.append(sub_block)
                sub_block = []

                maybe_else_block.append(line)

            elif not indent and line != "":
                # If we were in a maybe block, we now know it wasn't part of the else block
                # Sort the sub-blocks in the else block
                else_block = [[sorted(sub_block) if isinstance(sub_block, list) else sub_block for sub_block in sub_blocks] for sub_blocks in else_block]

                # Flatten the lists so they're all lines
                else_block = [line for sub_block in else_block for line in sub_block]

                # Add the else block to the file lines and reset it
                new_init_file_lines.extend(else_block)
                else_block = []
                in_else_block = False

                # Add the maybe block
                new_init_file_lines.extend(maybe_else_block)
                maybe_else_block = []

            # All import structures are at the same level so we can just add them directly
            elif line.strip().startswith("_import_structure") and line.endswith(("]", ")")):
                sub_block.append(line)

            elif line.strip().startswith("_import_structure") and line.endswith(("[", "(")):
                if sub_block:
                    else_block.append(sub_block)
                    sub_block = []

                indented_block.append(line)

            elif indented_block:
                if line.strip().endswith(("]", ")")) and indent == 4:
                    # We're at the end of the indented block
                    indented_block.append(line)
                    sub_block.append("\n".join(indented_block))
                    indented_block = []
                else:
                    # We're still in the indented block
                    indented_block.append(line)

            # We have a comment line - create a new sub-block so lines above the comment are grouped together when sorting
            elif line.strip().startswith("#"):
                if sub_block:
                    else_block.append(sub_block)
                    sub_block = []

                sub_block.append(line)

            else:
                sub_block.append(line)

    # Add the last sub-block
    if else_block:
        # Sort the sub-blocks in the else block
        else_block = [[sorted(sub_block) if isinstance(sub_block, list) else sub_block for sub_block in sub_blocks] for sub_blocks in else_block]

        # Flatten the lists so they're all lines
        else_block = [line for sub_block in else_block for line in sub_block]

    return "\n".join(new_init_file_lines)


def update_init_file(filename, models):
    with open(filename, "r") as f:
        init_file = f.read()

    # 1. For each model, find all the instances of model.model_name and replace with model.deprecated.model_name
    for model in models:
        init_file = init_file.replace(f"models.{model}", f"models.deprecated.{model}")

    # 2. Resort the imports
    init_file = update_alphabetic_ordering_of_imports(init_file)

    with open(filename, "w") as f:
        f.write(init_file)


# update_init_file("src/transformers/models/__init__.py", models)

def remove_model_references_from_file(filename, models, condition = None):
    """
    Remove all references to the given models from the given file

    Args:
        filename (str): The file to remove the references from
        models (List[str]): The models to remove
        condition (Callable): A function that takes the line and model and returns True if the line should be removed
    """
    if condition is None:
        condition = lambda line, model: model == line.strip()

    with open(filename, "r") as f:
        init_file = f.read()

    new_file_lines = []
    for i, line in enumerate(init_file.split("\n")):
        # stripped_line = line.strip().strip(",")
        if any(condition(line, model) for model in models):
            continue
        new_file_lines.append(line)

    with open(filename, "w") as f:
        f.write("\n".join(new_file_lines))

# remove_model_references_from_file("src/transformers/models/__init__.py", models, lambda line, model: model == line.strip().strip(","))
# remove_model_references_from_file("utils/slow_documentation_tests.txt", models, lambda line, model: "/" + model + "/" in line)

# Get the models config class names



model_config_classes = [CONFIG_MAPPING[model_name].__name__ for model_name in models + ["oneformer", "seamless_m4t", "whisper"]]


def remove_model_config_classes_from_config_check(filename, model_config_classes):
    with open(filename, "r") as f:
        check_config_attributes = f.read()

    # Keep track as we have to delete comment above too
    in_special_cases_to_allow = False
    in_indent = False
    new_file_lines = []

    for line in check_config_attributes.split("\n"):
        if line.strip() == "SPECIAL_CASES_TO_ALLOW = {":
            in_special_cases_to_allow = True

        elif in_special_cases_to_allow and line.strip() == "}":
            in_special_cases_to_allow = False

        if in_indent:
            if line.strip().endswith(("]", "],")):
                in_indent = False
            continue

        if in_special_cases_to_allow and any(model_config_class in line for model_config_class in model_config_classes):
            while new_file_lines[-1].strip().startswith("#"):
                new_file_lines.pop()

            if line.strip().endswith("["):
                in_indent = True

            continue

        elif any(model_config_class in line for model_config_class in model_config_classes):
            continue

        new_file_lines.append(line)

    with open(filename, "w") as f:
        f.write("\n".join(new_file_lines))


remove_model_config_classes_from_config_check("src/transformers/configuration_utils.py", model_config_classes)

def 






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="List of models to deprecate")
