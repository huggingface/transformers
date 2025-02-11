"""
Script which deprecates a list of given models

Example usage:
python utils/deprecate_models.py --models bert distilbert
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import requests
from custom_init_isort import sort_imports_in_all_inits
from git import Repo
from packaging import version

from transformers import CONFIG_MAPPING, logging
from transformers import __version__ as current_version


REPO_PATH = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
repo = Repo(REPO_PATH)

logger = logging.get_logger(__name__)


def get_last_stable_minor_release():
    # Get the last stable release of transformers
    url = "https://pypi.org/pypi/transformers/json"
    release_data = requests.get(url).json()

    # Find the last stable release of of transformers (version below current version)
    major_version, minor_version, patch_version, _ = current_version.split(".")
    last_major_minor = f"{major_version}.{int(minor_version) - 1}"
    last_stable_minor_releases = [
        release for release in release_data["releases"] if release.startswith(last_major_minor)
    ]
    last_stable_release = sorted(last_stable_minor_releases, key=version.parse)[-1]

    return last_stable_release


def build_tip_message(last_stable_release):
    return (
        """
<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
"""
        + f"""If you run into any issues running this model, please reinstall the last version that supported this model: v{last_stable_release}.
You can do so by running the following command: `pip install -U transformers=={last_stable_release}`.

</Tip>"""
    )


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


def get_model_doc_path(model: str) -> Tuple[Optional[str], Optional[str]]:
    # Possible variants of the model name in the model doc path
    model_names = [model, model.replace("_", "-"), model.replace("_", "")]

    model_doc_paths = [REPO_PATH / f"docs/source/en/model_doc/{model_name}.md" for model_name in model_names]

    for model_doc_path, model_name in zip(model_doc_paths, model_names):
        if os.path.exists(model_doc_path):
            return model_doc_path, model_name

    return None, None


def extract_model_info(model):
    model_info = {}
    model_doc_path, model_doc_name = get_model_doc_path(model)
    model_path = REPO_PATH / f"src/transformers/models/{model}"

    if model_doc_path is None:
        print(f"Model doc path does not exist for {model}")
        return None
    model_info["model_doc_path"] = model_doc_path
    model_info["model_doc_name"] = model_doc_name

    if not os.path.exists(model_path):
        print(f"Model path does not exist for {model}")
        return None
    model_info["model_path"] = model_path

    return model_info


def update_relative_imports(filename, model):
    with open(filename, "r") as f:
        filelines = f.read()

    new_file_lines = []
    for line in filelines.split("\n"):
        if line.startswith("from .."):
            new_file_lines.append(line.replace("from ..", "from ..."))
        else:
            new_file_lines.append(line)

    with open(filename, "w") as f:
        f.write("\n".join(new_file_lines))


def remove_copied_from_statements(model):
    model_path = REPO_PATH / f"src/transformers/models/{model}"
    for file in os.listdir(model_path):
        if file == "__pycache__":
            continue
        file_path = model_path / file
        with open(file_path, "r") as f:
            file_lines = f.read()

        new_file_lines = []
        for line in file_lines.split("\n"):
            if "# Copied from" in line:
                continue
            new_file_lines.append(line)

        with open(file_path, "w") as f:
            f.write("\n".join(new_file_lines))


def move_model_files_to_deprecated(model):
    model_path = REPO_PATH / f"src/transformers/models/{model}"
    deprecated_model_path = REPO_PATH / f"src/transformers/models/deprecated/{model}"

    if not os.path.exists(deprecated_model_path):
        os.makedirs(deprecated_model_path)

    for file in os.listdir(model_path):
        if file == "__pycache__":
            continue
        repo.git.mv(f"{model_path}/{file}", f"{deprecated_model_path}/{file}")

        # For deprecated files, we then need to update the relative imports
        update_relative_imports(f"{deprecated_model_path}/{file}", model)


def delete_model_tests(model):
    tests_path = REPO_PATH / f"tests/models/{model}"

    if os.path.exists(tests_path):
        repo.git.rm("-r", tests_path)


def get_line_indent(s):
    return len(s) - len(s.lstrip())


def update_main_init_file(models):
    """
    Replace all instances of model.model_name with model.deprecated.model_name in the __init__.py file

    Args:
        models (List[str]): The models to mark as deprecated
    """
    filename = REPO_PATH / "src/transformers/__init__.py"
    with open(filename, "r") as f:
        init_file = f.read()

    # 1. For each model, find all the instances of model.model_name and replace with model.deprecated.model_name
    for model in models:
        init_file = init_file.replace(f'models.{model}"', f'models.deprecated.{model}"')
        init_file = init_file.replace(f"models.{model} import", f"models.deprecated.{model} import")

    with open(filename, "w") as f:
        f.write(init_file)

    # 2. Resort the imports
    sort_imports_in_all_inits(check_only=False)


def remove_model_references_from_file(filename, models, condition):
    """
    Remove all references to the given models from the given file

    Args:
        filename (str): The file to remove the references from
        models (List[str]): The models to remove
        condition (Callable): A function that takes the line and model and returns True if the line should be removed
    """
    filename = REPO_PATH / filename
    with open(filename, "r") as f:
        init_file = f.read()

    new_file_lines = []
    for i, line in enumerate(init_file.split("\n")):
        if any(condition(line, model) for model in models):
            continue
        new_file_lines.append(line)

    with open(filename, "w") as f:
        f.write("\n".join(new_file_lines))


def remove_model_config_classes_from_config_check(model_config_classes):
    """
    Remove the deprecated model config classes from the check_config_attributes.py file

    Args:
        model_config_classes (List[str]): The model config classes to remove e.g. ["BertConfig", "DistilBertConfig"]
    """
    filename = REPO_PATH / "utils/check_config_attributes.py"
    with open(filename, "r") as f:
        check_config_attributes = f.read()

    # Keep track as we have to delete comment above too
    in_special_cases_to_allow = False
    in_indent = False
    new_file_lines = []

    for line in check_config_attributes.split("\n"):
        indent = get_line_indent(line)
        if (line.strip() == "SPECIAL_CASES_TO_ALLOW = {") or (line.strip() == "SPECIAL_CASES_TO_ALLOW.update("):
            in_special_cases_to_allow = True

        elif in_special_cases_to_allow and indent == 0 and line.strip() in ("}", ")"):
            in_special_cases_to_allow = False

        if in_indent:
            if line.strip().endswith(("]", "],")):
                in_indent = False
            continue

        if in_special_cases_to_allow and any(
            model_config_class in line for model_config_class in model_config_classes
        ):
            # Remove comments above the model config class to remove
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


def add_models_to_deprecated_models_in_config_auto(models):
    """
    Add the models to the DEPRECATED_MODELS list in configuration_auto.py and sorts the list
    to be in alphabetical order.
    """
    filepath = REPO_PATH / "src/transformers/models/auto/configuration_auto.py"
    with open(filepath, "r") as f:
        config_auto = f.read()

    new_file_lines = []
    deprecated_models_list = []
    in_deprecated_models = False
    for line in config_auto.split("\n"):
        if line.strip() == "DEPRECATED_MODELS = [":
            in_deprecated_models = True
            new_file_lines.append(line)
        elif in_deprecated_models and line.strip() == "]":
            in_deprecated_models = False
            # Add the new models to deprecated models list
            deprecated_models_list.extend([f'    "{model}", ' for model in models])
            # Sort so they're in alphabetical order in the file
            deprecated_models_list = sorted(deprecated_models_list)
            new_file_lines.extend(deprecated_models_list)
            # Make sure we still have the closing bracket
            new_file_lines.append(line)
        elif in_deprecated_models:
            deprecated_models_list.append(line)
        else:
            new_file_lines.append(line)

    with open(filepath, "w") as f:
        f.write("\n".join(new_file_lines))


def deprecate_models(models):
    # Get model info
    skipped_models = []
    models_info = defaultdict(dict)
    for model in models:
        single_model_info = extract_model_info(model)
        if single_model_info is None:
            skipped_models.append(model)
        else:
            models_info[model] = single_model_info

    model_config_classes = []
    for model, model_info in models_info.items():
        if model in CONFIG_MAPPING:
            model_config_classes.append(CONFIG_MAPPING[model].__name__)
        elif model_info["model_doc_name"] in CONFIG_MAPPING:
            model_config_classes.append(CONFIG_MAPPING[model_info["model_doc_name"]].__name__)
        else:
            skipped_models.append(model)
            print(f"Model config class not found for model: {model}")

    # Filter out skipped models
    models = [model for model in models if model not in skipped_models]

    if skipped_models:
        print(f"Skipped models: {skipped_models} as the model doc or model path could not be found.")
    print(f"Models to deprecate: {models}")

    # Remove model config classes from config check
    print("Removing model config classes from config checks")
    remove_model_config_classes_from_config_check(model_config_classes)

    tip_message = build_tip_message(get_last_stable_minor_release())

    for model, model_info in models_info.items():
        print(f"Processing model: {model}")
        # Add the tip message to the model doc page directly underneath the title
        print("Adding tip message to model doc page")
        insert_tip_to_model_doc(model_info["model_doc_path"], tip_message)

        # Remove #Copied from statements from model's files
        print("Removing #Copied from statements from model's files")
        remove_copied_from_statements(model)

        # Move the model file to deprecated: src/transfomers/models/model -> src/transformers/models/deprecated/model
        print("Moving model files to deprecated for model")
        move_model_files_to_deprecated(model)

        # Delete the model tests: tests/models/model
        print("Deleting model tests")
        delete_model_tests(model)

    # # We do the following with all models passed at once to avoid having to re-write the file multiple times
    print("Updating __init__.py file to point to the deprecated models")
    update_main_init_file(models)

    # Remove model references from other files
    print("Removing model references from other files")
    remove_model_references_from_file(
        "src/transformers/models/__init__.py", models, lambda line, model: model == line.strip().strip(",")
    )
    remove_model_references_from_file(
        "utils/slow_documentation_tests.txt", models, lambda line, model: "/" + model + "/" in line
    )
    remove_model_references_from_file("utils/not_doctested.txt", models, lambda line, model: "/" + model + "/" in line)

    # Add models to DEPRECATED_MODELS in the configuration_auto.py
    print("Adding models to DEPRECATED_MODELS in configuration_auto.py")
    add_models_to_deprecated_models_in_config_auto(models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="List of models to deprecate")
    args = parser.parse_args()
    deprecate_models(args.models)
