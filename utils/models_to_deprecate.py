"""
Script to find a candidate list of models to deprecate based on the number of downloads and the date of the last commit.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from git import Repo
from huggingface_hub import HfApi


api = HfApi()

PATH_TO_REPO = Path(__file__).parent.parent.resolve()
repo = Repo(PATH_TO_REPO)


def _extract_commit_hash(commits):
    for commit in commits:
        if commit.startswith("commit "):
            return commit.split(" ")[1]
    return ""


def get_list_of_repo_model_paths(models_dir="src/transformers/models"):
    # Get list of all models in the library
    models = glob.glob(os.path.join(models_dir, "*/modeling_*.py"))

    # Remove flax and tf models
    models = [model for model in models if "_flax_" not in model]
    models = [model for model in models if "_tf_" not in model]

    # Get list of all deprecated models in the library
    deprecated_models = glob.glob(os.path.join(models_dir, "deprecated", "*"))
    # For each deprecated model, remove the deprecated models from the list of all models as well as the symlink path
    for deprecated_model in deprecated_models:
        deprecated_model_name = "/" + deprecated_model.split("/")[-1] + "/"
        models = [model for model in models if deprecated_model_name not in models]
    # Remove deprecated models
    models = [model for model in models if "/deprecated" not in model]
    # Remove init
    models = [model for model in models if "__init__" not in model]
    # Remove auto
    models = [model for model in models if "/auto/" not in model]
    return models


def get_list_of_models_to_deprecate(
    thresh_num_downloads=1_000,
    thresh_date=None,
    use_cache=False,
    save_model_info=False,
    max_num_models=-1,
):
    if thresh_date is None:
        thresh_date = datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 1)
    else:
        thresh_date = datetime.strptime(thresh_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    model_paths = get_list_of_repo_model_paths()

    if use_cache and os.path.exists("models_info.json"):
        with open("models_info.json", "r") as f:
            models_info = json.load(f)
        # Convert datetimes back to datetime objects
        for model, info in models_info.items():
            info["first_commit_datetime"] = datetime.fromisoformat(info["first_commit_datetime"])

    else:
        # Build a dictionary of model info: first commit datetime, commit hash, model path
        models_info = defaultdict(dict)
        for model_path in model_paths:
            model = model_path.split("/")[-2]
            if model in models_info:
                continue
            commits = repo.git.log("--diff-filter=A", "--", model_path).split("\n")
            commit_hash = _extract_commit_hash(commits)
            commit_obj = repo.commit(commit_hash)
            committed_datetime = commit_obj.committed_datetime
            models_info[model]["commit_hash"] = commit_hash
            models_info[model]["first_commit_datetime"] = committed_datetime
            models_info[model]["model_path"] = model_path

        # Filter out models which were added less than a year ago
        models_info = {
            model: info for model, info in models_info.items() if info["first_commit_datetime"] < thresh_date
        }

        # For each model on the hub, find it corresponding model based on its tags and update the number of downloads
        for model, info in models_info.items():
            models_info[model]["downloads"] = 0

        for i, hub_model in enumerate(api.list_models()):
            if i % 100 == 0:
                print(f"Processing model {i}")
            if max_num_models != -1 and i > max_num_models:
                break
            if hub_model.private:
                continue
            for tag in hub_model.tags:
                tag = tag.lower().replace("-", "_")
                if tag in models_info:
                    models_info[tag]["downloads"] += hub_model.downloads

    if save_model_info:
        # Make datetimes serializable
        for model, info in models_info.items():
            info["first_commit_datetime"] = info["first_commit_datetime"].isoformat()
        with open("models_info.json", "w") as f:
            json.dump(models_info, f, indent=4)

    print("\nModels to deprecate:")
    n_models_to_deprecate = 0
    models_to_deprecate = {}
    for model, info in models_info.items():
        n_downloads = info["downloads"]
        if n_downloads < thresh_num_downloads:
            n_models_to_deprecate += 1
            models_to_deprecate[model] = info
            print(f"\nModel: {model}")
            print(f"Downloads: {n_downloads}")
            print(f"Date: {info['first_commit_datetime']}")
    print(f"\nNumber of models to deprecate: {n_models_to_deprecate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="models_to_deprecate.json")
    parser.add_argument("--save-model-info", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument(
        "--thresh_num_downloads",
        type=int,
        default=1_000,
        help="Threshold number of downloads below which a model should be deprecated. Default is 1,000.",
    )
    parser.add_argument(
        "--thresh_date",
        type=str,
        default=None,
        help="Date to consider the first commit from. Format: YYYY-MM-DD, default is one year ago.",
    )
    parser.add_argument(
        "--max_num_models",
        type=int,
        default=-1,
        help="Maximum number of models to consider from the hub. -1 means all models. Useful for testing.",
    )
    args = parser.parse_args()

    models_to_deprecate = get_list_of_models_to_deprecate(
        thresh_num_downloads=args.thresh_num_downloads,
        thresh_date=args.thresh_date,
        use_cache=args.use_cache,
        save_model_info=args.save_model_info,
        max_num_models=args.max_num_models,
    )
