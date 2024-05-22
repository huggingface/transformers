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


class HubModelLister:
    """
    Utility for getting models from the hub based on tags. Handles errors without crashing the script.
    """

    def __init__(self, tags):
        self.tags = tags
        self.model_list = api.list_models(tags=tags)

    def __iter__(self):
        try:
            yield from self.model_list
        except Exception as e:
            print(f"Error: {e}")
            return


def _extract_commit_hash(commits):
    for commit in commits:
        if commit.startswith("commit "):
            return commit.split(" ")[1]
    return ""


def get_list_of_repo_model_paths(models_dir):
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
        models = [model for model in models if deprecated_model_name not in model]
    # Remove deprecated models
    models = [model for model in models if "/deprecated" not in model]
    # Remove auto
    models = [model for model in models if "/auto/" not in model]
    return models


def get_list_of_models_to_deprecate(
    thresh_num_downloads=5_000,
    thresh_date=None,
    use_cache=False,
    save_model_info=False,
    max_num_models=-1,
):
    if thresh_date is None:
        thresh_date = datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 1)
    else:
        thresh_date = datetime.strptime(thresh_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    models_dir = PATH_TO_REPO / "src/transformers/models"
    model_paths = get_list_of_repo_model_paths(models_dir=models_dir)

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
            models_info[model]["downloads"] = 0

            # Some tags on the hub are formatted differently than in the library
            tags = [model]
            if "_" in model:
                tags.append(model.replace("_", "-"))
            models_info[model]["tags"] = tags

        # Filter out models which were added less than a year ago
        models_info = {
            model: info for model, info in models_info.items() if info["first_commit_datetime"] < thresh_date
        }

        # We make successive calls to the hub, filtering based on the model tags
        n_seen = 0
        for model, model_info in models_info.items():
            for model_tag in model_info["tags"]:
                model_list = HubModelLister(tags=model_tag)
                for i, hub_model in enumerate(model_list):
                    n_seen += 1
                    if i % 100 == 0:
                        print(f"Processing model {i} for tag {model_tag}")
                    if max_num_models != -1 and i > n_seen:
                        break
                    if hub_model.private:
                        continue
                    model_info["downloads"] += hub_model.downloads

    if save_model_info and not (use_cache and os.path.exists("models_info.json")):
        # Make datetimes serializable
        for model, info in models_info.items():
            info["first_commit_datetime"] = info["first_commit_datetime"].isoformat()
        with open("models_info.json", "w") as f:
            json.dump(models_info, f, indent=4)

    print("\nFinding models to deprecate:")
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
    print("\nModels to deprecate: ", "\n" + "\n".join(models_to_deprecate.keys()))
    print(f"\nNumber of models to deprecate: {n_models_to_deprecate}")
    print("Before deprecating make sure to verify the models, including if they're used as a module in other models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model_info", action="store_true", help="Save the retrieved model info to a json file.")
    parser.add_argument(
        "--use_cache", action="store_true", help="Use the cached model info instead of calling the hub."
    )
    parser.add_argument(
        "--thresh_num_downloads",
        type=int,
        default=5_000,
        help="Threshold number of downloads below which a model should be deprecated. Default is 5,000.",
    )
    parser.add_argument(
        "--thresh_date",
        type=str,
        default=None,
        help="Date to consider the first commit from. Format: YYYY-MM-DD. If unset, defaults to one year ago from today.",
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
