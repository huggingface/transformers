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
"""A script running `create_dummy_models.py` with a pre-defined set of arguments.

This file is intended to be used in a CI workflow file without the need of specifying arguments. It creates and uploads
tiny models for all new model classes (i.e. those not yet present in the summary), and updates the tiny model summary
on the Hub (specified via --summary-repo-id). The summary repo acts as the source of truth: it is fetched at the start
to determine which models to skip, and the merged result is pushed back at the end.
"""

import argparse
import json
import multiprocessing
import os
import time

from create_dummy_models import COMPOSITE_MODELS, create_tiny_models
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

import transformers
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer, logging
from transformers.image_processing_utils import BaseImageProcessor


logger = logging.get_logger(__name__)


def get_all_model_names():
    model_names = set()

    module_name = "modeling_auto"
    module = getattr(transformers.models.auto, module_name, None)
    if module is not None:
        # all mappings in a single auto modeling file
        mapping_names = [x for x in dir(module) if x.endswith("_MAPPING_NAMES") and x.startswith("MODEL_")]
        for name in mapping_names:
            mapping = getattr(module, name)
            if mapping is not None:
                for v in mapping.values():
                    if isinstance(v, (list, tuple)):
                        model_names.update(v)
                    elif isinstance(v, str):
                        model_names.add(v)

    return sorted(model_names)


def get_tiny_model_names_from_repo(summary_repo_id):
    try:
        path = hf_hub_download(repo_id=summary_repo_id, filename="tiny_model_summary.json")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []
    with open(path) as fp:
        tiny_model_info = json.load(fp)
    tiny_models_names = set()
    for model_base_name in tiny_model_info:
        tiny_models_names.update(tiny_model_info[model_base_name]["model_classes"])

    return sorted(tiny_models_names)


def get_tiny_model_summary_from_hub(output_path):
    api = HfApi()
    special_models = COMPOSITE_MODELS.values()

    # All tiny model base names on Hub
    model_names = get_all_model_names()
    models = api.list_models(author="hf-internal-testing")
    _models = set()
    for x in models:
        model = x.id
        org, model = model.split("/")
        if not model.startswith("tiny-random-"):
            continue
        model = model.replace("tiny-random-", "")
        if not model[0].isupper():
            continue
        if model not in model_names and model not in special_models:
            continue
        _models.add(model)

    models = sorted(_models)
    # All tiny model names on Hub
    summary = {}
    for model in models:
        repo_id = f"hf-internal-testing/tiny-random-{model}"
        model = model.split("-")[0]
        try:
            repo_info = api.repo_info(repo_id)
            content = {
                "tokenizer_classes": set(),
                "processor_classes": set(),
                "model_classes": set(),
                "sha": repo_info.sha,
            }
        except Exception:
            continue
        try:
            time.sleep(1)
            tokenizer_fast = AutoTokenizer.from_pretrained(repo_id)
            content["tokenizer_classes"].add(tokenizer_fast.__class__.__name__)
        except Exception as e:
            logger.debug(f"Could not load fast tokenizer for {repo_id}: {e}")
        try:
            time.sleep(1)
            tokenizer_slow = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
            content["tokenizer_classes"].add(tokenizer_slow.__class__.__name__)
        except Exception as e:
            logger.debug(f"Could not load slow tokenizer for {repo_id}: {e}")
        try:
            time.sleep(1)
            img_p = AutoImageProcessor.from_pretrained(repo_id)
            content["processor_classes"].add(img_p.__class__.__name__)
        except Exception as e:
            logger.debug(f"Could not load image processor for {repo_id}: {e}")
        try:
            time.sleep(1)
            feat_p = AutoFeatureExtractor.from_pretrained(repo_id)
            if not isinstance(feat_p, BaseImageProcessor):
                content["processor_classes"].add(feat_p.__class__.__name__)
        except Exception as e:
            logger.debug(f"Could not load feature extractor for {repo_id}: {e}")
        try:
            time.sleep(1)
            model_class = getattr(transformers, model)
            m = model_class.from_pretrained(repo_id)
            content["model_classes"].add(m.__class__.__name__)
        except Exception as e:
            logger.debug(f"Could not load model for {repo_id}: {e}")

        content["tokenizer_classes"] = sorted(content["tokenizer_classes"])
        content["processor_classes"] = sorted(content["processor_classes"])
        content["model_classes"] = sorted(content["model_classes"])

        summary[model] = content
        with open(os.path.join(output_path, "hub_tiny_model_summary.json"), "w") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=1, type=int, help="The number of workers to run.")
    parser.add_argument("--organization", required=True, type=str, help="The Hugging Face organization to upload tiny models to.")
    parser.add_argument("--ignore-existing", action="store_true", help="Create and upload all models, ignoring the existing tiny model summary.")
    parser.add_argument("--summary-repo-id", default=None, type=str, help="Hub repo ID to fetch the base tiny_model_summary.json from and upload the merged result back to.")
    parser.add_argument("--model-types", default=None, type=str, help="Comma-separated list of model types to create. If not set, all model types are created.")
    args = parser.parse_args()

    # This has to be `spawn` to avoid hanging forever!
    multiprocessing.set_start_method("spawn")

    output_path = "tiny_models"
    model_types = [m.strip() for m in args.model_types.split(",")] if args.model_types else None
    all = model_types is None
    models_to_skip = set() if (args.ignore_existing or args.summary_repo_id is None) else get_tiny_model_names_from_repo(args.summary_repo_id)
    no_check = True
    upload = True
    organization = args.organization
    # When --ignore-existing, don't merge: we're doing a fresh run so the Hub base is irrelevant.
    summary_repo_id = None if args.ignore_existing else args.summary_repo_id

    create_tiny_models(
        output_path,
        all,
        model_types,
        models_to_skip,
        no_check,
        upload,
        organization,
        token=os.environ.get("TOKEN", None),
        num_workers=args.num_workers,
        summary_repo_id=summary_repo_id,
    )
