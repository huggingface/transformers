import argparse
import json
import re
import string
from functools import cache

from transformers.models.auto.configuration_auto import CONFIG_MAPPING, model_type_to_module_name


MAX_NUM_JOBS_TO_SUGGEST = 16


def _iter_descendants(config, seen: set[str]):
    """Yield the model_type of every sub-config under `config`, at any depth, each once."""
    specs = getattr(config, "sub_configs_defaults", None) or {}
    for spec in specs.values():
        model_type = getattr(spec, "model_type", None)
        if not model_type or model_type in seen:
            continue
        seen.add(model_type)
        yield model_type
        yield from _iter_descendants(spec, seen)


@cache
def _reverse_backbone_map() -> dict[str, list[str]]:
    """
    {backbone_model: [composite_models, ...]}, e.g. `clip` -> [`llava`], so that a change to `clip`
    triggers the tests of `llava`. Nesting is followed at any depth: if `a` contains `b` which
    contains `c`, then `c` maps to both `b` and `a`. Built once per process: the config imports
    are the expensive part.
    """
    reverse_map: dict[str, set[str]] = {}
    for model_type in CONFIG_MAPPING:
        parent = model_type_to_module_name(model_type)
        for descendant in _iter_descendants(CONFIG_MAPPING[model_type], {model_type}):
            child = model_type_to_module_name(descendant)
            if child != parent:
                reverse_map.setdefault(child, set()).add(parent)

    return {child: sorted(parents) for child, parents in reverse_map.items()}


def get_composite_files(backbone_name: str) -> list[str, ...]:
    """Composite-model files that depend on `backbone_file` (empty if none)."""
    return _reverse_backbone_map().get(backbone_name, [])


def get_jobs_to_run():
    # The file `pr_files.txt` contains the information about the files changed in a pull request, and it is prepared by
    # the caller (using GitHub api).
    # We can also use the following api to get the information if we don't have them before calling this script.
    # url = f"https://api.github.com/repos/huggingface/transformers/pulls/PULL_NUMBER/files?ref={pr_sha}"
    with open("pr_files.txt", encoding="utf-8") as fp:
        pr_files = json.load(fp)
        pr_files = [{k: v for k, v in item.items() if k in ["filename", "status"]} for item in pr_files]
    pr_files = [item["filename"] for item in pr_files if item["status"] in ["added", "modified"]]

    # models or quantizers
    re_1 = re.compile(r"src/transformers/(models/.*)/modeling_.*\.py")
    re_2 = re.compile(r"src/transformers/(quantizers/quantizer_.*)\.py")

    # tests for models or quantizers
    re_3 = re.compile(r"tests/(models/.*)/test_.*\.py")
    re_4 = re.compile(r"tests/(quantization/.*)/test_.*\.py")

    # files in a model directory but not necessary a modeling file
    re_5 = re.compile(r"src/transformers/(models/.*)/.*\.py")

    regexes = [re_1, re_2, re_3, re_4, re_5]

    jobs_to_run = []
    for pr_file in pr_files:
        for regex in regexes:
            matched = regex.findall(pr_file)
            if len(matched) > 0:
                item = matched[0]
                item = item.replace("quantizers/quantizer_", "quantization/")
                # TODO: for files in `quantizers`, the processed item above may not exist. Try using a fuzzy matching
                if item in repo_content:
                    jobs_to_run.append(item)
                if multimodal_parents := get_composite_files(item):
                    jobs_to_run.extend(multimodal_parents)
                break
    jobs_to_run = sorted(set(jobs_to_run))

    return jobs_to_run


def parse_message(message: str) -> str:
    """
    Parses a GitHub pull request's comment to find the models specified in it to run slow CI.

    Args:
        message (`str`): The body of a GitHub pull request's comment.

    Returns:
        `str`: The substring in `message` after `run-slow`, run_slow` or run slow`. If no such prefix is found, the
        empty string is returned.
    """
    if message is None:
        return ""

    message = message.strip().lower()

    # run-slow: model_1, model_2, quantization_1, quantization_2
    if not message.startswith(("run-slow", "run_slow", "run slow")):
        return ""
    message = message[len("run slow") :]
    # remove leading `:`
    while message.strip().startswith(":"):
        message = message.strip()[1:]

    return message


def get_jobs(message: str):
    models = parse_message(message)
    return models.replace(",", " ").split()


def check_name(model_name: str):
    allowed = string.ascii_letters + string.digits + "_"
    return not (model_name.startswith("_") or model_name.endswith("_")) and all(c in allowed for c in model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="", help="The content of a comment.")
    parser.add_argument("--quantization", action="store_true", help="If we collect quantization tests")
    args = parser.parse_args()

    # The files are prepared by the caller (using GitHub api).
    # We can also use the following api to get the information if we don't have them before calling this script.
    # url = f"https://api.github.com/repos/OWNER/REPO/contents/PATH?ref={pr_sha}"
    # (we avoid to checkout the repository using `actions/checkout` to reduce the run time, but mostly to avoid the potential security issue as much as possible)
    repo_content = []
    for filename in ["tests_dir.txt", "tests_models_dir.txt", "tests_quantization_dir.txt"]:
        with open(filename, encoding="utf-8") as fp:
            data = json.load(fp)
            data = [item["path"][len("tests/") :] for item in data if item["type"] == "dir"]
            repo_content.extend(data)

    # These don't have the prefix `models/` or `quantization/`, so we need to add them.
    if args.message:
        specified_jobs = get_jobs(args.message)
        specified_jobs = [job for job in specified_jobs if check_name(job)]

        # Add prefix (`models/` or `quantization`)
        jobs_to_run = []
        for job in specified_jobs:
            if not args.quantization:
                if f"models/{job}" in repo_content:
                    jobs_to_run.append(f"models/{job}")
                elif job in repo_content and job != "quantization":
                    jobs_to_run.append(job)
            elif f"quantization/{job}" in repo_content:
                jobs_to_run.append(f"quantization/{job}")

        print(sorted(set(jobs_to_run)))

    else:
        # Compute (from the added/modified files) the directories under `tests/`, `tests/models/` and `tests/quantization`to run tests.
        # These are already with the prefix `models/` or `quantization/`, so we don't need to add them.
        jobs_to_run = get_jobs_to_run()
        jobs_to_run = [x.replace("models/", "").replace("quantization/", "") for x in jobs_to_run]
        jobs_to_run = [job for job in jobs_to_run if check_name(job)]

        if len(jobs_to_run) > MAX_NUM_JOBS_TO_SUGGEST:
            jobs_to_run = jobs_to_run[:MAX_NUM_JOBS_TO_SUGGEST]

        suggestion = f"{', '.join(jobs_to_run)}"

        print(suggestion)
