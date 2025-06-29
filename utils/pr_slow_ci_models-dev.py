import argparse
import json

import requests
import sys
import os
import re

# def get_pr(pr_number):
#     from github import Github
#
#     g = Github()
#     repo = g.get_repo("huggingface/transformers")
#     pr = repo.get_pull(pr_number)
#
#     print(pr)
#     for file in pr.get_files():
#         print(file)
#         print(file.filename)
#         print(file.status)


def get_pr_files():

    import json
    fp = open("pr_files.txt")
    files = json.load(fp)
    fp.close()
    files = [{k: v for k, v in item.items() if k in ["filename", "status"]} for item in files]
    print(files)

    # TODO: get directories under `(tests/)models/xxx`, `(tests/)models/quantization` and `(tests/)xxx`
    # GOAL: get new modeling files / get list of test files to suggest to run / match a list of specified items to run

    new_files = [item["filename"] for item in files if item["status"] == "added"]
    modified_files = [item["filename"] for item in files if item["status"] == "modified"]

    print(new_files)
    print(modified_files)

    # models or quantizers
    file_re_1 = re.compile(r"src/transformers/(models/.*)/modeling_.*\.py")

    # Unfortunately, there is no proper way to map this to quantization tests.
    file_re_2 = re.compile(r"src/transformers/(quantizers/quantizer_.*)\.py")

    # tests for models or quantizers
    file_re_3 = re.compile(r"tests/(models/.*)/test_.*\.py")
    file_re_4 = re.compile(r"tests/(quantization/.*)/test_.*\.py")

    # directories of models or quantizers
    file_re_5 = re.compile(r"src/transformers/(models/.*)/.*\.py")


    regexes = [file_re_1, file_re_2, file_re_3, file_re_4, file_re_5]

    new_files_to_run = []
    for new_file in new_files:
        for regex in regexes:
            matched = regex.findall(new_file)
            if len(matched) > 0:
                item = matched[0]
                item = item.replace("quantizers/quantizer_", "quantization/")
                new_files_to_run.append(item)
                break

    modified_files_to_run = []
    for modified_file in modified_files:
        for regex in regexes:
            matched = regex.findall(modified_file)
            if len(matched) > 0:
                item = matched[0]
                item = item.replace("quantizers/quantizer_", "quantization/")
                modified_files_to_run.append(item)
                break

    new_files_to_run = sorted(set(new_files_to_run))
    modified_files_to_run = sorted(set(modified_files_to_run))

    print(new_files_to_run)
    print(modified_files_to_run)

    return new_files_to_run, modified_files_to_run


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


def get_models(message: str):
    models = parse_message(message)
    return models.replace(",", " ").split()


if __name__ == '__main__':
    # pr_number = "39100"
    # pr_number = int(pr_number)
    # get_pr2(pr_number)

    # # get file information without checkout
    # pr_number = "39100"
    # pr_sha = "d213aefed5922956a92d47d5f1bc806a562936cf"
    # pr_sha = "7e6427d2091156aa0c02a31efc55744046cf85ac"
    #
    # # use `refs/pull/39100/head` is not good!
    # # but if we want to use sha value, it has to be `OWNER/REPO`
    # url = f"https://api.github.com/repos/huggingface/transformers/contents/tests/quantization?ref={pr_sha}"
    # response = requests.get(url)
    # data = response.json()
    # print(data)
    #
    # url = f"https://api.github.com/repos/huggingface/transformers/contents/tests/models?ref={pr_sha}"
    # response = requests.get(url)
    # data = response.json()
    # print(json.dumps(data, indent=4))

    # exit(0)

    import json

    for filename in ["tests_dir.txt", "tests_models_dir.txt", "tests_quantization_dir.txt"]:
        with open(filename) as fp:
            data = json.load(fp)
            # data = [{k: v for k, v in item.items() if k in ["filename", "status"]} for item in data]
            print(data)

    exit(0)


    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="", help="The content of a comment.")
    args = parser.parse_args()

    # These don't have the prefix `models/` or `quantization/`, so we need to add them.
    # At this moment, we don't know if they are in tests/models or in tests/quantization, or if they even exist
    specified_models = []
    if args.message:
        specified_models = get_models(args.message)

    # Computed from the changed files.
    # These are already with the prefix `models/` or `quantization/`, so we don't need to add them.
    # TODO: However, we don't know if the inferred directories in tests/quantization actually exist
    new_files_to_run, modified_files_to_run = get_pr_files()

