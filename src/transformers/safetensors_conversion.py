from typing import Optional

import requests
from huggingface_hub import Discussion, HfApi

from .utils import cached_file, logging


logger = logging.get_logger(__name__)


def previous_pr(api: "HfApi", model_id: str, pr_title: str) -> Optional["Discussion"]:
    try:
        main_commit = api.list_repo_commits(model_id)[0].commit_id
        discussions = api.get_repo_discussions(repo_id=model_id)
    except Exception:
        return None
    for discussion in discussions:
        if discussion.status == "open" and discussion.is_pull_request and discussion.title == pr_title:
            commits = api.list_repo_commits(model_id, revision=discussion.git_reference)

            if main_commit == commits[1].commit_id:
                return discussion
    return None


def spawn_conversion(token: str, model_id: str):
    print("Sending conversion request")
    import json
    import uuid

    sse_url = "https://safetensors-convert.hf.space/queue/join"
    sse_data_url = "https://safetensors-convert.hf.space/queue/data"
    hash_data = {"fn_index": 1, "session_hash": str(uuid.uuid4())}

    def start(_sse_connection, payload):
        for line in _sse_connection.iter_lines():
            line = line.decode()
            if line.startswith("data:"):
                resp = json.loads(line[5:])
                print(resp)

                if resp["msg"] == "queue_full":
                    raise ValueError("Queue is full! Please try again.")
                elif resp["msg"] == "send_data":
                    event_id = resp["event_id"]
                    response = requests.post(
                        sse_data_url,
                        stream=True,
                        params=hash_data,
                        json={"event_id": event_id, **payload, **hash_data},
                    )
                    response.raise_for_status()
                elif resp["msg"] == "process_completed":
                    return

    print("======================")

    with requests.get(sse_url, stream=True, params=hash_data) as sse_connection:
        data = {"data": [model_id, False, token]}
        try:
            start(sse_connection, data)
        except Exception as e:
            print(f"Error during space conversion: {repr(e)}")


def get_sha(model_id: str, filename: str, **kwargs):
    api = HfApi(token=kwargs.get("token"))
    # model_info = api.model_info(model_id)
    # refs = api.list_repo_refs(model_id)

    # main_refs = [branch.target_commit for branch in refs.branches if branch.ref == "refs/heads/main"]
    # main_sha = None
    # if main_refs:
    #     main_sha = main_refs[0]

    logger.info("Attempting to create safetensors variant")
    pr_title = "Adding `safetensors` variant of this model"
    pr = previous_pr(api, model_id, pr_title)
    if pr is None:
        from multiprocessing import Process

        process = Process(target=spawn_conversion, args=(kwargs.get("token"), model_id))
        process.start()
        process.join()
        pr = previous_pr(api, model_id, pr_title)
        sha = f"refs/pr/{pr.num}"
    else:
        logger.info("Safetensors PR exists")
        sha = f"refs/pr/{pr.num}"
    return sha


def auto_conversion(pretrained_model_name_or_path: str, filename: str, **cached_file_kwargs):
    sha = get_sha(pretrained_model_name_or_path, filename, **cached_file_kwargs)
    if sha is None:
        return None, None
    cached_file_kwargs["revision"] = sha
    del cached_file_kwargs["_commit_hash"]
    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
    return resolved_archive_file, sha
