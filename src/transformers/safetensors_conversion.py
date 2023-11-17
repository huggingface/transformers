from typing import Optional

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
    import asyncio
    import json
    import uuid

    import websockets

    async def start(websocket, payload):
        _hash = str(uuid.uuid4())
        while True:
            data = await websocket.recv()
            print(f"<{data}")
            data = json.loads(data)
            if data["msg"] == "send_hash":
                data = json.dumps({"fn_index": 0, "session_hash": _hash})
                print(f">{data}")
                await websocket.send(data)
            elif data["msg"] == "send_data":
                data = json.dumps({"fn_index": 0, "session_hash": _hash, "data": payload})
                print(f">{data}")
                await websocket.send(data)
            elif data["msg"] == "process_completed":
                break

    async def main():
        print("======================")
        uri = "wss://safetensors-convert.hf.space/queue/join"
        async with websockets.connect(uri) as websocket:
            # inputs and parameters are classic, "id" is a way to track that query
            data = [token, model_id]
            try:
                await start(websocket, data)
            except Exception as e:
                print(f"Error during space conversion: {e}")

    asyncio.run(main())


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

        process = Process(target=spawn_conversion, args=("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", model_id))
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
