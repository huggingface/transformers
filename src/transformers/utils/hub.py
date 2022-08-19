# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Hub utilities: utilities related to download and cache models
"""
import json
import os
import re
import shutil
import sys
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import huggingface_hub
import requests
from huggingface_hub import (
    CommitOperationAdd,
    HfFolder,
    create_commit,
    create_repo,
    hf_hub_download,
    hf_hub_url,
    whoami,
)
from huggingface_hub.constants import HUGGINGFACE_HEADER_X_LINKED_ETAG, HUGGINGFACE_HEADER_X_REPO_COMMIT
from huggingface_hub.file_download import REGEX_COMMIT_HASH
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from requests.exceptions import HTTPError
from transformers.utils.logging import tqdm

from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    _tf_version,
    _torch_version,
    is_tf_available,
    is_torch_available,
    is_training_run_on_sagemaker,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def is_offline_mode():
    return _is_offline_mode


torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
old_default_cache_path = os.path.join(torch_cache_home, "transformers")
# New default cache, shared with the Datasets library
hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "hub")

# Onetime move from the old location to the new one if no ENV variable has been set.
if (
    os.path.isdir(old_default_cache_path)
    and not os.path.isdir(default_cache_path)
    and "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ
    and "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    and "TRANSFORMERS_CACHE" not in os.environ
):
    logger.warning(
        "In Transformers v4.0.0, the default path to cache downloaded models changed from"
        " '~/.cache/torch/transformers' to '~/.cache/huggingface/transformers'. Since you don't seem to have"
        " overridden and '~/.cache/torch/transformers' is a directory that exists, we're moving it to"
        " '~/.cache/huggingface/transformers' to avoid redownloading models you have already in the cache. You should"
        " only see this message once."
    )
    shutil.move(old_default_cache_path, default_cache_path)

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", PYTORCH_TRANSFORMERS_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", HUGGINGFACE_HUB_CACHE)
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"
SESSION_ID = uuid4().hex
DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", False) in ENV_VARS_TRUE_VALUES

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"

_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co"

HUGGINGFACE_CO_RESOLVE_ENDPOINT = _default_endpoint
if os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None) is not None:
    warnings.warn(
        "Using the environment variable `HUGGINGFACE_CO_RESOLVE_ENDPOINT` is deprecated and will be removed in "
        "Transformers v5. Use `HF_ENDPOINT` instead.",
        FutureWarning,
    )
    HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None)
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", HUGGINGFACE_CO_RESOLVE_ENDPOINT)
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"
HUGGINGFACE_CO_EXAMPLES_TELEMETRY = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/api/telemetry/examples"


def get_cached_models(cache_dir: Union[str, Path] = None) -> List[Tuple]:
    """
    Returns a list of tuples representing model binaries that are cached locally. Each tuple has shape `(model_url,
    etag, size_MB)`. Filenames in `cache_dir` are use to get the metadata for each model, only urls ending with *.bin*
    are added.

    Args:
        cache_dir (`Union[str, Path]`, *optional*):
            The cache directory to search for models within. Will default to the transformers cache if unset.

    Returns:
        List[Tuple]: List of tuples each with shape `(model_url, etag, size_MB)`
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    elif isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if not os.path.isdir(cache_dir):
        return []

    cached_models = []
    for file in os.listdir(cache_dir):
        if file.endswith(".json"):
            meta_path = os.path.join(cache_dir, file)
            with open(meta_path, encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                url = metadata["url"]
                etag = metadata["etag"]
                if url.endswith(".bin"):
                    size_MB = os.path.getsize(meta_path.strip(".json")) / 1e6
                    cached_models.append((url, etag, size_MB))

    return cached_models


def define_sagemaker_information():
    try:
        instance_data = requests.get(os.environ["ECS_CONTAINER_METADATA_URI"]).json()
        dlc_container_used = instance_data["Image"]
        dlc_tag = instance_data["Image"].split(":")[1]
    except Exception:
        dlc_container_used = None
        dlc_tag = None

    sagemaker_params = json.loads(os.getenv("SM_FRAMEWORK_PARAMS", "{}"))
    runs_distributed_training = True if "sagemaker_distributed_dataparallel_enabled" in sagemaker_params else False
    account_id = os.getenv("TRAINING_JOB_ARN").split(":")[4] if "TRAINING_JOB_ARN" in os.environ else None

    sagemaker_object = {
        "sm_framework": os.getenv("SM_FRAMEWORK_MODULE", None),
        "sm_region": os.getenv("AWS_REGION", None),
        "sm_number_gpu": os.getenv("SM_NUM_GPUS", 0),
        "sm_number_cpu": os.getenv("SM_NUM_CPUS", 0),
        "sm_distributed_training": runs_distributed_training,
        "sm_deep_learning_container": dlc_container_used,
        "sm_deep_learning_container_tag": dlc_tag,
        "sm_account_id": account_id,
    }
    return sagemaker_object


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    if is_training_run_on_sagemaker():
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in define_sagemaker_information().items())
    # CI will set this value to True
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]):
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash

    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


def try_to_load_from_cache(cache_dir, repo_id, filename, revision=None, commit_hash=None):
    """
    Explores the cache to return the latest cached file for a given revision.
    """
    if commit_hash is not None and revision is not None:
        raise ValueError("`commit_hash` and `revision` are mutually exclusive, pick one only.")
    if revision is None and commit_hash is None:
        revision = "main"

    model_id = repo_id.replace("/", "--")
    model_cache = os.path.join(cache_dir, f"models--{model_id}")
    if not os.path.isdir(model_cache):
        # No cache for this model
        return None
    for subfolder in ["refs", "snapshots"]:
        if not os.path.isdir(os.path.join(model_cache, subfolder)):
            return None

    if commit_hash is None:
        # Resolve refs (for instance to convert main to the associated commit sha)
        cached_refs = os.listdir(os.path.join(model_cache, "refs"))
        if revision in cached_refs:
            with open(os.path.join(model_cache, "refs", revision)) as f:
                commit_hash = f.read()

    cached_shas = os.listdir(os.path.join(model_cache, "snapshots"))
    if commit_hash not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    cached_file = os.path.join(model_cache, "snapshots", commit_hash, filename)
    return cached_file if os.path.isfile(cached_file) else None


# If huggingface_hub changes the class of error for this to FileNotFoundError, we will be able to avoid that in the
# future.
LOCAL_FILES_ONLY_HF_ERROR = (
    "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co "
    "look-ups and downloads online, set 'local_files_only' to False."
)


# In the future, this ugly contextmanager can be removed when huggingface_hub as a released version where we can
# activate/deactivate progress bars.
@contextmanager
def _patch_hf_hub_tqdm():
    """
    A context manager to make huggingface hub use the tqdm version of Transformers (which is controlled by some utils)
    in logging.
    """
    old_tqdm = huggingface_hub.file_download.tqdm
    huggingface_hub.file_download.tqdm = tqdm
    yield
    huggingface_hub.file_download.tqdm = old_tqdm


def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    user_agent: Optional[Union[str, Dict[str, str]]] = None,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `use_auth_token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("bert-base-uncased", "pytorch_model.bin")
    ```"""
    # Private arguments
    #     _raise_exceptions_for_missing_entries: if False, do not raise an exception for missing entries but return
    #         None.
    #     _raise_exceptions_for_connection_errors: if False, do not raise an exception for connection errors but return
    #         None.
    #     _commit_hash: passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
    #         a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(f"Could not locate {full_filename} inside {path_or_repo_id}.")
            else:
                return None
        return resolved_file

    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if _commit_hash is not None:
        # If the file is cached under that commit hash, we return it directly.
        resolved_file = try_to_load_from_cache(cache_dir, path_or_repo_id, full_filename, commit_hash=_commit_hash)
        if resolved_file is not None:
            return resolved_file

    user_agent = http_user_agent(user_agent)
    try:
        # Load from URL or cache if already cached
        with _patch_hf_hub_tqdm():
            resolved_file = hf_hub_download(
                path_or_repo_id,
                filename,
                subfolder=None if len(subfolder) == 0 else subfolder,
                revision=revision,
                cache_dir=cache_dir,
                user_agent=user_agent,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )

    except RepositoryNotFoundError:
        raise EnvironmentError(
            f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
            "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to "
            "pass a token having permission to this repo with `use_auth_token` or log in with "
            "`huggingface-cli login` and pass `use_auth_token=True`."
        )
    except RevisionNotFoundError:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
            "for this model name. Check the model page at "
            f"'https://huggingface.co/{path_or_repo_id}' for available revisions."
        )
    except EntryNotFoundError:
        if not _raise_exceptions_for_missing_entries:
            return None
        if revision is None:
            revision = "main"
        raise EnvironmentError(
            f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
            f"'https://huggingface.co/{path_or_repo_id}/{revision}' for available files."
        )
    except HTTPError as err:
        # First we try to see if we have a cached version (not up to date):
        resolved_file = try_to_load_from_cache(cache_dir, path_or_repo_id, full_filename, revision=revision)
        if resolved_file is not None:
            return resolved_file
        if not _raise_exceptions_for_connection_errors:
            return None

        raise EnvironmentError(f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}")
    except ValueError as err:
        # HuggingFace Hub returns a ValueError for a missing file when local_files_only=True we need to catch it here
        # This could be caught above along in `EntryNotFoundError` if hf_hub sent a different error message here
        if LOCAL_FILES_ONLY_HF_ERROR in err.args[0] and local_files_only and not _raise_exceptions_for_missing_entries:
            return None

        # Otherwise we try to see if we have a cached version (not up to date):
        resolved_file = try_to_load_from_cache(cache_dir, path_or_repo_id, full_filename, revision=revision)
        if resolved_file is not None:
            return resolved_file
        if not _raise_exceptions_for_connection_errors:
            return None
        raise EnvironmentError(
            f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the"
            f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file named"
            f" {full_filename}.\nCheckout your internet connection or see how to run the library in offline mode at"
            " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        )

    return resolved_file


def get_file_from_repo(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `use_auth_token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo) or `None` if the
        file does not exist.

    Examples:

    ```python
    # Download a tokenizer configuration from huggingface.co and cache.
    tokenizer_config = get_file_from_repo("bert-base-uncased", "tokenizer_config.json")
    # This model does not have a tokenizer config so the result will be None.
    tokenizer_config = get_file_from_repo("xlm-roberta-base", "tokenizer_config.json")
    ```"""
    return cached_file(
        path_or_repo_id=path_or_repo,
        filename=filename,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        use_auth_token=use_auth_token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )


def has_file(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    revision: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
):
    """
    Checks if a repo contains a given file wihtout downloading it. Works for remote repos and local folders.

    <Tip warning={false}>

    This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist for
    this repo, but will return False for regular connection errors.

    </Tip>
    """
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    url = hf_hub_url(path_or_repo, filename=filename, revision=revision)

    headers = {"user-agent": http_user_agent()}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
        headers["authorization"] = f"Bearer {token}"

    r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=10)
    try:
        huggingface_hub.utils._errors._raise_for_status(r)
        return True
    except RepositoryNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(f"{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'.")
    except RevisionNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
            f"model name. Check the model page at 'https://huggingface.co/{path_or_repo}' for available revisions."
        )
    except requests.HTTPError:
        # We return false for EntryNotFoundError (logical) as well as any connection error.
        return False


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def _create_repo(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        """
        Create the repo if needed, cleans up repo_id with deprecated kwards `repo_url` and `organization`, retrives the
        token.
        """
        if repo_url is not None:
            warnings.warn(
                "The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` "
                "instead."
            )
            repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")
        if organization is not None:
            warnings.warn(
                "The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your "
                "organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`)."
            )
            if not repo_id.startswith(organization):
                if "/" in repo_id:
                    repo_id = repo_id.split("/")[-1]
                repo_id = f"{organization}/{repo_id}"

        token = HfFolder.get_token() if use_auth_token is True else use_auth_token
        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)

        # If the namespace is not there, add it or `upload_file` will complain
        if "/" not in repo_id and url != f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{repo_id}":
            repo_id = get_full_repo_name(repo_id, token=token)
        return repo_id, token

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        files_timestamps: Dict[str, float],
        commit_message: Optional[str] = None,
        token: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]
        operations = []
        for file in modified_files:
            operations.append(CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file))
        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        return create_commit(
            repo_id=repo_id, operations=operations, commit_message=commit_message, token=token, create_pr=create_pr
        )

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        create_pr: bool = False,
        **deprecated_kwargs
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub while synchronizing a local clone of the repo in
        `repo_path_or_name`.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private (requires a paying subscription).
            use_auth_token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.

        Examples:

        ```python
        from transformers import {object_class}

        {object} = {object_class}.from_pretrained("bert-base-cased")

        # Push the {object} to your namespace with the name "my-finetuned-bert".
        {object}.push_to_hub("my-finetuned-bert")

        # Push the {object} to an organization with the name "my-finetuned-bert".
        {object}.push_to_hub("huggingface/my-finetuned-bert")
        ```
        """
        if "repo_path_or_name" in deprecated_kwargs:
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead."
            )
            repo_id = deprecated_kwargs.pop("repo_path_or_name")
        # Deprecation warning will be sent after for repo_url and organization
        repo_url = deprecated_kwargs.pop("repo_url", None)
        organization = deprecated_kwargs.pop("organization", None)

        if os.path.isdir(repo_id):
            working_dir = repo_id
            repo_id = repo_id.split(os.path.sep)[-1]
        else:
            working_dir = repo_id.split("/")[-1]

        repo_id, token = self._create_repo(
            repo_id, private=private, use_auth_token=use_auth_token, repo_url=repo_url, organization=organization
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size)

            return self._upload_modified_files(
                work_dir, repo_id, files_timestamps, commit_message=commit_message, token=token, create_pr=create_pr
            )


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def send_example_telemetry(example_name, *example_args, framework="pytorch"):
    """
    Sends telemetry that helps tracking the examples use.

    Args:
        example_name (`str`): The name of the example.
        *example_args (dataclasses or `argparse.ArgumentParser`): The arguments to the script. This function will only
            try to extract the model and dataset name from those. Nothing else is tracked.
        framework (`str`, *optional*, defaults to `"pytorch"`): The framework for the example.
    """
    if is_offline_mode():
        return

    data = {"example": example_name, "framework": framework}
    for args in example_args:
        args_as_dict = {k: v for k, v in args.__dict__.items() if not k.startswith("_") and v is not None}
        if "model_name_or_path" in args_as_dict:
            model_name = args_as_dict["model_name_or_path"]
            # Filter out local paths
            if not os.path.isdir(model_name):
                data["model_name"] = args_as_dict["model_name_or_path"]
        if "dataset_name" in args_as_dict:
            data["dataset_name"] = args_as_dict["dataset_name"]
        elif "task_name" in args_as_dict:
            # Extract script name from the example_name
            script_name = example_name.replace("tf_", "").replace("flax_", "").replace("run_", "")
            script_name = script_name.replace("_no_trainer", "")
            data["dataset_name"] = f"{script_name}-{args_as_dict['task_name']}"

    headers = {"user-agent": http_user_agent(data)}
    try:
        r = requests.head(HUGGINGFACE_CO_EXAMPLES_TELEMETRY, headers=headers)
        r.raise_for_status()
    except Exception:
        # We don't want to error in case of connection errors of any kind.
        pass


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:
    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    use_auth_token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    _commit_hash=None,
):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    import json

    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(list(set(index["weight_map"].values())))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []
    for shard_filename in shard_filenames:
        try:
            # Load from URL
            cached_filename = cached_file(
                pretrained_model_name_or_path,
                shard_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=_commit_hash,
            )
        # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
        # we don't have to catch them here.
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            )
        except HTTPError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try"
                " again after checking your internet connection."
            )

        cached_filenames.append(cached_filename)

    return cached_filenames, sharded_metadata


# All what is below is for conversion between old cache format and new cache format.


def get_all_cached_files(cache_dir=None):
    """
    Returns a list for all files cached with appropriate metadata.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = str(cache_dir)
    if not os.path.isdir(cache_dir):
        return []

    cached_files = []
    for file in os.listdir(cache_dir):
        meta_path = os.path.join(cache_dir, f"{file}.json")
        if not os.path.isfile(meta_path):
            continue

        with open(meta_path, encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
            url = metadata["url"]
            etag = metadata["etag"].replace('"', "")
            cached_files.append({"file": file, "url": url, "etag": etag})

    return cached_files


def get_hub_metadata(url, token=None):
    """
    Returns the commit hash and associated etag for a given url.
    """
    if token is None:
        token = HfFolder.get_token()
    headers = {"user-agent": http_user_agent()}
    headers["authorization"] = f"Bearer {token}"

    r = huggingface_hub.file_download._request_with_retry(
        method="HEAD", url=url, headers=headers, allow_redirects=False
    )
    huggingface_hub.file_download._raise_for_status(r)
    commit_hash = r.headers.get(HUGGINGFACE_HEADER_X_REPO_COMMIT)
    etag = r.headers.get(HUGGINGFACE_HEADER_X_LINKED_ETAG) or r.headers.get("ETag")
    if etag is not None:
        etag = huggingface_hub.file_download._normalize_etag(etag)
    return etag, commit_hash


def extract_info_from_url(url):
    """
    Extract repo_name, revision and filename from an url.
    """
    search = re.search(r"^https://huggingface\.co/(.*)/resolve/([^/]*)/(.*)$", url)
    if search is None:
        return None
    repo, revision, filename = search.groups()
    cache_repo = "--".join(["models"] + repo.split("/"))
    return {"repo": cache_repo, "revision": revision, "filename": filename}


def clean_files_for(file):
    """
    Remove, if they exist, file, file.json and file.lock
    """
    for f in [file, f"{file}.json", f"{file}.lock"]:
        if os.path.isfile(f):
            os.remove(f)


def move_to_new_cache(file, repo, filename, revision, etag, commit_hash):
    """
    Move file to repo following the new huggingface hub cache organization.
    """
    os.makedirs(repo, exist_ok=True)

    # refs
    os.makedirs(os.path.join(repo, "refs"), exist_ok=True)
    if revision != commit_hash:
        ref_path = os.path.join(repo, "refs", revision)
        with open(ref_path, "w") as f:
            f.write(commit_hash)

    # blobs
    os.makedirs(os.path.join(repo, "blobs"), exist_ok=True)
    blob_path = os.path.join(repo, "blobs", etag)
    shutil.move(file, blob_path)

    # snapshots
    os.makedirs(os.path.join(repo, "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(repo, "snapshots", commit_hash), exist_ok=True)
    pointer_path = os.path.join(repo, "snapshots", commit_hash, filename)
    huggingface_hub.file_download._create_relative_symlink(blob_path, pointer_path)
    clean_files_for(file)


def move_cache(cache_dir=None, new_cache_dir=None, token=None):
    if new_cache_dir is None:
        new_cache_dir = TRANSFORMERS_CACHE
    if cache_dir is None:
        # Migrate from old cache in .cache/huggingface/hub
        old_cache = Path(TRANSFORMERS_CACHE).parent / "transformers"
        if os.path.isdir(str(old_cache)):
            cache_dir = str(old_cache)
        else:
            cache_dir = new_cache_dir
    if token is None:
        token = HfFolder.get_token()
    cached_files = get_all_cached_files(cache_dir=cache_dir)
    print(f"Moving {len(cached_files)} files to the new cache system")

    hub_metadata = {}
    for file_info in tqdm(cached_files):
        url = file_info.pop("url")
        if url not in hub_metadata:
            try:
                hub_metadata[url] = get_hub_metadata(url, token=token)
            except requests.HTTPError:
                continue

        etag, commit_hash = hub_metadata[url]
        if etag is None or commit_hash is None:
            continue

        if file_info["etag"] != etag:
            # Cached file is not up to date, we just throw it as a new version will be downloaded anyway.
            clean_files_for(os.path.join(cache_dir, file_info["file"]))
            continue

        url_info = extract_info_from_url(url)
        if url_info is None:
            # Not a file from huggingface.co
            continue

        repo = os.path.join(new_cache_dir, url_info["repo"])
        move_to_new_cache(
            file=os.path.join(cache_dir, file_info["file"]),
            repo=repo,
            filename=url_info["filename"],
            revision=url_info["revision"],
            etag=etag,
            commit_hash=commit_hash,
        )


cache_version_file = os.path.join(TRANSFORMERS_CACHE, "version.txt")
if not os.path.isfile(cache_version_file):
    cache_version = 0
else:
    with open(cache_version_file) as f:
        cache_version = int(f.read())


if cache_version < 1:
    if is_offline_mode():
        logger.warn(
            "You are offline and the cache for model files in Transformers v4.22.0 has been updated while your local "
            "cache seems to be the one of a previous version. It is very likely that all your calls to any "
            "`from_pretrained()` method will fail. Remove the offline mode and enable internet connection to have "
            "your cache be updated automatically, then you can go back to offline mode."
        )
    else:
        logger.warn(
            "The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a "
            "one-time only operation. You can interrupt this and resume the migration later on by calling "
            "`transformers.utils.move_cache()`."
        )
    try:
        if TRANSFORMERS_CACHE != default_cache_path:
            # Users set some env variable to customize cache storage
            move_cache(TRANSFORMERS_CACHE, TRANSFORMERS_CACHE)
        else:
            move_cache()
    except Exception as e:
        trace = "\n".join(traceback.format_tb(e.__traceback__))
        logger.error(
            f"There was a problem when trying to move your cache:\n\n{trace}\n\nPlease file an issue at "
            "https://github.com/huggingface/transformers/issues/new/choose and copy paste this whole message and we "
            "will do our best to help."
        )

    try:
        os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)
        with open(cache_version_file, "w") as f:
            f.write("1")
    except Exception:
        logger.warn(
            f"There was a problem when trying to write in your cache folder ({TRANSFORMERS_CACHE}). You should set "
            "the environment variable TRANSFORMERS_CACHE to a writable directory."
        )
