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
import sys
import tempfile
from concurrent import futures
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import httpx
from huggingface_hub import (
    _CACHED_NO_EXIST,
    CommitOperationAdd,
    ModelCard,
    ModelCardData,
    constants,
    create_branch,
    create_commit,
    create_repo,
    hf_hub_download,
    hf_hub_url,
    is_offline_mode,
    list_repo_tree,
    snapshot_download,
    try_to_load_from_cache,
)
from huggingface_hub.file_download import REGEX_COMMIT_HASH
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    build_hf_headers,
    get_session,
    hf_raise_for_status,
)

from . import __version__, logging
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    get_torch_version,
    is_torch_available,
    is_training_run_on_sagemaker,
)


LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE = "chat_template.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"
CHAT_TEMPLATE_DIR = "additional_chat_templates"


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DownloadKwargs(TypedDict, total=False):
    cache_dir: str | os.PathLike | None
    force_download: bool
    proxies: dict[str, str] | None
    local_files_only: bool
    token: str | bool | None
    revision: str | None
    subfolder: str
    commit_hash: str | None


# Determine default cache directory.
# The best way to set the cache path is with the environment variable HF_HOME. For more details, check out this
# documentation page: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables.

HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(constants.HF_HOME, "modules"))
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"
SESSION_ID = uuid4().hex

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"


def _get_cache_file_to_return(
    path_or_repo_id: str,
    full_filename: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    repo_type: str | None = None,
):
    # We try to see if we have a cached version (not up to date):
    resolved_file = try_to_load_from_cache(
        path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision, repo_type=repo_type
    )
    if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
        return resolved_file
    return None


def list_repo_templates(
    repo_id: str,
    *,
    local_files_only: bool,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    """List template files from a repo.

    A template is a jinja file located under the `additional_chat_templates/` folder.
    If working in offline mode or if internet is down, the method will list jinja template from the local cache - if any.
    """

    if not local_files_only:
        try:
            return [
                entry.path.removeprefix(f"{CHAT_TEMPLATE_DIR}/")
                for entry in list_repo_tree(
                    repo_id=repo_id,
                    revision=revision,
                    path_in_repo=CHAT_TEMPLATE_DIR,
                    recursive=False,
                    token=token,
                )
                if entry.path.endswith(".jinja")
            ]
        except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError):
            raise  # valid errors => do not catch
        except (HfHubHTTPError, OfflineModeIsEnabled, httpx.NetworkError):
            pass  # offline mode, internet down, etc. => try local files

    # check local files
    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id, revision=revision, cache_dir=cache_dir, local_files_only=True
        )
    except LocalEntryNotFoundError:  # No local repo means no local files
        return []
    templates_dir = Path(snapshot_dir, CHAT_TEMPLATE_DIR)
    if not templates_dir.is_dir():
        return []
    return [entry.stem for entry in templates_dir.iterdir() if entry.is_file() and entry.name.endswith(".jinja")]


def define_sagemaker_information():
    try:
        instance_data = httpx.get(os.environ["ECS_CONTAINER_METADATA_URI"]).json()
        dlc_container_used = instance_data["Image"]
        dlc_tag = instance_data["Image"].split(":")[1]
    except Exception:
        dlc_container_used = None
        dlc_tag = None

    sagemaker_params = json.loads(os.getenv("SM_FRAMEWORK_PARAMS", "{}"))
    runs_distributed_training = "sagemaker_distributed_dataparallel_enabled" in sagemaker_params
    training_job_arn = os.getenv("TRAINING_JOB_ARN")
    account_id = training_job_arn.split(":")[4] if training_job_arn is not None else None

    sagemaker_object = {
        "sm_framework": os.getenv("SM_FRAMEWORK_MODULE", None),
        "sm_region": os.getenv("AWS_REGION", None),
        "sm_number_gpu": os.getenv("SM_NUM_GPUS", "0"),
        "sm_number_cpu": os.getenv("SM_NUM_CPUS", "0"),
        "sm_distributed_training": runs_distributed_training,
        "sm_deep_learning_container": dlc_container_used,
        "sm_deep_learning_container_tag": dlc_tag,
        "sm_account_id": account_id,
    }
    return sagemaker_object


def http_user_agent(user_agent: dict | str | None = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{get_torch_version()}"
    if constants.HF_HUB_DISABLE_TELEMETRY:
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


def extract_commit_hash(resolved_file: str | None, commit_hash: str | None) -> str | None:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


def cached_file(
    path_or_repo_id: str | os.PathLike,
    filename: str,
    **kwargs,
) -> str | None:
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
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("google-bert/bert-base-uncased", "pytorch_model.bin")
    ```
    """
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    file = file[0] if file is not None else file
    return file


def cached_files(
    path_or_repo_id: str | os.PathLike,
    filenames: list[str],
    cache_dir: str | os.PathLike | None = None,
    force_download: bool = False,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    subfolder: str = "",
    repo_type: str | None = None,
    user_agent: str | dict[str, str] | None = None,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: str | None = None,
    **deprecated_kwargs,
) -> list[str] | None:
    """
    Tries to locate several files in a local folder and repo, downloads and cache them if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:
            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filenames (`list[str]`):
            The name of all the files to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    Private args:
        _raise_exceptions_for_gated_repo (`bool`):
            if False, do not raise an exception for gated repo error but return None.
        _raise_exceptions_for_missing_entries (`bool`):
            if False, do not raise an exception for missing entries but return None.
        _raise_exceptions_for_connection_errors (`bool`):
            if False, do not raise an exception for connection errors but return None.
        _commit_hash (`str`, *optional*):
            passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
            a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("google-bert/bert-base-uncased", "pytorch_model.bin")
    ```
    """
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    # Add folder to filenames
    full_filenames = [os.path.join(subfolder, file) for file in filenames]

    path_or_repo_id = str(path_or_repo_id)
    existing_files = []
    for filename in full_filenames:
        if os.path.isdir(path_or_repo_id):
            resolved_file = os.path.join(path_or_repo_id, filename)
            if not os.path.isfile(resolved_file):
                if _raise_exceptions_for_missing_entries and filename != os.path.join(subfolder, "config.json"):
                    revision_ = "main" if revision is None else revision
                    raise OSError(
                        f"{path_or_repo_id} does not appear to have a file named {filename}. Checkout "
                        f"'https://huggingface.co/{path_or_repo_id}/tree/{revision_}' for available files."
                    )
                else:
                    continue
            existing_files.append(resolved_file)

    if os.path.isdir(path_or_repo_id):
        return existing_files if existing_files else None

    if cache_dir is None:
        cache_dir = constants.HF_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    existing_files = []
    file_counter = 0
    if _commit_hash is not None and not force_download:
        for filename in full_filenames:
            # If the file is cached under that commit hash, we return it directly.
            resolved_file = try_to_load_from_cache(
                path_or_repo_id, filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
            )
            if resolved_file is not None:
                if resolved_file is not _CACHED_NO_EXIST:
                    file_counter += 1
                    existing_files.append(resolved_file)
                elif not _raise_exceptions_for_missing_entries:
                    file_counter += 1
                else:
                    raise OSError(f"Could not locate {filename} inside {path_or_repo_id}.")

    # Either all the files were found, or some were _CACHED_NO_EXIST but we do not raise for missing entries
    if file_counter == len(full_filenames):
        return existing_files if len(existing_files) > 0 else None

    user_agent = http_user_agent(user_agent)
    # download the files if needed
    try:
        if len(full_filenames) == 1:
            # This is slightly better for only 1 file
            hf_hub_download(
                path_or_repo_id,
                filenames[0],
                subfolder=None if len(subfolder) == 0 else subfolder,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                user_agent=user_agent,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )
        else:
            snapshot_download(
                path_or_repo_id,
                allow_patterns=full_filenames,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                user_agent=user_agent,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )

    except Exception as e:
        # We cannot recover from them
        if isinstance(e, RepositoryNotFoundError) and not isinstance(e, GatedRepoError):
            raise OSError(
                f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
                "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token "
                "having permission to this repo either by logging in with `hf auth login` or by passing "
                "`token=<your_token>`"
            ) from e
        elif isinstance(e, RevisionNotFoundError):
            raise OSError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
                "for this model name. Check the model page at "
                f"'https://huggingface.co/{path_or_repo_id}' for available revisions."
            ) from e
        elif isinstance(e, PermissionError):
            raise OSError(
                f"PermissionError at {e.filename} when downloading {path_or_repo_id}. "
                "Check cache directory permissions. Common causes: 1) another user is downloading the same model (please wait); "
                "2) a previous download was canceled and the lock file needs manual removal."
            ) from e
        elif isinstance(e, ValueError):
            raise OSError(f"{e}") from e

        # Now we try to recover if we can find all files correctly in the cache
        resolved_files = [
            _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision, repo_type)
            for filename in full_filenames
        ]
        if all(file is not None for file in resolved_files):
            return resolved_files

        # Raise based on the flags. Note that we will raise for missing entries at the very end, even when
        # not entering this Except block, as it may also happen when `snapshot_download` does not raise
        if isinstance(e, GatedRepoError):
            if not _raise_exceptions_for_gated_repo:
                return None
            raise OSError(
                "You are trying to access a gated repo.\nMake sure to have access to it at "
                f"https://huggingface.co/{path_or_repo_id}.\n{str(e)}"
            ) from e
        elif isinstance(e, LocalEntryNotFoundError):
            if not _raise_exceptions_for_connection_errors:
                return None
            # Here we only raise if both flags for missing entry and connection errors are True (because it can be raised
            # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
            elif _raise_exceptions_for_missing_entries:
                raise OSError(
                    f"We couldn't connect to '{constants.ENDPOINT}' to load the files, and couldn't find them in the"
                    f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"
                    " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
                ) from e
        # snapshot_download will not raise EntryNotFoundError, but hf_hub_download can. If this is the case, it will be treated
        # later on anyway and re-raised if needed
        elif isinstance(e, HfHubHTTPError) and not isinstance(e, EntryNotFoundError):
            if not _raise_exceptions_for_connection_errors:
                return None
            raise OSError(f"There was a specific connection error when trying to load {path_or_repo_id}:\n{e}") from e
        # Any other Exception type should now be re-raised, in order to provide helpful error messages and break the execution flow
        # (EntryNotFoundError will be treated outside this block and correctly re-raised if needed)
        elif not isinstance(e, EntryNotFoundError):
            raise e

    resolved_files = [
        _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    ]
    # If there are any missing file and the flag is active, raise
    if any(file is None for file in resolved_files) and _raise_exceptions_for_missing_entries:
        missing_entries = [original for original, resolved in zip(full_filenames, resolved_files) if resolved is None]
        # Last escape
        if len(resolved_files) == 1 and missing_entries[0] == os.path.join(subfolder, "config.json"):
            return None
        # Now we raise for missing entries
        revision_ = "main" if revision is None else revision
        msg = (
            f"a file named {missing_entries[0]}" if len(missing_entries) == 1 else f"files named {(*missing_entries,)}"
        )
        raise OSError(
            f"{path_or_repo_id} does not appear to have {msg}. Checkout 'https://huggingface.co/{path_or_repo_id}/tree/{revision_}'"
            " for available files."
        )

    # Remove potential missing entries (we can silently remove them at this point based on the flags)
    resolved_files = [file for file in resolved_files if file is not None]
    # Return `None` if the list is empty, coherent with other Exception when the flag is not active
    resolved_files = None if len(resolved_files) == 0 else resolved_files

    return resolved_files


def has_file(
    path_or_repo: str | os.PathLike,
    filename: str,
    revision: str | None = None,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    *,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    repo_type: str | None = None,
    **deprecated_kwargs,
):
    """
    Checks if a repo contains a given file without downloading it. Works for remote repos and local folders.

    If offline mode is enabled, checks if the file exists in the cache.

    <Tip warning={false}>

    This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist for
    this repo, but will return False for regular connection errors.

    </Tip>
    """
    # If path to local directory, check if the file exists
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    # Else it's a repo => let's check if the file exists in local cache or on the Hub

    # Check if file exists in cache
    # This information might be outdated so it's best to also make a HEAD call (if allowed).
    cached_path = try_to_load_from_cache(
        repo_id=path_or_repo,
        filename=filename,
        revision=revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
    )
    has_file_in_cache = isinstance(cached_path, str)

    # If local_files_only, don't try the HEAD call
    if local_files_only:
        return has_file_in_cache

    # Check if the file exists
    try:
        response = get_session().head(
            hf_hub_url(path_or_repo, filename=filename, revision=revision, repo_type=repo_type),
            headers=build_hf_headers(token=token, user_agent=http_user_agent()),
            follow_redirects=False,
            timeout=10,
        )
    except httpx.ProxyError:
        # Actually raise for those subclasses of ConnectionError
        raise
    except (httpx.ConnectError, httpx.TimeoutException, OfflineModeIsEnabled):
        return has_file_in_cache

    try:
        hf_raise_for_status(response)
        return True
    except GatedRepoError as e:
        logger.error(e)
        raise OSError(
            f"{path_or_repo} is a gated repository. Make sure to request access at "
            f"https://huggingface.co/{path_or_repo} and pass a token having permission to this repo either by "
            "logging in with `hf auth login` or by passing `token=<your_token>`."
        ) from e
    except RepositoryNotFoundError as e:
        logger.error(e)
        raise OSError(f"{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'.") from e
    except RevisionNotFoundError as e:
        logger.error(e)
        raise OSError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
            f"model name. Check the model page at 'https://huggingface.co/{path_or_repo}' for available revisions."
        ) from e
    except EntryNotFoundError:
        return False  # File does not exist
    except HfHubHTTPError:
        # Any authentication/authorization error will be caught here => default to cache
        return has_file_in_cache


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def _get_files_timestamps(self, working_dir: str | os.PathLike):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
        self,
        working_dir: str | os.PathLike,
        repo_id: str,
        files_timestamps: dict[str, float],
        commit_message: str | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
        revision: str | None = None,
        commit_description: str | None = None,
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

        # filter for actual files + folders at the root level
        modified_files = [
            f
            for f in modified_files
            if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))
        ]

        operations = []
        # upload standalone files
        for file in modified_files:
            if os.path.isdir(os.path.join(working_dir, file)):
                # go over individual files of folder
                for f in os.listdir(os.path.join(working_dir, file)):
                    operations.append(
                        CommitOperationAdd(
                            path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f)
                        )
                    )
            else:
                operations.append(
                    CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file)
                )

        if revision is not None and not revision.startswith("refs/pr"):
            try:
                create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
            except HfHubHTTPError as e:
                if e.response.status_code == 403 and create_pr:
                    # If we are creating a PR on a repo we don't have access to, we can't create the branch.
                    # so let's assume the branch already exists. If it's not the case, an error will be raised when
                    # calling `create_commit` below.
                    pass
                else:
                    raise

        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            create_pr=create_pr,
            revision=revision,
        )

    def save_pretrained(self, *args, **kwargs):
        # explicit contract
        raise NotImplementedError(f"{self.__class__.__name__} must implement `save_pretrained` to use `push_to_hub`.")

    def push_to_hub(
        self,
        repo_id: str,
        *,
        # Commit details
        commit_message: str | None = None,
        commit_description: str | None = None,
        # Repo / upload details
        private: bool | None = None,
        token: bool | str | None = None,
        revision: str | None = None,
        create_pr: bool = False,
        # Serialization details
        max_shard_size: int | str | None = "50GB",
        tags: list[str] | None = None,
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated
                when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`).
            tags (`list[str]`, *optional*):
                List of tags to push on the Hub.

        Examples:

        ```python
        from transformers import {object_class}

        {object} = {object_class}.from_pretrained("google-bert/bert-base-cased")

        # Push the {object} to your namespace with the name "my-finetuned-bert".
        {object}.push_to_hub("my-finetuned-bert")

        # Push the {object} to an organization with the name "my-finetuned-bert".
        {object}.push_to_hub("huggingface/my-finetuned-bert")
        ```
        """
        # Create repo if it doesn't exist yet
        repo_id = create_repo(repo_id, private=private, token=token, exist_ok=True).repo_id

        # Load model card or create a new one + eventually tag it
        model_card = create_and_tag_model_card(repo_id, tags, token=token)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save all files.
            self.save_pretrained(tmp_dir, max_shard_size=max_shard_size)

            # Update model card
            model_card.save(os.path.join(tmp_dir, "README.md"))

            # Upload
            return self._upload_modified_files(
                tmp_dir,
                repo_id,
                files_timestamps={},
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )


def convert_file_size_to_int(size: int | str):
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
    local_files_only=False,
    token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    _commit_hash=None,
    **deprecated_kwargs,
):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename) as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub. Try to get everything from cache,
    # or download the files
    cached_filenames = cached_files(
        pretrained_model_name_or_path,
        shard_filenames,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        local_files_only=local_files_only,
        token=token,
        user_agent=user_agent,
        revision=revision,
        subfolder=subfolder,
        _commit_hash=_commit_hash,
    )

    return cached_filenames, sharded_metadata


def create_and_tag_model_card(repo_id: str, tags: list[str] | None = None, token: str | None = None) -> ModelCard:
    """
    Creates or loads an existing model card and tags it.

    Args:
        repo_id (`str`):
            The repo_id where to look for the model card.
        tags (`list[str]`, *optional*):
            The list of tags to add in the model card
        token (`str`, *optional*):
            Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
    """
    try:
        # Check if the model card is present on the remote repo
        model_card = ModelCard.load(repo_id, token=token)
    except EntryNotFoundError:
        # Otherwise create a simple model card from template
        model_description = "This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated."
        card_data = ModelCardData(tags=[] if tags is None else tags, library_name="transformers")
        model_card = ModelCard.from_template(card_data, model_description=model_description)

    if tags is not None:
        # Ensure model_card.data.tags is a list and not None
        if model_card.data.tags is None:
            model_card.data.tags = []
        for model_tag in tags:
            if model_tag not in model_card.data.tags:
                model_card.data.tags.append(model_tag)

    return model_card


class PushInProgress:
    """
    Internal class to keep track of a push in progress (which might contain multiple `Future` jobs).
    """

    def __init__(self, jobs: futures.Future | None = None) -> None:
        self.jobs = [] if jobs is None else jobs

    def is_done(self):
        return all(job.done() for job in self.jobs)

    def wait_until_done(self):
        futures.wait(self.jobs)

    def cancel(self) -> None:
        self.jobs = [
            job
            for job in self.jobs
            # Cancel the job if it wasn't started yet and remove cancelled/done jobs from the list
            if not (job.cancel() or job.done())
        ]
