# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Utilities to dynamically load model and tokenizer from the Hub."""

import importlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from ...file_utils import (
    HF_MODULES_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    cached_path,
    hf_bucket_url,
    is_offline_mode,
)
from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def init_hf_modules():
    """
    Creates the cache directory for modules with an init, and adds it to the Python path.
    """
    # This function has already been executed if HF_MODULES_CACHE already is in the Python path.
    if HF_MODULES_CACHE in sys.path:
        return

    sys.path.append(HF_MODULES_CACHE)
    os.makedirs(HF_MODULES_CACHE, exist_ok=True)
    init_path = Path(HF_MODULES_CACHE) / "__init__.py"
    if not init_path.exists():
        init_path.touch()


def create_dynamic_module(name: Union[str, os.PathLike]):
    """
    Creates a dynamic module in the cache directory for modules.
    """
    init_hf_modules()
    dynamic_module_path = Path(HF_MODULES_CACHE) / name
    # If the parent module does not exist yet, recursively create it.
    if not dynamic_module_path.parent.exists():
        create_dynamic_module(dynamic_module_path.parent)
    os.makedirs(dynamic_module_path, exist_ok=True)
    init_path = dynamic_module_path / "__init__.py"
    if not init_path.exists():
        init_path.touch()


def check_imports(filename):
    """
    Check if the current Python environment contains all the libraries that are imported in a file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Imports of the form `import xxx`
    imports = re.findall("^\s*import\s+(\S+)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from xxx import yyy`
    imports += re.findall("^\s*from\s+(\S+)\s+import", content, flags=re.MULTILINE)
    # Only keep the top-level module
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]

    # Unique-ify and test we got them all
    imports = list(set(imports))
    missing_packages = []
    for imp in imports:
        try:
            importlib.import_module(imp)
        except ImportError:
            missing_packages.append(imp)

    if len(missing_packages) > 0:
        raise ImportError(
            "This modeling file requires the following packages that were not found in your environment: "
            f"{', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`"
        )


def get_class_in_module(class_name, module_path):
    """
    Import a module on the cache directory for modules and extract a class from it.
    """
    module_path = module_path.replace(os.path.sep, ".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_class_from_dynamic_module(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    module_file: str,
    class_name: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Extracts a class from a module file, present in the local folder or repository of a model.

    .. warning::

        Calling this function will execute the code in the module file found locally or downloaded from the Hub. It
        should therefore only be called on trusted repos.

    Args:
        pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
            This can be either:

            - a string, the `model id` of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
              namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
            - a path to a `directory` containing a configuration file saved using the
              :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g., ``./my_model_directory/``.

        module_file (:obj:`str`):
            The name of the module file containing the class to look for.
        class_name (:obj:`str`):
            The name of the class to import in the module.
        cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (:obj:`Dict[str, str]`, `optional`):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
            identifier allowed by git.
        local_files_only (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, will only try to load the tokenizer configuration from local files.

    .. note::

        Passing :obj:`use_auth_token=True` is required when you want to use a private model.


    Returns:
        :obj:`type`: The class, dynamically imported from the module.

    Examples::

        # Download module `modeling.py` from huggingface.co and cache then extract the class `MyBertModel` from this
        # module.
        cls = get_class_from_dynamic_module("sgugger/my-bert-model", "modeling.py", "MyBertModel")
    """
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # Download and cache module_file from the repo `pretrained_model_name_or_path` of grab it if it's a local file.
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
        module_file_or_url = os.path.join(pretrained_model_name_or_path, module_file)
        submodule = "local"
    else:
        module_file_or_url = hf_bucket_url(
            pretrained_model_name_or_path, filename=module_file, revision=revision, mirror=None
        )
        submodule = pretrained_model_name_or_path.replace("/", os.path.sep)

    try:
        # Load from URL or cache if already cached
        resolved_module_file = cached_path(
            module_file_or_url,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
        )

    except EnvironmentError:
        logger.error(f"Could not locate the {module_file} inside {pretrained_model_name_or_path}.")
        raise

    # Check we have all the requirements in our environment
    check_imports(resolved_module_file)

    # Now we move the module inside our cached dynamic modules.
    full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule
    if submodule == "local":
        # We always copy local files (we could hash the file to see if there was a change, and give them the name of
        # that hash, to only copy when there is a modification but it seems overkill for now).
        # The only reason we do the copy is to avoid putting too many folders in sys.path.
        module_name = module_file
        shutil.copy(resolved_module_file, submodule_path / module_file)
    else:
        # The module file will end up being named module_file + the etag. This way we get the benefit of versioning.
        resolved_module_file_name = Path(resolved_module_file).name
        module_name_parts = [module_file.replace(".py", "")] + resolved_module_file_name.split(".")
        module_name = "_".join(module_name_parts) + ".py"
        if not (submodule_path / module_name).exists():
            shutil.copy(resolved_module_file, submodule_path / module_name)

    # And lastly we get the class inside our newly created module
    final_module = os.path.join(full_submodule, module_name.replace(".py", ""))
    return get_class_in_module(class_name, final_module)
