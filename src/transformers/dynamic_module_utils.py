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
"""Utilities to dynamically load objects from the Hub."""

import ast
import filecmp
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import re
import shutil
import signal
import sys
import threading
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Union

from huggingface_hub import try_to_load_from_cache
from packaging import version

from .utils import (
    HF_MODULES_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    cached_file,
    extract_commit_hash,
    is_offline_mode,
    logging,
)
from .utils.import_utils import VersionComparison, split_package_version


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
_HF_REMOTE_CODE_LOCK = threading.Lock()


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
        importlib.invalidate_caches()


def create_dynamic_module(name: Union[str, os.PathLike]) -> None:
    """
    Creates a dynamic module in the cache directory for modules.

    Args:
        name (`str` or `os.PathLike`):
            The name of the dynamic module to create.
    """
    init_hf_modules()
    dynamic_module_path = (Path(HF_MODULES_CACHE) / name).resolve()
    # If the parent module does not exist yet, recursively create it.
    if not dynamic_module_path.parent.exists():
        create_dynamic_module(dynamic_module_path.parent)
    os.makedirs(dynamic_module_path, exist_ok=True)
    init_path = dynamic_module_path / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        # It is extremely important to invalidate the cache when we change stuff in those modules, or users end up
        # with errors about module that do not exist. Same for all other `invalidate_caches` in this file.
        importlib.invalidate_caches()


def get_relative_imports(module_file: Union[str, os.PathLike]) -> list[str]:
    """
    Get the list of modules that are relatively imported in a module file.

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.

    Returns:
        `list[str]`: The list of relative imports in the module.
    """
    with open(module_file, encoding="utf-8") as f:
        content = f.read()

    # Imports of the form `import .xxx`
    relative_imports = re.findall(r"^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from .xxx import yyy`
    relative_imports += re.findall(r"^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE)
    # Unique-ify
    return list(set(relative_imports))


def get_relative_import_files(module_file: Union[str, os.PathLike]) -> list[str]:
    """
    Get the list of all files that are needed for a given module. Note that this function recurses through the relative
    imports (if a imports b and b imports c, it will return module files for b and c).

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.

    Returns:
        `list[str]`: The list of all relative imports a given module needs (recursively), which will give us the list
        of module files a given module needs.
    """
    no_change = False
    files_to_check = [module_file]
    all_relative_imports = []

    # Let's recurse through all relative imports
    while not no_change:
        new_imports = []
        for f in files_to_check:
            new_imports.extend(get_relative_imports(f))

        module_path = Path(module_file).parent
        new_import_files = [str(module_path / m) for m in new_imports]
        new_import_files = [f for f in new_import_files if f not in all_relative_imports]
        files_to_check = [f"{f}.py" for f in new_import_files]

        no_change = len(new_import_files) == 0
        all_relative_imports.extend(files_to_check)

    return all_relative_imports


def get_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """
    Extracts all the libraries (not relative imports this time) that are imported in a file.

    Args:
        filename (`str` or `os.PathLike`): The module file to inspect.

    Returns:
        `list[str]`: The list of all packages required to use the input module.
    """
    with open(filename, encoding="utf-8") as f:
        content = f.read()
    imported_modules = set()

    import transformers.utils

    def recursive_look_for_imports(node):
        if isinstance(node, ast.Try):
            return  # Don't recurse into Try blocks and ignore imports in them
        elif isinstance(node, ast.If):
            test = node.test
            for condition_node in ast.walk(test):
                if isinstance(condition_node, ast.Call):
                    check_function = getattr(condition_node.func, "id", "")
                    if (
                        check_function.endswith("available")
                        and check_function.startswith("is_flash_attn")
                        or hasattr(transformers.utils.import_utils, check_function)
                    ):
                        # Don't recurse into "if flash_attn_available()" or any "if library_available" blocks
                        # that appears in `transformers.utils.import_utils` and ignore imports in them
                        return
        elif isinstance(node, ast.Import):
            # Handle 'import x' statements
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module:
                    imported_modules.add(top_module)
        elif isinstance(node, ast.ImportFrom):
            # Handle 'from x import y' statements, ignoring relative imports
            if node.level == 0 and node.module:
                top_module = node.module.split(".")[0]
                if top_module:
                    imported_modules.add(top_module)

        # Recursively visit all children
        for child in ast.iter_child_nodes(node):
            recursive_look_for_imports(child)

    tree = ast.parse(content)
    recursive_look_for_imports(tree)

    return sorted(imported_modules)


def check_imports(filename: Union[str, os.PathLike]) -> list[str]:
    """
    Check if the current Python environment contains all the libraries that are imported in a file. Will raise if a
    library is missing.

    Args:
        filename (`str` or `os.PathLike`): The module file to check.

    Returns:
        `list[str]`: The list of relative imports in the file.
    """
    imports = get_imports(filename)
    missing_packages = []
    for imp in imports:
        try:
            importlib.import_module(imp)
        except ImportError as exception:
            logger.warning(f"Encountered exception while importing {imp}: {exception}")
            # Some packages can fail with an ImportError because of a dependency issue.
            # This check avoids hiding such errors.
            # See https://github.com/huggingface/transformers/issues/33604
            if "No module named" in str(exception):
                missing_packages.append(imp)
            else:
                raise

    if len(missing_packages) > 0:
        raise ImportError(
            "This modeling file requires the following packages that were not found in your environment: "
            f"{', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`"
        )

    return get_relative_imports(filename)


def get_class_in_module(
    class_name: str,
    module_path: Union[str, os.PathLike],
    *,
    force_reload: bool = False,
) -> type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.
        force_reload (`bool`, *optional*, defaults to `False`):
            Whether to reload the dynamic module from file if it already exists in `sys.modules`.
            Otherwise, the module is only reloaded if the file has changed.

    Returns:
        `typing.Type`: The class looked for.
    """
    name = os.path.normpath(module_path)
    if name.endswith(".py"):
        name = name[:-3]
    name = name.replace(os.path.sep, ".")
    module_file: Path = Path(HF_MODULES_CACHE) / module_path
    with _HF_REMOTE_CODE_LOCK:
        if force_reload:
            sys.modules.pop(name, None)
            importlib.invalidate_caches()
        cached_module: Optional[ModuleType] = sys.modules.get(name)
        module_spec = importlib.util.spec_from_file_location(name, location=module_file)

        # Hash the module file and all its relative imports to check if we need to reload it
        module_files: list[Path] = [module_file] + sorted(map(Path, get_relative_import_files(module_file)))
        module_hash: str = hashlib.sha256(b"".join(bytes(f) + f.read_bytes() for f in module_files)).hexdigest()

        module: ModuleType
        if cached_module is None:
            module = importlib.util.module_from_spec(module_spec)
            # insert it into sys.modules before any loading begins
            sys.modules[name] = module
        else:
            module = cached_module
        # reload in both cases, unless the module is already imported and the hash hits
        if getattr(module, "__transformers_module_hash__", "") != module_hash:
            module_spec.loader.exec_module(module)
            module.__transformers_module_hash__ = module_hash
        return getattr(module, class_name)


def get_cached_module_file(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    module_file: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> str:
    """
    Prepares Downloads a module from a local folder or a distant repo and returns its path inside the cached
    Transformers module.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        module_file (`str`):
            The name of the module file containing the class to look for.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `str`: The path to the module inside the cache.
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # Download and cache module_file from the repo `pretrained_model_name_or_path` of grab it if it's a local file.
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:
        submodule = os.path.basename(pretrained_model_name_or_path)
    else:
        submodule = pretrained_model_name_or_path.replace("/", os.path.sep)
        cached_module = try_to_load_from_cache(
            pretrained_model_name_or_path, module_file, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
        )

    new_files = []
    try:
        # Load from URL or cache if already cached
        resolved_module_file = cached_file(
            pretrained_model_name_or_path,
            module_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            repo_type=repo_type,
            _commit_hash=_commit_hash,
        )
        if not is_local and cached_module != resolved_module_file:
            new_files.append(module_file)

    except OSError:
        logger.info(f"Could not locate the {module_file} inside {pretrained_model_name_or_path}.")
        raise

    # Check we have all the requirements in our environment
    modules_needed = check_imports(resolved_module_file)

    # Now we move the module inside our cached dynamic modules.
    full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule
    if submodule == os.path.basename(pretrained_model_name_or_path):
        # We copy local files to avoid putting too many folders in sys.path. This copy is done when the file is new or
        # has changed since last copy.
        if not (submodule_path / module_file).exists() or not filecmp.cmp(
            resolved_module_file, str(submodule_path / module_file)
        ):
            (submodule_path / module_file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()
        for module_needed in modules_needed:
            module_needed = Path(module_file).parent / f"{module_needed}.py"
            module_needed_file = os.path.join(pretrained_model_name_or_path, module_needed)
            if not (submodule_path / module_needed).exists() or not filecmp.cmp(
                module_needed_file, str(submodule_path / module_needed)
            ):
                shutil.copy(module_needed_file, submodule_path / module_needed)
                importlib.invalidate_caches()
    else:
        # Get the commit hash
        commit_hash = extract_commit_hash(resolved_module_file, _commit_hash)

        # The module file will end up being placed in a subfolder with the git hash of the repo. This way we get the
        # benefit of versioning.
        submodule_path = submodule_path / commit_hash
        full_submodule = full_submodule + os.path.sep + commit_hash
        full_submodule_module_file_path = os.path.join(full_submodule, module_file)
        create_dynamic_module(Path(full_submodule_module_file_path).parent)

        if not (submodule_path / module_file).exists():
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()
        # Make sure we also have every file with relative
        for module_needed in modules_needed:
            if not (submodule_path / f"{module_needed}.py").exists():
                get_cached_module_file(
                    pretrained_model_name_or_path,
                    f"{module_needed}.py",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                )
                new_files.append(f"{module_needed}.py")

    if len(new_files) > 0 and revision is None:
        new_files = "\n".join([f"- {f}" for f in new_files])
        repo_type_str = "" if repo_type is None else f"{repo_type}s/"
        url = f"https://huggingface.co/{repo_type_str}{pretrained_model_name_or_path}"
        logger.warning(
            f"A new version of the following files was downloaded from {url}:\n{new_files}"
            "\n. Make sure to double-check they do not contain any added malicious code. To avoid downloading new "
            "versions of the code file, you can pin a revision."
        )

    return os.path.join(full_submodule, module_file)


def get_class_from_dynamic_module(
    class_reference: str,
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    code_revision: Optional[str] = None,
    **kwargs,
) -> type:
    """
    Extracts a class from a module file, present in the local folder or repository of a model.

    <Tip warning={true}>

    Calling this function will execute the code in the module file found locally or downloaded from the Hub. It should
    therefore only be called on trusted repos.

    </Tip>



    Args:
        class_reference (`str`):
            The full name of the class to load, including its module and optionally its repo.
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

            This is used when `class_reference` does not specify another repo.
        module_file (`str`):
            The name of the module file containing the class to look for.
        class_name (`str`):
            The name of the class to import in the module.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).
        code_revision (`str`, *optional*, defaults to `"main"`):
            The specific revision to use for the code on the Hub, if the code leaves in a different repository than the
            rest of the model. It can be a branch name, a tag name, or a commit id, since we use a git-based system for
            storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `typing.Type`: The class, dynamically imported from the module.

    Examples:

    ```python
    # Download module `modeling.py` from huggingface.co and cache then extract the class `MyBertModel` from this
    # module.
    cls = get_class_from_dynamic_module("modeling.MyBertModel", "sgugger/my-bert-model")

    # Download module `modeling.py` from a given repo and cache then extract the class `MyBertModel` from this
    # module.
    cls = get_class_from_dynamic_module("sgugger/my-bert-model--modeling.MyBertModel", "sgugger/another-bert-model")
    ```"""
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # Catch the name of the repo if it's specified in `class_reference`
    if "--" in class_reference:
        repo_id, class_reference = class_reference.split("--")
    else:
        repo_id = pretrained_model_name_or_path
    module_file, class_name = class_reference.split(".")

    if code_revision is None and pretrained_model_name_or_path == repo_id:
        code_revision = revision
    # And lastly we get the class inside our newly created module
    final_module = get_cached_module_file(
        repo_id,
        module_file + ".py",
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=code_revision,
        local_files_only=local_files_only,
        repo_type=repo_type,
    )
    return get_class_in_module(class_name, final_module, force_reload=force_download)


def custom_object_save(obj: Any, folder: Union[str, os.PathLike], config: Optional[dict] = None) -> list[str]:
    """
    Save the modeling files corresponding to a custom model/configuration/tokenizer etc. in a given folder. Optionally
    adds the proper fields in a config.

    Args:
        obj (`Any`): The object for which to save the module files.
        folder (`str` or `os.PathLike`): The folder where to save.
        config (`PretrainedConfig` or dictionary, `optional`):
            A config in which to register the auto_map corresponding to this custom object.

    Returns:
        `list[str]`: The list of files saved.
    """
    if obj.__module__ == "__main__":
        logger.warning(
            f"We can't save the code defining {obj} in {folder} as it's been defined in __main__. You should put "
            "this code in a separate module so we can include it in the saved folder and make it easier to share via "
            "the Hub."
        )
        return

    def _set_auto_map_in_config(_config):
        module_name = obj.__class__.__module__
        last_module = module_name.split(".")[-1]
        full_name = f"{last_module}.{obj.__class__.__name__}"
        # Special handling for tokenizers
        if "Tokenizer" in full_name:
            slow_tokenizer_class = None
            fast_tokenizer_class = None
            if obj.__class__.__name__.endswith("Fast"):
                # Fast tokenizer: we have the fast tokenizer class and we may have the slow one has an attribute.
                fast_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"
                if getattr(obj, "slow_tokenizer_class", None) is not None:
                    slow_tokenizer = getattr(obj, "slow_tokenizer_class")
                    slow_tok_module_name = slow_tokenizer.__module__
                    last_slow_tok_module = slow_tok_module_name.split(".")[-1]
                    slow_tokenizer_class = f"{last_slow_tok_module}.{slow_tokenizer.__name__}"
            else:
                # Slow tokenizer: no way to have the fast class
                slow_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"

            full_name = (slow_tokenizer_class, fast_tokenizer_class)

        if isinstance(_config, dict):
            auto_map = _config.get("auto_map", {})
            auto_map[obj._auto_class] = full_name
            _config["auto_map"] = auto_map
        elif getattr(_config, "auto_map", None) is not None:
            _config.auto_map[obj._auto_class] = full_name
        else:
            _config.auto_map = {obj._auto_class: full_name}

    # Add object class to the config auto_map
    if isinstance(config, (list, tuple)):
        for cfg in config:
            _set_auto_map_in_config(cfg)
    elif config is not None:
        _set_auto_map_in_config(config)

    result = []
    # Copy module file to the output folder.
    object_file = sys.modules[obj.__module__].__file__
    dest_file = Path(folder) / (Path(object_file).name)
    shutil.copy(object_file, dest_file)
    result.append(dest_file)

    # Gather all relative imports recursively and make sure they are copied as well.
    for needed_file in get_relative_import_files(object_file):
        dest_file = Path(folder) / (Path(needed_file).name)
        shutil.copy(needed_file, dest_file)
        result.append(dest_file)

    return result


def _raise_timeout_error(signum, frame):
    raise ValueError(
        "Loading this model requires you to execute custom code contained in the model repository on your local "
        "machine. Please set the option `trust_remote_code=True` to permit loading of this model."
    )


TIME_OUT_REMOTE_CODE = 15


def resolve_trust_remote_code(
    trust_remote_code, model_name, has_local_code, has_remote_code, error_message=None, upstream_repo=None
):
    """
    Resolves the `trust_remote_code` argument. If there is remote code to be loaded, the user must opt-in to loading
    it.

    Args:
        trust_remote_code (`bool` or `None`):
            User-defined `trust_remote_code` value.
        model_name (`str`):
            The name of the model repository in huggingface.co.
        has_local_code (`bool`):
            Whether the model has local code.
        has_remote_code (`bool`):
            Whether the model has remote code.
        error_message (`str`, *optional*):
            Custom error message to display if there is remote code to load and the user didn't opt-in. If unset, the error
            message will be regarding loading a model with custom code.

    Returns:
        The resolved `trust_remote_code` value.
    """
    if error_message is None:
        if upstream_repo is not None:
            error_message = (
                f"The repository {model_name} references custom code contained in {upstream_repo} which "
                f"must be executed to correctly load the model. You can inspect the repository "
                f"content at https://hf.co/{upstream_repo} .\n"
            )
        elif os.path.isdir(model_name):
            error_message = (
                f"The repository {model_name} contains custom code which must be executed "
                f"to correctly load the model. You can inspect the repository "
                f"content at {os.path.abspath(model_name)} .\n"
            )
        else:
            error_message = (
                f"The repository {model_name} contains custom code which must be executed "
                f"to correctly load the model. You can inspect the repository "
                f"content at https://hf.co/{model_name} .\n"
            )

    if trust_remote_code is None:
        if has_local_code:
            trust_remote_code = False
        elif has_remote_code and TIME_OUT_REMOTE_CODE > 0:
            prev_sig_handler = None
            try:
                prev_sig_handler = signal.signal(signal.SIGALRM, _raise_timeout_error)
                signal.alarm(TIME_OUT_REMOTE_CODE)
                while trust_remote_code is None:
                    answer = input(
                        f"{error_message} You can inspect the repository content at https://hf.co/{model_name}.\n"
                        f"You can avoid this prompt in future by passing the argument `trust_remote_code=True`.\n\n"
                        f"Do you wish to run the custom code? [y/N] "
                    )
                    if answer.lower() in ["yes", "y", "1"]:
                        trust_remote_code = True
                    elif answer.lower() in ["no", "n", "0", ""]:
                        trust_remote_code = False
                signal.alarm(0)
            except Exception:
                # OS which does not support signal.SIGALRM
                raise ValueError(
                    f"{error_message} You can inspect the repository content at https://hf.co/{model_name}.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )
            finally:
                if prev_sig_handler is not None:
                    signal.signal(signal.SIGALRM, prev_sig_handler)
                    signal.alarm(0)
        elif has_remote_code:
            # For the CI which puts the timeout at 0
            _raise_timeout_error(None, None)

    if has_remote_code and not has_local_code and not trust_remote_code:
        raise ValueError(
            f"{error_message} You can inspect the repository content at https://hf.co/{model_name}.\n"
            f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
        )

    return trust_remote_code


def check_python_requirements(path_or_repo_id, requirements_file="requirements.txt", **kwargs):
    """
    Tries to locate `requirements_file` in a local folder or repo, and confirms that the environment has all the
    python dependencies installed.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:
            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        kwargs (`dict[str, Any]`, *optional*):
            Additional arguments to pass to `cached_file`.
    """
    failed = []  # error messages regarding requirements
    try:
        requirements = cached_file(path_or_repo_id=path_or_repo_id, filename=requirements_file, **kwargs)
        with open(requirements, "r") as f:
            requirements = f.readlines()

        for requirement in requirements:
            requirement = requirement.strip()
            if not requirement or requirement.startswith("#"):  # skip empty lines and comments
                continue

            try:
                # e.g. "torch>2.6.0" -> "torch", ">", "2.6.0"
                package_name, delimiter, version_number = split_package_version(requirement)
            except ValueError:  # e.g. "torch", as opposed to "torch>2.6.0"
                package_name = requirement
                delimiter, version_number = None, None

            try:
                local_package_version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                failed.append(f"{requirement} (installed: None)")
                continue

            if delimiter is not None and version_number is not None:
                is_satisfied = VersionComparison.from_string(delimiter)(
                    version.parse(local_package_version), version.parse(version_number)
                )
            else:
                is_satisfied = True

            if not is_satisfied:
                failed.append(f"{requirement} (installed: {local_package_version})")

    except OSError:  # no requirements.txt
        pass

    if failed:
        raise ImportError(
            f"Missing requirements in your local environment for `{path_or_repo_id}`:\n" + "\n".join(failed)
        )
