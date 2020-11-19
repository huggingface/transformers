from .dependency_versions_table import deps
from .file_utils import is_tokenizers_available
from .utils.versions import require_version_core


# define which module versions we always want to check at run time (usually the ones defined in `install_requires` in setup.py)
pkgs_to_check_at_runtime = "python tokenizers tqdm regex numpy".split()
for pkg in pkgs_to_check_at_runtime:
    if pkg in deps:
        if pkg == "tokenizers" and not is_tokenizers_available():
            continue  # not required, check version only if installed
        require_version_core(deps[pkg])
    else:
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")
