from .dependency_versions_table import deps
from .utils.versions import require_version_core


# define which module versions we always want to check at run time (usually the ones defined in `install_requires` in setup.py)
pkgs_to_check_at_runtime = "python tokenizers tqdm regex".split()
for k in pkgs_to_check_at_runtime:
    if k in deps:
        require_version_core(deps[k])
    else:
        raise ValueError(f"can't find {k} in {deps.keys()}, check dependency_versions_table.py")
