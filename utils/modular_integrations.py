import os

import libcst as cst


# Files from external libraries that should not be tracked
# E.g. for habana, we don't want to track the dependencies from `modeling_all_models.py` as it is not part of the transformers library
EXCLUDED_EXTERNAL_FILES = {
    "habana": [{"name": "modeling_all_models", "type": "modeling"}],
}


def convert_relative_import_to_absolute(
    import_node: cst.ImportFrom,
    file_path: str,
    package_name: str | None = "transformers",
) -> cst.ImportFrom:
    """
    Convert a relative libcst.ImportFrom node into an absolute one,
    using the file path and package name.

    Args:
        import_node: A relative import node (e.g. `from ..utils import helper`)
        file_path: Path to the file containing the import (can be absolute or relative)
        package_name: The top-level package name (e.g. 'myproject')

    Returns:
        A new ImportFrom node with the absolute import path
    """
    if not (import_node.relative and len(import_node.relative) > 0):
        return import_node  # Already absolute

    file_path = os.path.abspath(file_path)
    rel_level = len(import_node.relative)

    # Strip file extension and split into parts
    file_path_no_ext = file_path[:-3] if file_path.endswith(".py") else file_path
    file_parts = file_path_no_ext.split(os.path.sep)

    # Ensure the file path includes the package name
    if package_name not in file_parts:
        raise ValueError(f"Package name '{package_name}' not found in file path '{file_path}'")

    # Slice file_parts starting from the package name
    pkg_index = file_parts.index(package_name)
    module_parts = file_parts[pkg_index + 1 :]  # e.g. ['module', 'submodule', 'foo']
    if len(module_parts) < rel_level:
        raise ValueError(f"Relative import level ({rel_level}) goes beyond package root.")

    base_parts = module_parts[:-rel_level]

    # Flatten the module being imported (if any)
    def flatten_module(module: cst.BaseExpression | None) -> list[str]:
        if not module:
            return []
        if isinstance(module, cst.Name):
            return [module.value]
        elif isinstance(module, cst.Attribute):
            parts = []
            while isinstance(module, cst.Attribute):
                parts.insert(0, module.attr.value)
                module = module.value
            if isinstance(module, cst.Name):
                parts.insert(0, module.value)
            return parts
        return []

    import_parts = flatten_module(import_node.module)

    # Combine to get the full absolute import path
    full_parts = [package_name] + base_parts + import_parts

    # Handle special case where the import comes from a namespace package (e.g. optimum with `optimum.habana`, `optimum.intel` instead of `src.optimum`)
    if package_name != "transformers" and file_parts[pkg_index - 1] != "src":
        full_parts = [file_parts[pkg_index - 1]] + full_parts

    # Build the dotted module path
    dotted_module: cst.BaseExpression | None = None
    for part in full_parts:
        name = cst.Name(part)
        dotted_module = name if dotted_module is None else cst.Attribute(value=dotted_module, attr=name)

    # Return a new ImportFrom node with absolute import
    return import_node.with_changes(module=dotted_module, relative=[])


def convert_to_relative_import(import_node: cst.ImportFrom, file_path: str, package_name: str) -> cst.ImportFrom:
    """
    Convert an absolute import to a relative one if it belongs to `package_name`.

    Parameters:
    - node: The ImportFrom node to possibly transform.
    - file_path: Absolute path to the file containing the import (e.g., '/path/to/mypackage/foo/bar.py').
    - package_name: The top-level package name (e.g., 'mypackage').

    Returns:
    - A possibly modified ImportFrom node.
    """
    if import_node.relative:
        return import_node  # Already relative import

    # Extract module name string from ImportFrom
    def get_module_name(module):
        if isinstance(module, cst.Name):
            return module.value, [module.value]
        elif isinstance(module, cst.Attribute):
            parts = []
            while isinstance(module, cst.Attribute):
                parts.append(module.attr.value)
                module = module.value
            if isinstance(module, cst.Name):
                parts.append(module.value)
            parts.reverse()
            return ".".join(parts), parts
        return "", None

    module_name, submodule_list = get_module_name(import_node.module)

    # Check if it's from the target package
    if (
        not (module_name.startswith(package_name + ".") or module_name.startswith("optimum." + package_name + "."))
        and module_name != package_name
    ):
        return import_node  # Not from target package

    # Locate the package root inside the file path
    norm_file_path = os.path.normpath(file_path)
    parts = norm_file_path.split(os.sep)

    try:
        pkg_index = parts.index(package_name)
    except ValueError:
        # Package name not found in path — assume we can't resolve relative depth
        return import_node

    # Depth is how many directories after the package name before the current file
    depth = len(parts) - pkg_index - 1  # exclude the .py file itself
    for i, submodule in enumerate(parts[pkg_index + 1 :]):
        if submodule == submodule_list[2 + i]:
            depth -= 1
        else:
            break

    # Create the correct number of dots
    relative = [cst.Dot()] * depth if depth > 0 else [cst.Dot()]

    # Strip package prefix from import module path
    if module_name.startswith("optimum." + package_name + "."):
        stripped_name = module_name[len("optimum." + package_name) :].lstrip(".")
    else:
        stripped_name = module_name[len(package_name) :].lstrip(".")

    # Build new module node
    if stripped_name == "":
        new_module = None
    else:
        name_parts = stripped_name.split(".")[i:]
        new_module = cst.Name(name_parts[0])
        for part in name_parts[1:]:
            new_module = cst.Attribute(value=new_module, attr=cst.Name(part))

    return import_node.with_changes(module=new_module, relative=relative)


class AbsoluteImportTransformer(cst.CSTTransformer):
    def __init__(self, relative_path: str, source_library: str):
        super().__init__()
        self.relative_path = relative_path
        self.source_library = source_library

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.ImportFrom:
        return convert_relative_import_to_absolute(
            import_node=updated_node, file_path=self.relative_path, package_name=self.source_library
        )


class RelativeImportTransformer(cst.CSTTransformer):
    def __init__(self, relative_path: str, source_library: str):
        super().__init__()
        self.relative_path = relative_path
        self.source_library = source_library

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.ImportFrom:
        return convert_to_relative_import(updated_node, self.relative_path, self.source_library)
