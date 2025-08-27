import ast
import re
from collections import defaultdict


# Function to perform topological sorting
def topological_sort(dependencies: dict) -> list[list[str]]:
    """Given the dependencies graph, construct a sorted list of list of modular files.

    Examples:

        The returned list of lists might be:
        [
            ["../modular_mistral.py", "../modular_gemma.py"],  # level 0
            ["../modular_llama4.py", "../modular_gemma2.py"],  # level 1
            ["../modular_glm4.py"],                            # level 2
        ]
        which means mistral and gemma do not depend on any other modular models, while llama4 and gemma2
        depend on the models in the first list, and glm4 depends on the models in the second and (optionally) in the first list.
    """

    # Nodes are the name of the models to convert (we only add those to the graph)
    nodes = {node.rsplit("modular_", 1)[1].replace(".py", "") for node in dependencies}
    # This will be a graph from models to convert, to models to convert that should be converted before (as they are a dependency)
    graph = {}
    name_mapping = {}
    for node, deps in dependencies.items():
        node_name = node.rsplit("modular_", 1)[1].replace(".py", "")
        dep_names = {dep.split(".")[-2] for dep in deps}
        dependencies = {dep for dep in dep_names if dep in nodes and dep != node_name}
        graph[node_name] = dependencies
        name_mapping[node_name] = node

    sorting_list = []
    while len(graph) > 0:
        # Find the nodes with 0 out-degree
        leaf_nodes = {node for node in graph if len(graph[node]) == 0}
        # Add them to the list as next level
        sorting_list.append([name_mapping[node] for node in leaf_nodes])
        # Remove the leafs from the graph (and from the deps of other nodes)
        graph = {node: deps - leaf_nodes for node, deps in graph.items() if node not in leaf_nodes}

    return sorting_list


# All the model file types that may be imported in modular files
ALL_FILE_TYPES = (
    "modeling",
    "configuration",
    "tokenization",
    "processing",
    "image_processing",
    "video_processing",
    "feature_extraction",
)


def is_model_import(module: str) -> bool:
    """Check whether `module` is a model import or not."""
    patterns = "|".join(ALL_FILE_TYPES)
    regex = rf"(\w+)\.(?:{patterns})_(\w+)"
    match_object = re.search(regex, module)
    if match_object is not None:
        model_name = match_object.group(1)
        if model_name in match_object.group(2) and model_name != "auto":
            return True
    return False


def extract_model_imports_from_file(file_path):
    """From a python file `file_path`, extract the model-specific imports (the imports related to any model file in
    Transformers)"""
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if is_model_import(node.module):
                imports.add(node.module)
    return imports


def find_priority_list(modular_files: list[str]) -> tuple[list[list[str]], dict[str, set]]:
    """
    Given a list of modular files, sorts them by topological order. Modular models that DON'T depend on other modular
    models will be lower in the topological order.

    Args:
        modular_files (`list[str]`):
            List of paths to the modular files.

    Returns:
        A tuple `ordered_files` and `dependencies`.

        `ordered_file` is a list of lists consisting of the models at each level of the dependency graph. For example,
        it might be:
        [
            ["../modular_mistral.py", "../modular_gemma.py"],  # level 0
            ["../modular_llama4.py", "../modular_gemma2.py"],  # level 1
            ["../modular_glm4.py"],                            # level 2
        ]
        which means mistral and gemma do not depend on any other modular models, while llama4 and gemma2 depend on the
        models in the first list, and glm4 depends on the models in the second and (optionally) in the first list.

        `dependencies` is a dictionary mapping each modular file to the models on which it relies (the models that are
        imported in order to use inheritance).
    """
    dependencies = defaultdict(set)
    for file_path in modular_files:
        dependencies[file_path].update(extract_model_imports_from_file(file_path))
    ordered_files = topological_sort(dependencies)
    return ordered_files, dependencies
