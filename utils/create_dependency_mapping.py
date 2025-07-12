import ast
from collections import defaultdict


# Function to perform topological sorting
def topological_sort(dependencies: dict) -> list[list[str]]:
    """Given the dependencies graph construct sorted list of list of modular files

    For example, returned list of lists might be:
        [
            ["../modular_llama.py", "../modular_gemma.py"],    # level 0
            ["../modular_llama4.py", "../modular_gemma2.py"],  # level 1
            ["../modular_glm4.py"],                            # level 2
        ]
        which means llama and gemma do not depend on any other modular models, while llama4 and gemma2
        depend on the models in the first list, and glm4 depends on the models in the second and (optionally) in the first list.
    """

    # Nodes are the name of the models to convert (we only add those to the graph)
    nodes = {node.rsplit("modular_", 1)[1].replace(".py", "") for node in dependencies.keys()}
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


# Function to extract class and import info from a file
def extract_classes_and_imports(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if module and (".modeling_" in module or "transformers.models" in module):
                imports.add(module)
    return imports


# Function to map dependencies between classes
def map_dependencies(py_files):
    dependencies = defaultdict(set)
    # First pass: Extract all classes and map to files
    for file_path in py_files:
        # dependencies[file_path].add(None)
        class_to_file = extract_classes_and_imports(file_path)
        for module in class_to_file:
            dependencies[file_path].add(module)
    return dependencies


def find_priority_list(py_files):
    """
    Given a list of modular files, sorts them by topological order. Modular models that DON'T depend on other modular
    models will be higher in the topological order.

    Args:
        py_files: List of paths to the modular files

    Returns:
        Ordered list of lists of files and their dependencies (dict)

        For example, ordered_files might be:
        [
            ["../modular_llama.py", "../modular_gemma.py"],    # level 0
            ["../modular_llama4.py", "../modular_gemma2.py"],  # level 1
            ["../modular_glm4.py"],                            # level 2
        ]
        which means llama and gemma do not depend on any other modular models, while llama4 and gemma2
        depend on the models in the first list, and glm4 depends on the models in the second and (optionally) in the first list.
    """
    dependencies = map_dependencies(py_files)
    ordered_files = topological_sort(dependencies)
    return ordered_files, dependencies
