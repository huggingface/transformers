import ast
from collections import defaultdict, deque


# Function to perform topological sorting
def topological_sort(dependencies):
    # Create a graph and in-degree count for each node
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    # Build the graph
    for node, deps in dependencies.items():
        for dep in deps:
            graph[dep].append(node)  # node depends on dep
            in_degree[node] += 1  # increase in-degree of node

    # Add all nodes with zero in-degree to the queue
    zero_in_degree_queue = deque([node for node in dependencies if in_degree[node] == 0])

    sorted_list = []
    # Perform topological sorting
    while zero_in_degree_queue:
        current = zero_in_degree_queue.popleft()
        sorted_list.append(current)

        # For each node that current points to, reduce its in-degree
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)

    # Handle nodes that have no dependencies and were not initially part of the loop
    for node in dependencies:
        if node not in sorted_list:
            sorted_list.append(node)

    return sorted_list


# Function to extract class and import info from a file
def extract_classes_and_imports(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if module and "transformers" in module:
                imports.add(module)
    return imports


# Function to map dependencies between classes
def map_dependencies(py_files):
    dependencies = defaultdict(set)
    # First pass: Extract all classes and map to files
    for file_path in py_files:
        dependencies[file_path].add(None)
        class_to_file = extract_classes_and_imports(file_path)
        for module in class_to_file:
            dependencies[file_path].add(module)
    return dependencies


def find_priority_list(py_files):
    dependencies = map_dependencies(py_files)
    ordered_classes = topological_sort(dependencies)
    return ordered_classes[::-1]
