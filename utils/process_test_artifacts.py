import json
import os
import math

MAX_PARALLEL_NODES = 8
AVERAGE_TESTS_PER_NODES = 5
def count_lines(filepath):
    """Count the number of lines in a file."""
    try:
        with open(filepath, 'r') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def compute_parallel_nodes(line_count, max_tests_per_node=10):
    """Compute the number of parallel nodes required."""
    num_nodes = math.ceil(line_count / AVERAGE_TESTS_PER_NODES)
    if line_count < 4:
        return 1
    return min(MAX_PARALLEL_NODES, num_nodes)

def process_artifacts(input_file, output_file):
    # Read the JSON data from the input file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Process items and build the new JSON structure
    transformed_data = {}
    for item in data.get('items', []):
        if "test_list" in item:
            key = os.path.splitext(os.path.basename(item['path']))[0]
            transformed_data[key] = item['url']
            parallel_key = key.split("test")[0]
            file_path = os.path.join('test_preparation', f'{key}.txt')
            line_count = count_lines(file_path)
            transformed_data[parallel_key] =  compute_parallel_nodes(line_count)

    # Remove the "generated_config" key if it exists
    if 'generated_config' in transformed_data:
        del transformed_data['generated_config']

    # Write the transformed data to the output file
    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, indent=2)

if __name__ == '__main__':
    input_file = 'test_preparation/artifacts.json'
    output_file = 'test_preparation/transformed_artifacts.json'
    process_artifacts(input_file, output_file)