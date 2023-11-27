import argparse
import json
import os
import ast

def process_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            namespace = {}

            exec(content, namespace)

            model_config = namespace.get("model")
            data_config = namespace.get("data")
            short_size = data_config.get('train', {}).get('short_size')
            config = model_config.get('backbone', {}).get('config')
            return short_size, config
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

def process_folder(folder_path):
    results = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                short_size, config = process_file(file_path)
                if short_size is not None and config is not None:
                    results[file] = {'short_size': short_size, 'config': config}
    return results

def main():
    parser = argparse.ArgumentParser(description='Process Python files in a given folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing Python files')
    args = parser.parse_args()

    folder_path = args.folder_path
    results = process_folder(folder_path)
    with open("/Users/eaxxkra/Downloads/fast_model_configs.json",'w') as f:
        json.dump(results,f,indent=4)

if __name__ == '__main__':
    main()
