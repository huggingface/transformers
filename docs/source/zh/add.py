import os

def find_copyright_files(directory, output_file):
    with open(output_file, 'w') as out_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding ='utf-8') as f:
                        first_line = f.readline()
                        if "Copyright 2022 The HuggingFace Team" in first_line:
                            out_file.write(file_path + '\n')
                except:
                    print(f "Cannot open file {file_path}")

directory = "./docs/source/zh/" # Replace with your directory
output_file = "paths.txt"
find_copyright_files(directory, output_file)
