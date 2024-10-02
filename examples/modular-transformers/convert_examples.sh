#!/bin/bash

# Iterate over each file in the current directory
for file in examples/modular-transformers/modular_*; do
    # Check if it's a regular file
    if [ -f "$file" ]; then
        # Call the Python script with the file name as an argument
        python utils/modular_model_converter.py --files_to_parse "$file"
    fi
done