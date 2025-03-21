#!/bin/bash
# fixup.sh: Format all Python files one-by-one to avoid argument list limits

for file in $(git ls-files '*.py'); do
    echo "Formatting $file"
    black "$file"
    isort "$file"
done
