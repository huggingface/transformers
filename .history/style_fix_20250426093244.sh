#!/bin/bash
# Script to fix common code style issues

echo "Fixing code style issues..."

# Create a temporary script to fix the issues
cat > /tmp/fix_style.py << 'EOF'
import os
import sys
import re

def fix_file(filepath):
    print(f"Processing {filepath}")
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Fix trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Fix blank lines containing whitespace
    content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Ensure files end with a newline
    if not content.endswith('\n'):
        content += '\n'
    
    # Fix f-strings without placeholders in configuration_hindi_causal_lm.py
    if 'configuration_hindi_causal_lm.py' in filepath:
        content = content.replace(
            'f"Config positional_encoding_type is \'rope\', but the adapted modeling code uses standard embeddings "',
            '"Config positional_encoding_type is \'rope\', but the adapted modeling code uses standard embeddings "'
        )
        content = content.replace(
            'f"to match the original training script. RoPE is not implemented."',
            '"to match the original training script. RoPE is not implemented."'
        )

    # Write the fixed content back to the file
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)

# Paths to fix
paths = [
    "src/transformers/__init__.py",
    "src/transformers/models/__init__.py",
    "src/transformers/models/auto/configuration_auto.py",
    "src/transformers/models/auto/modeling_auto.py",
    "src/transformers/models/auto/tokenization_auto.py",
    "src/transformers/models/hindi_causal_lm/__init__.py",
    "src/transformers/models/hindi_causal_lm/configuration_hindi_causal_lm.py",
    "src/transformers/models/hindi_causal_lm/modeling_hindi_causal_lm.py", 
    "src/transformers/models/hindi_causal_lm/tokenization_hindi_causal_lm.py",
    "tests/models/hindi_causal_lm/test_modeling_hindi_causal_lm.py"
]

for path in paths:
    if os.path.exists(path):
        fix_file(path)
    else:
        print(f"Warning: {path} not found")

print("Done fixing basic style issues")
EOF

python /tmp/fix_style.py

# Now run the ruff linter with --fix to handle more complex issues
echo "Running ruff to fix remaining issues..."
cd /root/transformers
python -m ruff check --fix src/transformers/models/hindi_causal_lm/
python -m ruff check --fix tests/models/hindi_causal_lm/

echo "Style fixes applied. Run checks again to verify."