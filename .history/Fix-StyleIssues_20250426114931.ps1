# Fix-StyleIssues.ps1
# PowerShell script to fix common code style issues

Write-Host "Fixing code style issues..." -ForegroundColor Green

# Create a temporary Python script to handle the fixes
$tempScriptPath = [System.IO.Path]::GetTempFileName() + ".py"
@"
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
    with open(filepath, 'w', encoding='utf-8', newline='\n') as file:
        file.write(content)

# Paths to fix - update with full paths if needed
repo_root = '.'  # Set this to your repo root path if script is run from elsewhere
paths = [
    f"{repo_root}/src/transformers/__init__.py",
    f"{repo_root}/src/transformers/models/__init__.py",
    f"{repo_root}/src/transformers/models/auto/configuration_auto.py",
    f"{repo_root}/src/transformers/models/auto/modeling_auto.py",
    f"{repo_root}/src/transformers/models/auto/tokenization_auto.py",
    f"{repo_root}/src/transformers/models/hindi_causal_lm/__init__.py",
        f"{repo_root}/src/transformers/models/hindi_causal_lm/__init__.py",
    f"{repo_root}/src/transformers/models/hindi_causal_lm/configuration_hindi_causal_lm.py",
    f"{repo_root}/src/transformers/models/hindi_causal_lm/modeling_hindi_causal_lm.py", 
    f"{repo_root}/src/transformers/models/hindi_causal_lm/tokenization_hindi_causal_lm.py",
    f"{repo_root}/tests/models/hindi_causal_lm/test_modeling_hindi_causal_lm.py"
]

for path in paths:
    if os.path.exists(path):
        fix_file(path)
    else:
        print(f"Warning: {path} not found")

print("Done fixing basic style issues")

# Fix duplicate test method in test file
test_file = f"{repo_root}/tests/models/hindi_causal_lm/test_modeling_hindi_causal_lm.py"
if os.path.exists(test_file):
    with open(test_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove the second definition of test_gradient_checkpointing
    pattern = r'    # --- Corrected test_gradient_checkpointing override ---\n    @unittest\.skip\(reason="Gradient checkpointing tests skipped\."\)\n    def test_gradient_checkpointing\(self\):\n        pass'
    content = re.sub(pattern, '', content)
    
    # Fix the whitespace after removal
    content = re.sub(r'\n\s+\n', '\n\n', content)
    
    with open(test_file, 'w', encoding='utf-8', newline='\n') as file:
        file.write(content)
    
    print(f"Fixed duplicate test in {test_file}")

# Fix tokenization_auto.py
tokenization_auto_file = f"{repo_root}/src/transformers/models/auto/tokenization_auto.py"
if os.path.exists(tokenization_auto_file):
    with open(tokenization_auto_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Fix the specific line with trailing whitespace
    content = content.replace('                 None \n', '                None\n')
    
    # Fix unused import
    content = content.replace('from ...tokenization_utils import PreTrainedTokenizer\n', '')
    
    with open(tokenization_auto_file, 'w', encoding='utf-8', newline='\n') as file:
        file.write(content)
    
    print(f"Fixed issues in {tokenization_auto_file}")
"@ | Out-File -FilePath $tempScriptPath -Encoding UTF8

# Execute the Python script
python $tempScriptPath

# Clean up the temporary file
Remove-Item $tempScriptPath

Write-Host "Style fixes applied. Now let's run the ruff formatter for more complex issues..." -ForegroundColor Green

# Run ruff formatter on the specific directories
$repoRoot = "."  # Set this to your repo root if running from elsewhere
& python -m ruff check --fix "$repoRoot/src/transformers/models/hindi_causal_lm/"
& python -m ruff check --fix "$repoRoot/tests/models/hindi_causal_lm/"

Write-Host "Style fixes complete. Run the repo checks to verify all issues are fixed." -ForegroundColor Green
Write-Host "You can run: python utils/check_repo.py" -ForegroundColor Cyan