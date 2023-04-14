# 2) Add the new classes to the module __init__ (CHECK)
# 3) Add the new classes to the module autodoc (CHECK)
# 4) Update the root __init__ to import the new classes (CHECK)
# 5) Add the missing Auto classes (CHECK)
# 6) Add the module file to the doctest list
#   - Happens automatically: Update the model support checklist
#   - Happens automatically: Update the dummies

from pathlib import Path
import re

init_file = Path("src/transformers/models/gpt_neo/__init__.py")
contents = init_file.read_text()
if "is_tf_available" in contents:
    raise ValueError("This file already contains TF imports!")
torch_block_re = r"try:[\n\s]+if not is_torch_available\(\)\:[\n\s]+raise OptionalDependencyNotAvailable.*?else.*?\].*?\]"
torch_block = re.search(torch_block_re, contents, re.DOTALL)
torch_block_text = torch_block.group(0)
breakpoint()
print()
