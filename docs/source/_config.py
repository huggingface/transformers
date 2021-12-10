# docstyle-ignore
INSTALL_CONTENT = """
# Transformers installation
! pip install transformers datasets
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/transformers.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]