# docstyle-ignore
INSTALL_CONTENT = """
# Instalare Transformers
! pip install transformers datasets evaluate accelerate
# Pentru a instala din codul sursă în loc de cea mai nouă versiune stabilă, fă linia de mai sus comentariu, iar pe cea de mai jos activă.
# ! pip install git+https://github.com/huggingface/transformers.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]
black_avoid_patterns = {
    "{processor_class}": "FakeProcessorClass",
    "{model_class}": "FakeModelClass",
    "{object_class}": "FakeObjectClass",
}
