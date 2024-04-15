# docstyle-ignore
INSTALL_CONTENT = """
# Installazione di Transformers
! pip install transformers datasets evaluate accelerate
# Per installare dalla fonte invece dell'ultima versione rilasciata, commenta il comando sopra e
# rimuovi la modalità commento al comando seguente.
# ! pip install git+https://github.com/huggingface/transformers.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]
black_avoid_patterns = {
    "{processor_class}": "FakeProcessorClass",
    "{model_class}": "FakeModelClass",
    "{object_class}": "FakeObjectClass",
}
