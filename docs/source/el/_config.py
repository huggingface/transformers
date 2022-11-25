# docstyle-ignore
INSTALL_CONTENT = """
# Εγκατάσταση Transformers
! pip install transformers datasets
# Για εγκατάσταση απο το τo source αντί του τελευταίου release, κάνε comment το command από πάνω και uncomment το ακόλουθο.
# ! pip install git+https://github.com/huggingface/transformers.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]
black_avoid_patterns = {
    "{processor_class}": "FakeProcessorClass",
    "{model_class}": "FakeModelClass",
    "{object_class}": "FakeObjectClass",    
}
