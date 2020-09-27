import re


try:
    import nltk

    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    try:
        nltk.download("punkt", quiet=True)
    except FileExistsError:  # multiprocessing race condition
        pass


def add_newline_to_end_of_each_sentence(x: str) -> str:
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert NLTK_AVAILABLE, "nltk must be installed to separate newlines betwee sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))
