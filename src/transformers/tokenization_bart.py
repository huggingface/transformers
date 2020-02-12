from .tokenization_roberta import RobertaTokenizer


# vocab and merges same as roberta
vocab_url = "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json"
merges_url = "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt"
_all_bart_models = [
    "bart-large",
    "bart-large-mnli",
    # "bart-large-cnn"
]


class BartTokenizer(RobertaTokenizer):
    # merges and vocab same as roberta
    max_model_input_sizes = {m: 1024 for m in _all_bart_models}
    pretrained_vocab_files_map = {
        "vocab_file": {m: vocab_url for m in _all_bart_models},
        "merges_file": {m: merges_url for m in _all_bart_models},
    }
    n_seps_between_sentences = 1
