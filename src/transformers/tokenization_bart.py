from transformers import RobertaTokenizer


class BartTokenizer(RobertaTokenizer):
    # merges and vocab same as roberta
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
        "bart-large": 1024,
        "bart-large-mnli": 1024,
        "bart-large-cnn": 1024,
    }
