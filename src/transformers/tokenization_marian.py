#from .tokenization_utils import

from .tokenization_xlm_roberta import XLMRobertaTokenizer

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "marian/en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/marian/en-de/spm32k.vocab.yml"
    }}
VOCAB_NAME = 'spm32k.vocab.yml'


class MarianSPTokenizer(XLMRobertaTokenizer):
    vocab_files_names = {"vocab_file": VOCAB_NAME}
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = {m: 512 for m in PRETRAINED_VOCAB_FILES_MAP}
    # TODO(SS): model_input_names = ["attention_mask"]
