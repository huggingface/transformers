#this code is almost the same as tokenization_gpt2. 
#the original blenderbot is also using byte-level bpe tokenization based on subword-nmt
import logging
import os
from .tokenization_roberta import RobertaTokenizer

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# will update paths once uploded files on S3
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/blenderbot-90M": os.path.abspath(os.path.expanduser("blenderbot-vocab.json")),
    },
    "merges_file": {
        "facebook/blenderbot-90M": os.path.abspath(os.path.expanduser("blenderbot-merges.txt")),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "blenderbot": 1024,
}
logger = logging.getLogger(__name__)

def BlenderbotTokenizer(RobertaTokenizer):
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    