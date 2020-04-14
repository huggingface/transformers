
from pathlib import Path
import json
import yaml
from transformers.tokenization_utils import ADDED_TOKENS_FILE, TOKENIZER_CONFIG_FILE

def write_json(content, path):
    with open(path, 'w') as f:
        json.dump(content, f)
def write_metadata(dest_dir):
    dest = Path(dest_dir)
    dname = dest.name.split('-')
    dct = dict(target_lang= dname[-1], source_lang='-'.join(dname[:-1]))
    write_json(dct, dest_dir /TOKENIZER_CONFIG_FILE)

#def write_added_tokens_file(dest_dir, vocab_size):
#    added_tokens = {''}
