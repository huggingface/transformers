from pathlib import Path
from tokenizers import BertWordPieceTokenizer

root_path = r"/home/wuyan/usr/material/bert_corpus"
paths = [str(x) for x in Path(root_path).glob("**/*.txt")]

tokenizer = BertWordPieceTokenizer()

tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "[UNK]",
    "[SEP]",
    "[CLS]",
    "[PAD]",
    "[MASK]",
])

# Save files to disk
tokenizer.save_model(".", "material")
