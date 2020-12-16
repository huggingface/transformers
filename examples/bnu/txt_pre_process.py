from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tokenizers import BertWordPieceTokenizer
import platform
import os

# paths = [str(x) for x in Path(r"G:\BTU\bert_corpus\bert_corpus\alloys\1").glob("**/*.txt")]
paths = r"G:\BTU\bert_corpus\bert_corpus\alloys\1\1.txt"
print(paths)


# Initialize a tokenizer
# tokenizer = ByteLevelBPETokenizer()
# tokenizer = ByteLevelBPETokenizer()
#
#
# # Customize training
# tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
#     "[UNK]",
#     "[SEP]",
#     "[CLS]",
#     "[PAD]",
#     "[MASK]",
# ])

# Save files to disk
# tokenizer.save_model(".", "material")


print(platform.platform())
print(os.name)