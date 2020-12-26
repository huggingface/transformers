from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing

root_path = r"/home/wuyan/usr/material/bert_corpus"
paths = [str(x) for x in Path(root_path).glob("**/*.txt")]
# paths = [r'1.txt']

tokenizer = BertWordPieceTokenizer()
# print(tokenizer)
tokenizer.train(files=paths, vocab_size=28_996, min_frequency=2, special_tokens=[
    "[UNK]",
    "[SEP]",
    "[CLS]",
    "[PAD]",
    "[MASK]",
])

# Save files to disk
tokenizer.save_model(".", "material")

# tokenizer._tokenizer.post_processor = BertProcessing(('[SEP]',tokenizer.token_to_id('[SEP]')),('[CLS]',tokenizer.token_to_id('[CLS]')))
# tokenizer.enable_truncation(max_length=512)
#
# print(tokenizer.encode("hello i am wuyan . i am  a boy.").tokens)
