from arguments import TokenizerTrainingArguments
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer, HfArgumentParser
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


# Iterator for Training
def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, args.n_examples, batch_size)):
        yield [next(iter_dataset)[args.text_column] for _ in range(batch_size)]


# Configuration
parser = HfArgumentParser(TokenizerTrainingArguments)
args = parser.parse_args()

# Base tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
base_vocab = list(bytes_to_unicode().values())

# Load dataset
dataset = load_dataset(args.dataset_name, split="train", streaming=True)
iter_dataset = iter(dataset)


# Training and saving
new_tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(), vocab_size=args.vocab_size, initial_alphabet=base_vocab
)
new_tokenizer.save_pretrained(args.tokenizer_name, push_to_hub=args.push_to_hub)
