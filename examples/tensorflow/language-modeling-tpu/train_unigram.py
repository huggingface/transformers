#!/usr/bin/env python
import argparse
import logging
import os

import datasets
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

from transformers import AlbertTokenizerFast

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a unigram tokenizer on the wikitext dataset.")
    parser.add_argument(
        "--dataset_name", type=str, default="wikitext",
        help="Name of the dataset (see hf.co/datasets)."
    )
    parser.add_argument(
        "--dataset_config", type=str, default="wikitext-103-raw-v1",
        help="Configuration name of the dataset."
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Whether to trust execution of remote code."
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size during training.")
    parser.add_argument("--vocab_size", type=int, default=10048, help="Vocabulary size.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples.")
    parser.add_argument("--export_to_hub", action="store_true", help="Push tokenizer to Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default="./unigram_tokenizer", help="Directory to save tokenizer.")
    return parser.parse_args()


def main(args):
    dataset = datasets.load_dataset(
        args.dataset_name, args.dataset_config, split="train", trust_remote_code=args.trust_remote_code
    )

    if args.limit is not None:
        max_train_samples = min(len(dataset), args.limit)
        dataset = dataset.select(range(max_train_samples))
        logger.info(f"Limiting the dataset to {args.limit} entries.")

    def batch_iterator():
        for i in range(0, len(dataset), args.batch_size):
            yield dataset[i : i + args.batch_size]["text"]

    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"')
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    trainer = UnigramTrainer(
        unk_token="<unk>",
        special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"],
        vocab_size=args.vocab_size,
    )

    logger.info("Training the tokenizer.")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    logger.info("Tokenizer training complete!")

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    tokenizer.decoder = decoders.Metaspace()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")

    fast_tokenizer = AlbertTokenizerFast(tokenizer_file=tokenizer_path)

    if args.export_to_hub:
        logger.info("Exporting the tokenizer to the Hub.")
        fast_tokenizer.push_to_hub("unigram-tokenizer-dataset")


if __name__ == "__main__":
    args = parse_args()
    main(args)
