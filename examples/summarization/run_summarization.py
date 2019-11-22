import argparse
from collections import namedtuple
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from transformers import BertTokenizer

from modeling_bertabs import BertAbs, build_predictor

from utils_summarization import (
    SummarizationDataset,
    encode_for_summarization,
    build_mask,
    fit_to_block_size,
    compute_token_type_ids,
)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


Batch = namedtuple(
    "Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"]
)


def evaluate(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = bertabs = BertAbs.from_pretrained(
        "bertabs-finetuned-{}".format(args.finetuned_model)
    )
    bertabs.to(args.device)
    bertabs.eval()

    symbols = {
        "BOS": tokenizer.vocab["[unused0]"],
        "EOS": tokenizer.vocab["[unused1]"],
        "PAD": tokenizer.vocab["[PAD]"],
    }

    # these (unused) arguments are defined to keep the compatibility
    # with the legacy code and will be deleted in a next iteration.
    args.result_path = ""
    args.temp_dir = ""

    data_iterator = build_data_iterator(args, tokenizer)
    predictor = build_predictor(args, tokenizer, symbols, model)

    logger.info("***** Running evaluation *****")
    logger.info("  Number examples = %d", len(data_iterator.dataset))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("")
    logger.info("***** Beam Search parameters *****")
    logger.info("  Beam size = %d", args.beam_size)
    logger.info("  Minimum length = %d", args.min_length)
    logger.info("  Maximum length = %d", args.max_length)
    logger.info("  Alpha (length penalty) = %.2f", args.alpha)
    logger.info("  Trigrams %s be blocked", ("will" if args.block_trigram else "will NOT"))

    for batch in tqdm(data_iterator):
        batch_data = predictor.translate_batch(batch)
        translations = predictor.from_batch(batch_data)
        summaries = [format_summary(t) for t in translations]
        save_summaries(summaries, args.summaries_output_dir, batch.document_names)


def format_summary(translation):
    """ Transforms the output of the `from_batch` function
    into nicely formatted summaries.
    """
    raw_summary, _, _ = translation
    summary = (
        raw_summary.replace("[unused0]", "")
        .replace("[unused3]", "")
        .replace("[PAD]", "")
        .replace("[unused1]", "")
        .replace(r" +", " ")
        .replace(" [unused2] ", ". ")
        .replace("[unused2]", "")
        .strip()
    )

    return summary


def save_summaries(summaries, path, original_document_name):
    """ Write the summaries in fies that are prefixed by the original
    files' name with the `_summary` appended.

    Attributes:
        original_document_names: List[string]
            Name of the document that was summarized.
        path: string
            Path were the summaries will be written
        summaries: List[string]
            The summaries that we produced.
    """
    for summary, document_name in zip(summaries, original_document_name):
        # Prepare the summary file's name
        if "." in document_name:
            bare_document_name = ".".join(document_name.split(".")[:-1])
            extension = document_name.split(".")[-1]
            name = bare_document_name + "_summary." + extension
        else:
            name = document_name + "_summary"

        file_path = os.path.join(path, name)
        with open(file_path, "w") as output:
            output.write(summary)


#
# LOAD the dataset
#


def build_data_iterator(args, tokenizer):
    dataset = load_and_cache_examples(args, tokenizer)
    sampler = SequentialSampler(dataset)
    collate_fn = lambda data: collate(data, tokenizer, block_size=512)
    iterator = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn,
    )

    return iterator


def load_and_cache_examples(args, tokenizer):
    dataset = SummarizationDataset(args.documents_dir)
    return dataset


def collate(data, tokenizer, block_size):
    """ Collate formats the data passed to the data loader.

    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    data = [x for x in data if not len(x[1]) == 0]  # remove empty_files
    names = [name for name, _, _ in data]

    encoded_text = [
        encode_for_summarization(story, summary, tokenizer) for _, story, summary in data
    ]
    stories = torch.tensor(
        [
            fit_to_block_size(story, block_size, tokenizer.pad_token_id)
            for story, _ in encoded_text
        ]
    )
    encoder_token_type_ids = compute_token_type_ids(stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(stories, tokenizer.pad_token_id)

    batch = Batch(
        document_names=names,
        batch_size=len(stories),
        src=stories,
        segs=encoder_token_type_ids,
        mask_src=encoder_mask,
        tgt_str=[""] * len(stories),
    )

    return batch


def decode_summary(summary_tokens, tokenizer):
    """ Decode the summary and return it in a format
    suitable for evaluation.
    """
    summary_tokens = summary_tokens.to("cpu").numpy()
    summary = tokenizer.decode(summary_tokens)
    sentences = summary.split(".")
    sentences = [s + "." for s in sentences]
    return sentences


def main():
    """ The main function defines the interface with the users.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents_dir",
        default=None,
        type=str,
        required=True,
        help="The folder where the documents to summarize are located.",
    )
    parser.add_argument(
        "--summaries_output_dir",
        default=None,
        type=str,
        required=True,
        help="The folder in wich the summaries should be written.",
    )
    # EVALUATION options
    parser.add_argument(
        "--visible_gpus",
        default=-1,
        type=int,
        help="Number of GPUs with which to do the training.",
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    # BEAM SEARCH arguments
    parser.add_argument(
        "--min_length",
        default=50,
        type=int,
        help="Minimum number of tokens for the summaries.",
    )
    parser.add_argument(
        "--max_length",
        default=200,
        type=int,
        help="Maixmum number of tokens for the summaries.",
    )
    parser.add_argument(
        "--beam_size",
        default=5,
        type=int,
        help="The number of beams to start with for each example.",
    )
    parser.add_argument(
        "--alpha",
        default=0.95,
        type=float,
        help="The value of alpha for the length penalty in the beam search.",
    )
    parser.add_argument(
        "--block_trigram",
        default=True,
        type=bool,
        help="Whether to block the existence of repeating trigrams in the text generated by beam search.",
    )
    args = parser.parse_args()
    args.device = torch.device("cpu") if args.visible_gpus == -1 else torch.device("cuda")

    if not documents_dir_is_valid(args.documents_dir):
        raise FileNotFoundError(
            "We could not find the directory you specified for the documents to summarize, or it was empty. Please specify a valid path."
        )
    maybe_create_output_dir(args.summaries_output_dir)

    evaluate(args)


def documents_dir_is_valid(path):
    if not os.path.exists(path):
        return False

    file_list = os.listdir(path)
    if len(file_list) == 0:
        return False

    return True


def maybe_create_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    main()
