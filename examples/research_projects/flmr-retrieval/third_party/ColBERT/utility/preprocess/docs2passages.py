"""
    Divide a document collection into N-word/token passage spans (with wrap-around for last passage).
"""

import os
import math
import ujson
import random

from multiprocessing import Pool
from argparse import ArgumentParser
from colbert.utils.utils import print_message

Format1 = 'docid,text'  # MS MARCO Passages
Format2 = 'docid,text,title'   # DPR Wikipedia
Format3 = 'docid,url,title,text'  # MS MARCO Documents


def process_page(inp):
    """
        Wraps around if we split: make sure last passage isn't too short.
        This is meant to be similar to the DPR preprocessing.
    """

    (nwords, overlap, tokenizer), (title_idx, docid, title, url, content) = inp

    if tokenizer is None:
        words = content.split()
    else:
        words = tokenizer.tokenize(content)

    words_ = (words + words) if len(words) > nwords else words
    passages = [words_[offset:offset + nwords] for offset in range(0, len(words) - overlap, nwords - overlap)]

    assert all(len(psg) in [len(words), nwords] for psg in passages), (list(map(len, passages)), len(words))

    if tokenizer is None:
        passages = [' '.join(psg) for psg in passages]
    else:
        passages = [' '.join(psg).replace(' ##', '') for psg in passages]

    if title_idx % 100000 == 0:
        print("#> ", title_idx, '\t\t\t', title)

        for p in passages:
            print("$$$ ", '\t\t', p)
            print()

        print()
        print()
        print()

    return (docid, title, url, passages)


def main(args):
    random.seed(12345)
    print_message("#> Starting...")

    letter = 'w' if not args.use_wordpiece else 't'
    output_path = f'{args.input}.{letter}{args.nwords}_{args.overlap}'
    assert not os.path.exists(output_path)

    RawCollection = []
    Collection = []

    NumIllFormattedLines = 0

    with open(args.input) as f:
        for line_idx, line in enumerate(f):
            if line_idx % (100*1000) == 0:
                print(line_idx, end=' ')

            title, url = None, None

            try:
                line = line.strip().split('\t')

                if args.format == Format1:
                    docid, doc = line
                elif args.format == Format2:
                    docid, doc, title = line
                elif args.format == Format3:
                    docid, url, title, doc = line

                RawCollection.append((line_idx, docid, title, url, doc))
            except:
                NumIllFormattedLines += 1

                if NumIllFormattedLines % 1000 == 0:
                    print(f'\n[{line_idx}] NumIllFormattedLines = {NumIllFormattedLines}\n')

    print()
    print_message("# of documents is", len(RawCollection), '\n')

    p = Pool(args.nthreads)

    print_message("#> Starting parallel processing...")

    tokenizer = None
    if args.use_wordpiece:
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    process_page_params = [(args.nwords, args.overlap, tokenizer)] * len(RawCollection)
    Collection = p.map(process_page, zip(process_page_params, RawCollection))

    print_message(f"#> Writing to {output_path} ...")
    with open(output_path, 'w') as f:
        line_idx = 1

        if args.format == Format1:
            f.write('\t'.join(['id', 'text']) + '\n')
        elif args.format == Format2:
            f.write('\t'.join(['id', 'text', 'title']) + '\n')
        elif args.format == Format3:
            f.write('\t'.join(['id', 'text', 'title', 'docid']) + '\n')

        for docid, title, url, passages in Collection:
            for passage in passages:
                if args.format == Format1:
                    f.write('\t'.join([str(line_idx), passage]) + '\n')
                elif args.format == Format2:
                    f.write('\t'.join([str(line_idx), passage, title]) + '\n')
                elif args.format == Format3:
                    f.write('\t'.join([str(line_idx), passage, title, docid]) + '\n')

                line_idx += 1


if __name__ == "__main__":
    parser = ArgumentParser(description="docs2passages.")

    # Input Arguments.
    parser.add_argument('--input', dest='input', required=True)
    parser.add_argument('--format', dest='format', required=True, choices=[Format1, Format2, Format3])

    # Output Arguments.
    parser.add_argument('--use-wordpiece', dest='use_wordpiece', default=False, action='store_true')
    parser.add_argument('--nwords', dest='nwords', default=100, type=int)
    parser.add_argument('--overlap', dest='overlap', default=0, type=int)

    # Other Arguments.
    parser.add_argument('--nthreads', dest='nthreads', default=28, type=int)

    args = parser.parse_args()
    assert args.nwords in range(50, 500)

    main(args)
