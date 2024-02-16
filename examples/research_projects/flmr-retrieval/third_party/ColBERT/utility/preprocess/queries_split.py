"""
    Divide a query set into two.
"""

import os
import math
import ujson
import random

from argparse import ArgumentParser
from collections import OrderedDict
from colbert.utils.utils import print_message


def main(args):
    random.seed(12345)

    """
    Load the queries
    """
    Queries = OrderedDict()

    print_message(f"#> Loading queries from {args.input}..")
    with open(args.input) as f:
        for line in f:
            qid, query = line.strip().split('\t')

            assert qid not in Queries
            Queries[qid] = query

    """
    Apply the splitting
    """
    size_a = len(Queries) - args.holdout
    size_b = args.holdout
    size_a, size_b = max(size_a, size_b), min(size_a, size_b)

    assert size_a > 0 and size_b > 0, (len(Queries), size_a, size_b)

    print_message(f"#> Deterministically splitting the queries into ({size_a}, {size_b})-sized splits.")

    keys = list(Queries.keys())
    sample_b_indices = sorted(list(random.sample(range(len(keys)), size_b)))
    sample_a_indices = sorted(list(set.difference(set(list(range(len(keys)))), set(sample_b_indices))))

    assert len(sample_a_indices) == size_a
    assert len(sample_b_indices) == size_b

    sample_a = [keys[idx] for idx in sample_a_indices]
    sample_b = [keys[idx] for idx in sample_b_indices]

    """
    Write the output
    """

    output_path_a = f'{args.input}.a'
    output_path_b = f'{args.input}.b'

    assert not os.path.exists(output_path_a), output_path_a
    assert not os.path.exists(output_path_b), output_path_b

    print_message(f"#> Writing the splits out to {output_path_a} and {output_path_b} ...")

    for output_path, sample in [(output_path_a, sample_a), (output_path_b, sample_b)]:
        with open(output_path, 'w') as f:
            for qid in sample:
                query = Queries[qid]
                line = '\t'.join([qid, query]) + '\n'
                f.write(line)


if __name__ == "__main__":
    parser = ArgumentParser(description="queries_split.")

    # Input Arguments.
    parser.add_argument('--input', dest='input', required=True)
    parser.add_argument('--holdout', dest='holdout', required=True, type=int)

    args = parser.parse_args()

    main(args)
