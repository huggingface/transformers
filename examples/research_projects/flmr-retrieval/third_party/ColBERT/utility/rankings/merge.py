"""
    Divide two or more ranking files, by score.
"""

import os
import tqdm

from argparse import ArgumentParser
from collections import defaultdict
from colbert.utils.utils import print_message, file_tqdm


def main(args):
    Rankings = defaultdict(list)

    for path in args.input:
        print_message(f"#> Loading the rankings in {path} ..")

        with open(path) as f:
            for line in file_tqdm(f):
                qid, pid, rank, score = line.strip().split('\t')
                qid, pid, rank = map(int, [qid, pid, rank])
                score = float(score)

                Rankings[qid].append((score, rank, pid))

    with open(args.output, 'w') as f:
        print_message(f"#> Writing the output rankings to {args.output} ..")

        for qid in tqdm.tqdm(Rankings):
            ranking = sorted(Rankings[qid], reverse=True)

            for rank, (score, original_rank, pid) in enumerate(ranking):
                rank = rank + 1  # 1-indexed

                if (args.depth > 0) and (rank > args.depth):
                    break

                line = [qid, pid, rank, score]
                line = '\t'.join(map(str, line)) + '\n'
                f.write(line)


if __name__ == "__main__":
    parser = ArgumentParser(description="merge_rankings.")

    # Input Arguments.
    parser.add_argument('--input', dest='input', required=True, nargs='+')
    parser.add_argument('--output', dest='output', required=True, type=str)

    parser.add_argument('--depth', dest='depth', required=True, type=int)

    args = parser.parse_args()

    assert not os.path.exists(args.output), args.output

    main(args)
