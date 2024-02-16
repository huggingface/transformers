"""
Split the ranked lists after retrieval with a merged query set.
"""

import os
import random

from argparse import ArgumentParser


def main(args):
    output_paths = ['{}.{}'.format(args.ranking, split) for split in args.names]
    assert all(not os.path.exists(path) for path in output_paths), output_paths

    output_files = [open(path, 'w') for path in output_paths]

    with open(args.ranking) as f:
        for line in f:
            qid, pid, rank, *other = line.strip().split('\t')
            qid = int(qid)
            split_output_path = output_files[qid // args.gap - 1]
            qid = qid % args.gap

            split_output_path.write('\t'.join([str(x) for x in [qid, pid, rank, *other]]) + '\n')
        
        print(f.name)
    
    _ = [f.close() for f in output_files]
    
    print("#> Done!")


if __name__ == "__main__":
    random.seed(12345)

    parser = ArgumentParser(description='Subsample the dev set.')
    parser.add_argument('--ranking', dest='ranking', required=True)

    parser.add_argument('--names', dest='names', required=False, default=['train', 'dev', 'test'], type=str, nargs='+')  # order matters!
    parser.add_argument('--gap', dest='gap', required=False, default=1_000_000_000, type=int)  # larger than any individual query set

    args = parser.parse_args()

    main(args)
