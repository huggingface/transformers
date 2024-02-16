import os
import sys
import tqdm
import ujson
import random

from argparse import ArgumentParser
from collections import OrderedDict
from colbert.utils.utils import print_message, file_tqdm


def main(args):
    qid_to_file_idx = {}

    for qrels_idx, qrels in enumerate(args.all_queries):
        with open(qrels) as f:
            for line in f:
                qid, *_ = line.strip().split('\t')
                qid = int(qid)

                assert qid_to_file_idx.get(qid, qrels_idx) == qrels_idx, (qid, qrels_idx)
                qid_to_file_idx[qid] = qrels_idx

    all_outputs_paths = [f'{args.ranking}.{idx}' for idx in range(len(args.all_queries))]

    assert all(not os.path.exists(path) for path in all_outputs_paths)

    all_outputs = [open(path, 'w') for path in all_outputs_paths]

    with open(args.ranking) as f:
        print_message(f"#> Loading ranked lists from {f.name} ..")

        last_file_idx = -1

        for line in file_tqdm(f):
            qid, *_ = line.strip().split('\t')

            file_idx = qid_to_file_idx[int(qid)]

            if file_idx != last_file_idx:
                print_message(f"#> Switched to file #{file_idx} at {all_outputs[file_idx].name}")

            last_file_idx = file_idx

            all_outputs[file_idx].write(line)

    print()

    for f in all_outputs:
        print(f.name)
        f.close()

    print("#> Done!")


if __name__ == "__main__":
    random.seed(12345)

    parser = ArgumentParser(description='.')

    # Input Arguments
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--all-queries', dest='all_queries', required=True, type=str, nargs='+')

    args = parser.parse_args()

    main(args)
