import os
import ujson
import random

from argparse import ArgumentParser

from colbert.utils.utils import print_message, create_directory, load_ranking, groupby_first_item
from utility.utils.qa_loaders import load_qas_


def main(args):
    print_message("#> Loading all..")
    qas = load_qas_(args.qas)
    rankings = load_ranking(args.ranking)
    qid2rankings = groupby_first_item(rankings)

    print_message("#> Subsampling all..")
    qas_sample = random.sample(qas, args.sample)

    with open(args.output, 'w') as f:
        for qid, *_ in qas_sample:
            for items in qid2rankings[qid]:
                items = [qid] + items
                line = '\t'.join(map(str, items)) + '\n'
                f.write(line)

    print('\n\n')
    print(args.output)
    print("#> Done.")


if __name__ == "__main__":
    random.seed(12345)

    parser = ArgumentParser(description='Subsample the dev set.')
    parser.add_argument('--qas', dest='qas', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True)
    parser.add_argument('--output', dest='output', required=True)

    parser.add_argument('--sample', dest='sample', default=1500, type=int)

    args = parser.parse_args()

    assert not os.path.exists(args.output), args.output
    create_directory(os.path.dirname(args.output))

    main(args)
