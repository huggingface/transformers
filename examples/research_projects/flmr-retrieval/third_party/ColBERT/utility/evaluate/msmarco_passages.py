"""
    Evaluate MS MARCO Passages ranking.
"""

import os
import math
import tqdm
import ujson
import random

from argparse import ArgumentParser
from collections import defaultdict
from colbert.utils.utils import print_message, file_tqdm


def main(args):
    qid2positives = defaultdict(list)
    qid2ranking = defaultdict(list)
    qid2mrr = {}
    qid2recall = {depth: {} for depth in [50, 200, 1000, 5000, 10000]}

    with open(args.qrels) as f:
        print_message(f"#> Loading QRELs from {args.qrels} ..")
        for line in file_tqdm(f):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1

            qid2positives[qid].append(pid)

    with open(args.ranking) as f:
        print_message(f"#> Loading ranked lists from {args.ranking} ..")
        for line in file_tqdm(f):
            qid, pid, rank, *score = line.strip().split('\t')
            qid, pid, rank = int(qid), int(pid), int(rank)

            if len(score) > 0:
                assert len(score) == 1
                score = float(score[0])
            else:
                score = None

            qid2ranking[qid].append((rank, pid, score))

    assert set.issubset(set(qid2ranking.keys()), set(qid2positives.keys()))

    num_judged_queries = len(qid2positives)
    num_ranked_queries = len(qid2ranking)

    if num_judged_queries != num_ranked_queries:
        print()
        print_message("#> [WARNING] num_judged_queries != num_ranked_queries")
        print_message(f"#> {num_judged_queries} != {num_ranked_queries}")
        print()

    print_message(f"#> Computing MRR@10 for {num_judged_queries} queries.")

    for qid in tqdm.tqdm(qid2positives):
        ranking = qid2ranking[qid]
        positives = qid2positives[qid]

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                if rank <= 10:
                    qid2mrr[qid] = 1.0 / rank
                break

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                for depth in qid2recall:
                    if rank <= depth:
                        qid2recall[depth][qid] = qid2recall[depth].get(qid, 0) + 1.0 / len(positives)

    assert len(qid2mrr) <= num_ranked_queries, (len(qid2mrr), num_ranked_queries)

    print()
    mrr_10_sum = sum(qid2mrr.values())
    print_message(f"#> MRR@10 = {mrr_10_sum / num_judged_queries}")
    print_message(f"#> MRR@10 (only for ranked queries) = {mrr_10_sum / num_ranked_queries}")
    print()

    for depth in qid2recall:
        assert len(qid2recall[depth]) <= num_ranked_queries, (len(qid2recall[depth]), num_ranked_queries)

        print()
        metric_sum = sum(qid2recall[depth].values())
        print_message(f"#> Recall@{depth} = {metric_sum / num_judged_queries}")
        print_message(f"#> Recall@{depth} (only for ranked queries) = {metric_sum / num_ranked_queries}")
        print()

    if args.annotate:
        print_message(f"#> Writing annotations to {args.output} ..")

        with open(args.output, 'w') as f:
            for qid in tqdm.tqdm(qid2positives):
                ranking = qid2ranking[qid]
                positives = qid2positives[qid]

                for rank, (_, pid, score) in enumerate(ranking):
                    rank = rank + 1  # 1-indexed
                    label = int(pid in positives)

                    line = [qid, pid, rank, score, label]
                    line = [x for x in line if x is not None]
                    line = '\t'.join(map(str, line)) + '\n'
                    f.write(line)


if __name__ == "__main__":
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--qrels', dest='qrels', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--annotate', dest='annotate', default=False, action='store_true')

    args = parser.parse_args()

    if args.annotate:
        args.output = f'{args.ranking}.annotated'
        assert not os.path.exists(args.output), args.output

    main(args)
