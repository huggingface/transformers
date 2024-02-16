"""
    Example:  --positives 5,50 1,1000        ~~>    best-5 (in top-50)  +  best-1 (in top-1000)
"""

import os
import sys
import git
import tqdm
import ujson
import random

from argparse import ArgumentParser
from colbert.utils.utils import print_message, load_ranking, groupby_first_item, create_directory
from utility.utils.save_metadata import save_metadata


MAX_NUM_TRIPLES = 40_000_000


def sample_negatives(negatives, num_sampled, biased=None):
    assert biased in [None, 100, 200], "NOTE: We bias 50% from the top-200 negatives, if there are twice or more."

    num_sampled = min(len(negatives), num_sampled)

    if biased and num_sampled < len(negatives):
        assert num_sampled % 2 == 0, num_sampled

        num_sampled_top100 = num_sampled // 2
        num_sampled_rest = num_sampled - num_sampled_top100

        oversampled, undersampled = negatives[:biased], negatives[biased:]

        if len(oversampled) < len(undersampled):
            return random.sample(oversampled, num_sampled_top100) + random.sample(undersampled, num_sampled_rest)

    return random.sample(negatives, num_sampled)


def sample_for_query(qid, ranking, args_positives, depth, permissive, biased):
    """
        Requires that the ranks are sorted per qid.
    """

    positives, negatives, triples = [], [], []

    for pid, rank, *_, label in ranking:
        assert rank >= 1, f"ranks should start at 1 \t\t got rank = {rank}"
        assert label in [0, 1]

        if rank > depth:
            break

        if label:
            take_this_positive = any(rank <= maxDepth and len(positives) < maxBest for maxBest, maxDepth in args_positives)

            if take_this_positive:
                positives.append((pid, 0))
            elif permissive:
                positives.append((pid, rank))  # utilize with a few negatives, starting at (next) rank

        else:
            negatives.append(pid)

    for pos, neg_start in positives:
        num_sampled = 100 if neg_start == 0 else 5
        negatives_ = negatives[neg_start:]

        biased_ = biased if neg_start == 0 else None
        for neg in sample_negatives(negatives_, num_sampled, biased=biased_):
            triples.append((qid, pos, neg))

    return triples


def main(args):
    try:
        rankings = load_ranking(args.ranking, types=[int, int, int, float, int])
    except:
        rankings = load_ranking(args.ranking, types=[int, int, int, int])

    print_message("#> Group by QID")
    qid2rankings = groupby_first_item(tqdm.tqdm(rankings))

    Triples = []
    NonEmptyQIDs = 0

    for processing_idx, qid in enumerate(qid2rankings):
        l = sample_for_query(qid, qid2rankings[qid], args.positives, args.depth, args.permissive, args.biased)
        NonEmptyQIDs += (len(l) > 0)
        Triples.extend(l)

        if processing_idx % (10_000) == 0:
            print_message(f"#> Done with {processing_idx+1} questions!\t\t "
                          f"{str(len(Triples) / 1000)}k triples for {NonEmptyQIDs} unqiue QIDs.")

    print_message(f"#> Sub-sample the triples (if > {MAX_NUM_TRIPLES})..")
    print_message(f"#> len(Triples) = {len(Triples)}")

    if len(Triples) > MAX_NUM_TRIPLES:
        Triples = random.sample(Triples, MAX_NUM_TRIPLES)

    ### Prepare the triples ###
    print_message("#> Shuffling the triples...")
    random.shuffle(Triples)

    print_message("#> Writing {}M examples to file.".format(len(Triples) / 1000.0 / 1000.0))

    with open(args.output, 'w') as f:
        for example in Triples:
            ujson.dump(example, f)
            f.write('\n')

    save_metadata(f'{args.output}.meta', args)

    print('\n\n', args, '\n\n')
    print(args.output)
    print_message("#> Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description='Create training triples from ranked list.')

    # Input / Output Arguments
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--output', dest='output', required=True, type=str)

    # Weak Supervision Arguments.
    parser.add_argument('--positives', dest='positives', required=True, nargs='+')
    parser.add_argument('--depth', dest='depth', required=True, type=int)  # for negatives

    parser.add_argument('--permissive', dest='permissive', default=False, action='store_true')
    # parser.add_argument('--biased', dest='biased', default=False, action='store_true')
    parser.add_argument('--biased', dest='biased', default=None, type=int)
    parser.add_argument('--seed', dest='seed', required=False, default=12345, type=int)

    args = parser.parse_args()
    random.seed(args.seed)

    assert not os.path.exists(args.output), args.output

    args.positives = [list(map(int, configuration.split(','))) for configuration in args.positives]

    assert all(len(x) == 2 for x in args.positives)
    assert all(maxBest <= maxDepth for maxBest, maxDepth in args.positives), args.positives

    create_directory(os.path.dirname(args.output))

    assert args.biased in [None, 100, 200]

    main(args)
