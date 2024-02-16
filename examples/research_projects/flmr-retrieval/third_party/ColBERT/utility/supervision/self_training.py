import os
import sys
import git
import tqdm
import ujson
import random

from argparse import ArgumentParser
from colbert.utils.utils import print_message, load_ranking, groupby_first_item


MAX_NUM_TRIPLES = 40_000_000


def sample_negatives(negatives, num_sampled, biased=False):
    num_sampled = min(len(negatives), num_sampled)

    if biased:
        assert num_sampled % 2 == 0
        num_sampled_top100 = num_sampled // 2
        num_sampled_rest = num_sampled - num_sampled_top100

        return random.sample(negatives[:100], num_sampled_top100) + random.sample(negatives[100:], num_sampled_rest)

    return random.sample(negatives, num_sampled)


def sample_for_query(qid, ranking, npositives, depth_positive, depth_negative, cutoff_negative):
    """
        Requires that the ranks are sorted per qid.
    """
    assert npositives <= depth_positive < cutoff_negative < depth_negative

    positives, negatives, triples = [], [], []

    for pid, rank, *_ in ranking:
        assert rank >= 1, f"ranks should start at 1 \t\t got rank = {rank}"

        if rank > depth_negative:
            break

        if rank <= depth_positive:
            positives.append(pid)
        elif rank > cutoff_negative:
            negatives.append(pid)

    num_sampled = 100

    for neg in sample_negatives(negatives, num_sampled):
        positives_ = random.sample(positives, npositives)
        positives_ = positives_[0] if npositives == 1 else positives_
        triples.append((qid, positives_, neg))

    return triples


def main(args):
    rankings = load_ranking(args.ranking, types=[int, int, int, float, int])

    print_message("#> Group by QID")
    qid2rankings = groupby_first_item(tqdm.tqdm(rankings))

    Triples = []
    NonEmptyQIDs = 0

    for processing_idx, qid in enumerate(qid2rankings):
        l = sample_for_query(qid, qid2rankings[qid], args.positives, args.depth_positive, args.depth_negative, args.cutoff_negative)
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

    with open(f'{args.output}.meta', 'w') as f:
        args.cmd = ' '.join(sys.argv)
        args.git_hash = git.Repo(search_parent_directories=True).head.object.hexsha
        ujson.dump(args.__dict__, f, indent=4)
        f.write('\n')

    print('\n\n', args, '\n\n')
    print(args.output)
    print_message("#> Done.")


if __name__ == "__main__":
    random.seed(12345)

    parser = ArgumentParser(description='Create training triples from ranked list.')

    # Input / Output Arguments
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--output', dest='output', required=True, type=str)

    # Weak Supervision Arguments.
    parser.add_argument('--positives', dest='positives', required=True, type=int)
    parser.add_argument('--depth+', dest='depth_positive', required=True, type=int)

    parser.add_argument('--depth-', dest='depth_negative', required=True, type=int)
    parser.add_argument('--cutoff-', dest='cutoff_negative', required=True, type=int)

    args = parser.parse_args()

    assert not os.path.exists(args.output), args.output

    main(args)
