import os
import sys
import git
import tqdm
import ujson
import random

from argparse import ArgumentParser
from multiprocessing import Pool

from colbert.utils.utils import print_message, load_ranking, groupby_first_item
from utility.utils.qa_loaders import load_qas_, load_collection_
from utility.utils.save_metadata import format_metadata, get_metadata
from utility.evaluate.annotate_EM_helpers import *


# TODO: Tokenize passages in advance, especially if the ranked list is long! This requires changes to the has_answer input, slightly.

def main(args):
    qas = load_qas_(args.qas)
    collection = load_collection_(args.collection, retain_titles=True)
    rankings = load_ranking(args.ranking)
    parallel_pool = Pool(30)

    print_message('#> Tokenize the answers in the Q&As in parallel...')
    qas = list(parallel_pool.map(tokenize_all_answers, qas))
    
    qid2answers = {qid: tok_answers for qid, _, tok_answers in qas}
    assert len(qas) == len(qid2answers), (len(qas), len(qid2answers))

    print_message('#> Lookup passages from PIDs...')
    expanded_rankings = [(qid, pid, rank, collection[pid], qid2answers[qid])
                         for qid, pid, rank, *_ in rankings]

    print_message('#> Assign labels in parallel...')
    labeled_rankings = list(parallel_pool.map(assign_label_to_passage, enumerate(expanded_rankings)))

    # Dump output.
    print_message("#> Dumping output to", args.output, "...")
    qid2rankings = groupby_first_item(labeled_rankings)

    num_judged_queries, num_ranked_queries = check_sizes(qid2answers, qid2rankings)

    # Evaluation metrics and depths.
    success, counts = compute_and_write_labels(args.output, qid2answers, qid2rankings)

    # Dump metrics.
    with open(args.output_metrics, 'w') as f:
        d = {'num_ranked_queries': num_ranked_queries, 'num_judged_queries': num_judged_queries}

        extra = '__WARNING' if num_judged_queries != num_ranked_queries else ''
        d[f'success{extra}'] = {k: v / num_judged_queries for k, v in success.items()}
        d[f'counts{extra}'] = {k: v / num_judged_queries for k, v in counts.items()}
        d['arguments'] = get_metadata(args)

        f.write(format_metadata(d) + '\n')

    print('\n\n')
    print(args.output)
    print(args.output_metrics)
    print("#> Done\n")


if __name__ == "__main__":
    random.seed(12345)

    parser = ArgumentParser(description='.')

    # Input / Output Arguments
    parser.add_argument('--qas', dest='qas', required=True, type=str)
    parser.add_argument('--collection', dest='collection', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)

    args = parser.parse_args()

    args.output = f'{args.ranking}.annotated'
    args.output_metrics = f'{args.ranking}.annotated.metrics'

    assert not os.path.exists(args.output), args.output

    main(args)
