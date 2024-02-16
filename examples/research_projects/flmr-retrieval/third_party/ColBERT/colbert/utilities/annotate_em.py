
import os
import sys
import git
import tqdm
import ujson
import random

from argparse import ArgumentParser
from multiprocessing import Pool

from colbert.utils.utils import groupby_first_item, print_message
from utility.utils.qa_loaders import load_qas_, load_collection_
from utility.utils.save_metadata import format_metadata, get_metadata
from utility.evaluate.annotate_EM_helpers import *

from colbert.infra.run import Run
from colbert.data.collection import Collection
from colbert.data.ranking import Ranking


class AnnotateEM:
    def __init__(self, collection, qas):
        # TODO: These should just be Queries! But Queries needs to support looking up answers as qid2answers below.
        qas = load_qas_(qas)
        collection = Collection.cast(collection)  # .tolist() #load_collection_(collection, retain_titles=True)

        self.parallel_pool = Pool(30)

        print_message('#> Tokenize the answers in the Q&As in parallel...')
        qas = list(self.parallel_pool.map(tokenize_all_answers, qas))

        qid2answers = {qid: tok_answers for qid, _, tok_answers in qas}
        assert len(qas) == len(qid2answers), (len(qas), len(qid2answers))

        self.qas, self.collection = qas, collection
        self.qid2answers = qid2answers

    def annotate(self, ranking):
        rankings = Ranking.cast(ranking)

        # print(len(rankings), rankings[0])

        print_message('#> Lookup passages from PIDs...')
        expanded_rankings = [(qid, pid, rank, self.collection[pid], self.qid2answers[qid])
                             for qid, pid, rank, *_ in rankings.tolist()]

        print_message('#> Assign labels in parallel...')
        labeled_rankings = list(self.parallel_pool.map(assign_label_to_passage, enumerate(expanded_rankings)))

        # Dump output.
        self.qid2rankings = groupby_first_item(labeled_rankings)

        self.num_judged_queries, self.num_ranked_queries = check_sizes(self.qid2answers, self.qid2rankings)

        # Evaluation metrics and depths.
        self.success, self.counts = self._compute_labels(self.qid2answers, self.qid2rankings)

        print(rankings.provenance(), self.success)

        return Ranking(data=self.qid2rankings, provenance=("AnnotateEM", rankings.provenance()))

    def _compute_labels(self, qid2answers, qid2rankings):
        cutoffs = [1, 5, 10, 20, 30, 50, 100, 1000, 'all']
        success = {cutoff: 0.0 for cutoff in cutoffs}
        counts = {cutoff: 0.0 for cutoff in cutoffs}

        for qid in qid2answers:
            if qid not in qid2rankings:
                continue

            prev_rank = 0  # ranks should start at one (i.e., and not zero)
            labels = []

            for pid, rank, label in qid2rankings[qid]:
                assert rank == prev_rank+1, (qid, pid, (prev_rank, rank))
                prev_rank = rank

                labels.append(label)

            for cutoff in cutoffs:
                if cutoff != 'all':
                    success[cutoff] += sum(labels[:cutoff]) > 0
                    counts[cutoff] += sum(labels[:cutoff])
                else:
                    success[cutoff] += sum(labels) > 0
                    counts[cutoff] += sum(labels)

        return success, counts

    def save(self, new_path):
        print_message("#> Dumping output to", new_path, "...")

        Ranking(data=self.qid2rankings).save(new_path)

        # Dump metrics.
        with Run().open(f'{new_path}.metrics', 'w') as f:
            d = {'num_ranked_queries': self.num_ranked_queries, 'num_judged_queries': self.num_judged_queries}

            extra = '__WARNING' if self.num_judged_queries != self.num_ranked_queries else ''
            d[f'success{extra}'] = {k: v / self.num_judged_queries for k, v in self.success.items()}
            d[f'counts{extra}'] = {k: v / self.num_judged_queries for k, v in self.counts.items()}
            # d['arguments'] = get_metadata(args)  # TODO: Need arguments...

            f.write(format_metadata(d) + '\n')


if __name__ == '__main__':
    r = sys.argv[2]

    a = AnnotateEM(collection='/dfs/scratch0/okhattab/OpenQA/collection.tsv',
                   qas=sys.argv[1])
    a.annotate(ranking=r)
