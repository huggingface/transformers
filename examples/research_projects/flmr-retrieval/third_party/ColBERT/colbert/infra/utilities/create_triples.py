import random

from colbert.utils.utils import print_message
from utility.utils.save_metadata import save_metadata
from utility.supervision.triples import sample_for_query

from colbert.data.ranking import Ranking
from colbert.data.examples import Examples

MAX_NUM_TRIPLES = 40_000_000


class Triples:
    def __init__(self, ranking, seed=12345):
        random.seed(seed)  # TODO: Use internal RNG instead..
        self.qid2rankings = Ranking.cast(ranking).todict()

    def create(self, positives, depth):
        assert all(len(x) == 2 for x in positives)
        assert all(maxBest <= maxDepth for maxBest, maxDepth in positives), positives

        Triples = []
        NonEmptyQIDs = 0

        for processing_idx, qid in enumerate(self.qid2rankings):
            l = sample_for_query(qid, self.qid2rankings[qid], positives, depth, False, None)
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

        self.Triples = Examples(data=Triples)

        return Triples

    def save(self, new_path):
        Examples(data=self.Triples).save(new_path)

        # save_metadata(f'{output}.meta', args)  # TODO: What args to save?? {seed, positives, depth, rankings if path or else whatever provenance the rankings object shares}

