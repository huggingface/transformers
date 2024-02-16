import random
from colbert.infra.provenance import Provenance

from utility.utils.save_metadata import save_metadata
from utility.supervision.triples import sample_for_query

from colbert.utils.utils import print_message

from colbert.data.ranking import Ranking
from colbert.data.examples import Examples

MAX_NUM_TRIPLES = 40_000_000


class Triples:
    def __init__(self, ranking, seed=12345):
        random.seed(seed)  # TODO: Use internal RNG instead..
        self.seed = seed

        ranking = Ranking.cast(ranking)
        self.ranking_provenance = ranking.provenance()
        self.qid2rankings = ranking.todict()

    def create(self, positives, depth):
        assert all(len(x) == 2 for x in positives)
        assert all(maxBest <= maxDepth for maxBest, maxDepth in positives), positives

        self.positives = positives
        self.depth = depth

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
        provenance = Provenance()
        provenance.source = 'Triples::create'
        provenance.seed = self.seed
        provenance.positives = self.positives
        provenance.depth = self.depth
        provenance.ranking = self.ranking_provenance

        Examples(data=self.Triples, provenance=provenance).save(new_path)
