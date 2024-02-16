from baleen.utils.loaders import *
from baleen.condenser.condense import Condenser


class Baleen:
    def __init__(self, collectionX_path: str, searcher, condenser: Condenser):
        self.collectionX = load_collectionX(collectionX_path)
        self.searcher = searcher
        self.condenser = condenser

    def search(self, query, num_hops, depth=100, verbose=False):
        assert depth % num_hops == 0, f"depth={depth} must be divisible by num_hops={num_hops}."
        k = depth // num_hops

        searcher = self.searcher
        condenser = self.condenser
        collectionX = self.collectionX

        facts = []
        stage1_preds = None
        context = None

        pids_bag = set()

        for hop_idx in range(0, num_hops):
            ranking = list(zip(*searcher.search(query, context=context, k=depth)))
            ranking_ = []

            facts_pids = set([pid for pid, _ in facts])

            for pid, rank, score in ranking:
                # print(f'[{score}] \t\t {searcher.collection[pid]}')
                if len(ranking_) < k and pid not in facts_pids:
                    ranking_.append(pid)
                
                if len(pids_bag) < k * (hop_idx+1):
                    pids_bag.add(pid)
            
            stage1_preds, facts, stage2_L3x = condenser.condense(query, backs=facts, ranking=ranking_)
            context = ' [SEP] '.join([collectionX.get((pid, sid), '') for pid, sid in facts])

        assert len(pids_bag) == depth

        return stage2_L3x, pids_bag, stage1_preds

            


            
                

        
            





