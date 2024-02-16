import torch
import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from colbert.infra.launcher import Launcher
from colbert.infra import Run, RunConfig
from colbert.modeling.reranker.electra import ElectraReranker
from colbert.utils.utils import flatten


DEFAULT_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


class Scorer:
    def __init__(self, queries, collection, model=DEFAULT_MODEL, maxlen=180, bsize=256):
        self.queries = queries
        self.collection = collection
        self.model = model

        self.maxlen = maxlen
        self.bsize = bsize

    def launch(self, qids, pids):
        launcher = Launcher(self._score_pairs_process, return_all=True)
        outputs = launcher.launch(Run().config, qids, pids)

        return flatten(outputs)

    def _score_pairs_process(self, config, qids, pids):
        assert len(qids) == len(pids), (len(qids), len(pids))
        share = 1 + len(qids) // config.nranks
        offset = config.rank * share
        endpos = (1 + config.rank) * share

        return self._score_pairs(qids[offset:endpos], pids[offset:endpos], show_progress=(config.rank < 1))

    def _score_pairs(self, qids, pids, show_progress=False):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModelForSequenceClassification.from_pretrained(self.model).cuda()

        assert len(qids) == len(pids), (len(qids), len(pids))

        scores = []

        model.eval()
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                for offset in tqdm.tqdm(range(0, len(qids), self.bsize), disable=(not show_progress)):
                    endpos = offset + self.bsize

                    queries_ = [self.queries[qid] for qid in qids[offset:endpos]]
                    passages_ = [self.collection[pid] for pid in pids[offset:endpos]]

                    features = tokenizer(queries_, passages_, padding='longest', truncation=True,
                                            return_tensors='pt', max_length=self.maxlen).to(model.device)

                    scores.append(model(**features).logits.flatten())

        scores = torch.cat(scores)
        scores = scores.tolist()

        Run().print(f'Returning with {len(scores)} scores')

        return scores


# LONG-TERM TODO: This can be sped up by sorting by length in advance.
