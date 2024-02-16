import torch

from colbert.utils.utils import load_checkpoint
from colbert.utils.amp import MixedPrecisionManager
from colbert.utils.utils import flatten

from baleen.utils.loaders import *
from baleen.condenser.model import ElectraReader
from baleen.condenser.tokenization import AnswerAwareTokenizer



class Condenser:
    def __init__(self, collectionX_path, checkpointL1, checkpointL2, deviceL1='cuda', deviceL2='cuda'):
        self.modelL1, self.maxlenL1 = self._load_model(checkpointL1, deviceL1)
        self.modelL2, self.maxlenL2 = self._load_model(checkpointL2, deviceL2)

        assert self.maxlenL1 == self.maxlenL2, "Add support for different maxlens: use two tokenizers."

        self.amp, self.tokenizer = self._setup_inference(self.maxlenL2)
        self.CollectionX, self.CollectionY = self._load_collection(collectionX_path)

    def condense(self, query, backs, ranking):
        stage1_preds = self._stage1(query, backs, ranking)
        stage2_preds, stage2_preds_L3x = self._stage2(query, stage1_preds)

        return stage1_preds, stage2_preds, stage2_preds_L3x

    def _load_model(self, path, device):
        model = torch.load(path, map_location='cpu')
        ElectraModels = ['google/electra-base-discriminator', 'google/electra-large-discriminator']
        assert model['arguments']['model'] in ElectraModels, model['arguments']

        model = ElectraReader.from_pretrained(model['arguments']['model'])
        checkpoint = load_checkpoint(path, model)

        model = model.to(device)
        model.eval()

        maxlen = checkpoint['arguments']['maxlen']

        return model, maxlen
    
    def _setup_inference(self, maxlen):
        amp = MixedPrecisionManager(activated=True)
        tokenizer = AnswerAwareTokenizer(total_maxlen=maxlen)

        return amp, tokenizer
    
    def _load_collection(self, collectionX_path):
        CollectionX = {}
        CollectionY = {}

        with open(collectionX_path) as f:
            for line_idx, line in enumerate(f):
                line = ujson.loads(line)

                assert type(line['text']) is list
                assert line['pid'] == line_idx, (line_idx, line)

                passage = [line['title']] + line['text']
                CollectionX[line_idx] = passage

                passage = [line['title'] + ' | ' + sentence for sentence in line['text']]

                for idx, sentence in enumerate(passage):
                    CollectionY[(line_idx, idx)] = sentence
        
        return CollectionX, CollectionY
    
    def _stage1(self, query, BACKS, ranking, TOPK=9):
        model = self.modelL1

        with torch.inference_mode():
            backs = [self.CollectionY[(pid, sid)] for pid, sid in BACKS if (pid, sid) in self.CollectionY]
            backs = [query] + backs
            query = ' # '.join(backs)

            # print(query)
            # print(backs)
            passages = []
            actual_ranking = []

            for pid in ranking:
                actual_ranking.append(pid)
                psg = self.CollectionX[pid]
                psg = ' [MASK] '.join(psg)

                passages.append(psg)

            obj = self.tokenizer.process([query], passages, None)

            with self.amp.context():
                scores = model(obj.encoding.to(model.device)).float()

            pids = [[pid] * scores.size(1) for pid in actual_ranking]
            pids = flatten(pids)

            sids = [list(range(scores.size(1))) for pid in actual_ranking]
            sids = flatten(sids)

            scores = scores.view(-1)

            topk = scores.topk(min(TOPK, len(scores))).indices.tolist()
            topk_pids = [pids[idx] for idx in topk]
            topk_sids = [sids[idx] for idx in topk]

            preds = [(pid, sid) for pid, sid in zip(topk_pids, topk_sids)]

            pred_plus = BACKS + preds
            pred_plus = f7(list(map(tuple, pred_plus)))[:TOPK]

        return pred_plus
    
    def _stage2(self, query, preds):
        model = self.modelL2

        psgX = [self.CollectionY[(pid, sid)] for pid, sid in preds if (pid, sid) in self.CollectionY]
        psg = ' [MASK] '.join([''] + psgX)
        passages = [psg]
        # print(passages)

        obj = self.tokenizer.process([query], passages, None)

        with self.amp.context():
            scores = model(obj.encoding.to(model.device)).float()
            scores = scores.view(-1).tolist()

            preds = [(score, (pid, sid)) for (pid, sid), score in zip(preds, scores)]
            preds = sorted(preds, reverse=True)[:5]

            preds_L3x = [x for score, x in preds if score > min(0, preds[1][0] - 1e-10)] # Take at least 2!
            preds = [x for score, x in preds if score > 0]

            earliest_pids = f7([pid for pid, _ in preds_L3x])[:4]  # Take at most 4 docs.
            preds_L3x = [(pid, sid) for pid, sid in preds_L3x if pid in earliest_pids]

            assert len(preds_L3x) >= 2
            assert len(f7([pid for pid, _ in preds_L3x])) <= 4

        return preds, preds_L3x
