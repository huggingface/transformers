import torch
import torch.nn as nn

from transformers import ElectraPreTrainedModel, ElectraModel


class ElectraReader(ElectraPreTrainedModel):
    def __init__(self, config, learn_labels=False):
        super(ElectraReader, self).__init__(config)

        self.electra = ElectraModel(config)

        self.relevance = nn.Linear(config.hidden_size, 1)

        if learn_labels:
            self.linear = nn.Linear(config.hidden_size, 2)
        else:
            self.linear = nn.Linear(config.hidden_size, 1)

        self.init_weights()

        self.learn_labels = learn_labels

    def forward(self, encoding):
        outputs = self.electra(encoding.input_ids,
                               attention_mask=encoding.attention_mask,
                               token_type_ids=encoding.token_type_ids)[0]

        scores = self.linear(outputs)

        if self.learn_labels:
            scores = scores[:, 0].squeeze(1)
        else:
            scores = scores.squeeze(-1)
            candidates = (encoding.input_ids == 103)
            scores = self._mask_2d_index(scores, candidates)

        return scores

    def _mask_2d_index(self, scores, mask):
        bsize, maxlen = scores.size()
        bsize_, maxlen_ = mask.size()

        assert bsize == bsize_, (scores.size(), mask.size())
        assert maxlen == maxlen_, (scores.size(), mask.size())

        # Get flat scores corresponding to the True mask positions, with -inf at the end
        flat_scores = scores[mask]
        flat_scores = torch.cat((flat_scores, torch.ones(1, device=self.device) * float('-inf')))

        # Get 2D indexes
        rowidxs, nnzs = torch.unique(torch.nonzero(mask, as_tuple=False)[:, 0], return_counts=True)
        max_nnzs = nnzs.max().item()

        rows = [[-1] * max_nnzs for _ in range(bsize)]
        offset = 0
        for rowidx, nnz in zip(rowidxs.tolist(), nnzs.tolist()):
            rows[rowidx] = [offset + i for i in range(nnz)]
            rows[rowidx] += [-1] * (max_nnzs - len(rows[rowidx]))
            offset += nnz

        indexes = torch.tensor(rows).to(self.device)

        # Index with the 2D indexes
        scores_2d = flat_scores[indexes]

        return scores_2d

    def _2d_index(self, embeddings, positions):
        bsize, maxlen, hdim = embeddings.size()
        bsize_, max_out = positions.size()

        assert bsize == bsize_
        assert positions.max() < maxlen

        embeddings = embeddings.view(bsize * maxlen, hdim)
        positions = positions + torch.arange(bsize, device=positions.device).unsqueeze(-1) * maxlen

        return embeddings[positions]
