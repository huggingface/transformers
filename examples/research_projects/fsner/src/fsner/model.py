import torch

from transformers import AutoModel


class FSNERModel(torch.nn.Module):
    """
    The FSNER model implements a few-shot named entity recognition method from the paper `Example-Based Named Entity Recognition <https://arxiv.org/abs/2008.10570>`__ by
    Morteza Ziyadi, Yuting Sun, Abhishek Goswami, Jade Huang, Weizhu Chen. To identify entity spans in a new domain, it
    uses a train-free few-shot learning approach inspired by question-answering.
    """

    def __init__(self, pretrained_model_name_or_path="sayef/fsner-bert-base-uncased"):
        super(FSNERModel, self).__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path, return_dict=True)
        self.cos = torch.nn.CosineSimilarity(3, 1e-08)
        self.softmax = torch.nn.Softmax(dim=1)

    def BERT(self, **inputs):
        return self.bert(**inputs).last_hidden_state

    def VectorSum(self, token_embeddings):
        return token_embeddings.sum(2, keepdim=True)

    def Atten(self, q_rep, S_rep, T=1):
        return self.softmax(T * self.cos(q_rep, S_rep))

    def forward(self, W_query, W_supports):
        """
        Find scores of each token being start and end token for an entity.
        Args:
            W_query (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of query sequence tokens in the vocabulary.
            W_supports (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of support sequence tokens in the vocabulary.
        Returns:
            p_start (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): Scores of each token as
            being start token of an entity
            p_end (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): Scores of each token as
            being end token of an entity
        """

        support_sizes = W_supports["sizes"].tolist()
        start_token_id = W_supports["start_token_id"].item()
        end_token_id = W_supports["end_token_id"].item()

        del W_supports["sizes"]
        del W_supports["start_token_id"]
        del W_supports["end_token_id"]

        q = self.BERT(**W_query)
        S = self.BERT(**W_supports)

        p_starts = None
        p_ends = None

        start_token_masks = W_supports["input_ids"] == start_token_id
        end_token_masks = W_supports["input_ids"] == end_token_id

        for i, size in enumerate(support_sizes):
            if i == 0:
                s = 0
            else:
                s = support_sizes[i - 1]

            s_start = S[s : s + size][start_token_masks[s : s + size]]
            s_end = S[s : s + size][end_token_masks[s : s + size]]

            p_start = torch.matmul(q[i], s_start.T).sum(1).softmax(0)
            p_end = torch.matmul(q[i], s_end.T).sum(1).softmax(0)

            if p_starts is not None:
                p_starts = torch.vstack((p_starts, p_start))
                p_ends = torch.vstack((p_ends, p_end))
            else:
                p_starts = p_start
                p_ends = p_end

        return p_starts, p_ends
