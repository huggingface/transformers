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
        q = self.BERT(**W_query)
        S = self.BERT(**W_supports)

        # reshape from (batch_size, 384, 784) to (batch_size, 1, 384, 784)
        q = q.view(q.shape[0], -1, q.shape[1], q.shape[2])

        # reshape from (batch_size*n_exaples_per_entity, 384, 784) to (batch_size, n_exaples_per_entity, 384, 784)
        S = S.view(q.shape[0], -1, S.shape[1], S.shape[2])

        s_start = S[(W_supports["input_ids"] == 30522).view(S.shape[:3])].view(S.shape[0], -1, 1, S.shape[-1])
        s_end = S[(W_supports["input_ids"] == 30523).view(S.shape[:3])].view(S.shape[0], -1, 1, S.shape[-1])

        p_start = torch.sum(torch.einsum("bitf,bejf->bet", q, s_start), dim=1)
        p_end = torch.sum(torch.einsum("bitf,bejf->bet", q, s_end), dim=1)

        p_start = p_start.softmax(dim=1)
        p_end = p_end.softmax(dim=1)

        return p_start, p_end
