from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message, flatten
from colbert.modeling.base_colbert import BaseColBERT

import torch
import string

import os
import pathlib
from torch.utils.cpp_extension import load
import torch.distributed as dist

def get_rank():
    return dist.get_rank()

def get_world_size():
    return dist.get_world_size()

def get_default_group():
    return dist.group.WORLD





class ColBERT(BaseColBERT):
    """
        This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        super().__init__(name, colbert_config)
        self.use_gpu = colbert_config.total_visible_gpus > 0

        ColBERT.try_load_torch_extensions(self.use_gpu)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.loss_fn = torch.nn.CrossEntropyLoss()

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(f"Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims='return_mask')

        # Gather tensors from other GPUs
        # Q, D, D_mask = self.gather_tensors_from_other_gpus(Q, D, D_mask)
        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        # print('Q_duplicated', Q_duplicated.shape)
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss_new(Q, D, D_mask)
            # ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss

        return scores
    
    def compute_ib_loss_new(self, Q, D, D_mask):
        # Q: batch_size x q_len x dim
        # D: batch_size*n_docs x i_len x dim
        # D_mask: batch_size*n_docs x i_len x dim
        # 1 x batch_size*n_docs x i_len x dim matmul batch_size x 1 x q_len x dim
        # = batch_size x batch_size*n_docs x i_len x q_len

        scores = (D.float().unsqueeze(0) @ Q.float().permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze
        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)
        
        in_batch_scores = scores.reshape(Q.size(0), -1)
        # print('in_batch_scores', in_batch_scores.shape, in_batch_scores)

        batch_size = Q.shape[0]
        batch_size_with_pos_and_neg = D.shape[0]
        num_pos_and_neg = batch_size_with_pos_and_neg // batch_size
        num_pos = 1
        num_neg = num_pos_and_neg - num_pos
        
        # batch_size x dim  matmul  dim x (num_pos+num_neg)*batch_size  
        # -->  batch_size x (num_pos+num_neg)*batch_size
        in_batch_labels = torch.zeros(batch_size, batch_size_with_pos_and_neg).to(scores.device)
        step = num_pos_and_neg
        for i in range(batch_size):
            in_batch_labels[i, step*i] = 1
        # print('in_batch_labels', in_batch_labels)
        in_batch_labels = torch.argmax(in_batch_labels, dim=1)
        # print('in_batch_labels', in_batch_labels)
        
        loss = self.loss_fn(in_batch_scores, in_batch_labels)

        return loss

    def gather_tensors_from_other_gpus(self, query_embeddings, item_embeddings, item_mask):
        # print("get rank", get_rank())
        # print("get world size", get_world_size())
        # Gather embeddings from other GPUs
        n_nodes = get_world_size()
        if n_nodes == 1:
            return query_embeddings, item_embeddings, item_mask
        # Create placeholder to hold embeddings passed from other ranks
        global_query_embeddings_placeholder = [torch.zeros(*query_embeddings.shape, dtype=query_embeddings.dtype).to(query_embeddings.device) for _ in range(n_nodes)]
        global_item_embeddings_placeholder = [torch.zeros(*item_embeddings.shape, dtype=item_embeddings.dtype).to(item_embeddings.device) for _ in range(n_nodes)]
        global_item_mask_placeholder = [torch.zeros(*item_mask.shape, dtype=item_mask.dtype).to(item_mask.device) for _ in range(n_nodes)]
        dist.all_gather(global_query_embeddings_placeholder, query_embeddings.detach())
        dist.all_gather(global_item_embeddings_placeholder, item_embeddings.detach())
        dist.all_gather(global_item_mask_placeholder, item_mask.detach())

        global_query_embeddings = []
        global_item_embeddings = []
        global_item_mask = []
        # print(f"rank {get_rank()} global_query_embeddings", global_query_embeddings)
        # print(f"rank {get_rank()} global_item_embeddings", global_item_embeddings)
        # input()
        current_rank = get_rank()
        for rank_index, remote_q_embeddings in enumerate(global_query_embeddings_placeholder):
            # We append the embeddings from other GPUs if this embedding does not require gradients
            if rank_index != current_rank:
                global_query_embeddings.append(remote_q_embeddings)
            else:
                global_query_embeddings.append(query_embeddings)

        for rank_index, remote_item_embeddings in enumerate(global_item_embeddings_placeholder):
            # We append the embeddings from other GPUs if this embedding does not require gradients
            if rank_index != current_rank:
                global_item_embeddings.append(remote_item_embeddings)
            else:
                global_item_embeddings.append(item_embeddings)
        
        for rank_index, remote_item_mask in enumerate(global_item_mask_placeholder):
            # We append the embeddings from other GPUs if this embedding does not require gradients
            if rank_index != current_rank:
                global_item_mask.append(remote_item_mask)
            else:
                global_item_mask.append(item_mask)

        # Replace the previous variables with gathered tensors
        query_embeddings = torch.cat(global_query_embeddings)
        item_embeddings = torch.cat(global_item_embeddings)
        item_mask = torch.cat(global_item_mask)

        return query_embeddings, item_embeddings, item_mask


    def compute_ib_loss(self, Q, D, D_mask):
        # TODO: Organize the code below! Quite messy.
        scores = (D.float().unsqueeze(0) @ Q.float().permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze

        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)
        
        nway = self.colbert_config.nway
        all_except_self_negatives = [list(range(qidx*D.size(0), qidx*D.size(0) + nway*qidx+1)) +
                                     list(range(qidx*D.size(0) + nway * (qidx+1), qidx*D.size(0) + D.size(0)))
                                     for qidx in range(Q.size(0))]

        scores = scores[flatten(all_except_self_negatives)]
        scores = scores.view(Q.size(0), -1)  # D.size(0) - self.colbert_config.nway + 1)

        labels = torch.arange(0, Q.size(0), device=scores.device) * (self.colbert_config.nway)

        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D

    def score(self, Q, D_padded, D_mask):
        # assert self.colbert_config.similarity == 'cosine'

        if self.colbert_config.similarity == 'l2':
            assert self.colbert_config.interaction == 'colbert'
            return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config, use_gpu=self.use_gpu)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask



# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer

# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    # print('D_mask', D_mask.shape, D_mask)
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    # print('D_padding', D_padding.shape, D_padding)
    # print(D_padding[0].tolist())
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values
    # print("scores before reduce", scores.shape, scores)
    # print(scores[0].tolist())
    # print(scores[:, :32].sum(-1))
    # print(scores[:, 32:].sum(-1))
    assert config.interaction in ['colbert', 'flipr'], config.interaction

    if config.interaction == 'flipr':
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)

        return A + B

    return scores.sum(-1)



# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig(), use_gpu=False):
    """
        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.

        EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """
    # use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_packed(Q, D_packed, D_lengths, config=ColBERTConfig()):
    """
        Works with a single query only.
    """

    use_gpu = config.total_visible_gpus > 0

    if use_gpu:
        Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    if use_gpu or config.interaction == "flipr":
        scores_padded, scores_mask = StridedTensor(scores, D_lengths, use_gpu=use_gpu).as_padded_tensor()

        return colbert_score_reduce(scores_padded, scores_mask, config)
    else:
        return ColBERT.segmented_maxsim(scores, D_lengths)
