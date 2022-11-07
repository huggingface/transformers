import logging

import numpy as np
import torch
import torch.nn as nn

import faiss
from transformers import BertPreTrainedModel

from .modeling_roberta import RobertaLMHead, RobertaModel
from .utils import knnLoss


logger = logging.getLogger(__name__)


class RobertaForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.total_label_tokens = config.total_label_tokens

        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.tokenizer = None

        # For regression
        self.lb = None
        self.ub = None

        # For saving [MASK]
        self.cnt_batch = 0  # record current batch
        self.maskid2labelid = {}
        d, measure = config.hidden_size, faiss.METRIC_L2
        self.mask_features = faiss.IndexFlatIP(d)
        self.total_features = []

        self.loss_fct = knnLoss()

        # For semi
        self.semi_logits = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        block_flag_for_demo=None,
        labels=None,
        return_output=False,
        reduction="mean",
        only_mask_output=False,
        save_mask=False,
        demo_mask_features=None,
    ):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            block_flag_for_demo=block_flag_for_demo,
            demo_mask_features=demo_mask_features,
        )

        sequence_output, _ = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # early exit when calculate contrastive representation
        if only_mask_output:
            outputs = sequence_mask_output
            return outputs

        loss = None
        prediction_mask_scores, lm_feature = self.lm_head(sequence_mask_output)
        # lm_feature or sequence_mask_output
        if save_mask:  # save mask features
            bsz = input_ids.size(0)
            self.total_features.append(lm_feature.cpu().detach())
            for idx, label in zip(range(bsz), labels):
                self.maskid2labelid[idx + self.cnt_batch] = label.cpu().detach()
            self.cnt_batch += bsz
            return None

        knn_logits, combine_logits = None, None
        logits = prediction_mask_scores[:, self.label_word_list]
        if len(logits.shape) == 3:  # multi-token label:(4, 14, 2) bsz, n_label, n_per_label
            logits = logits.sum(-1)
        # self.semi_logits.extend(logits.cpu().detach().tolist())

        if self.model_args.knn_mode:
            bsz = input_ids.size(0)
            mask_embedding = np.array(lm_feature.cpu().detach(), dtype=np.float32)
            topk = self.model_args.knn_topk
            D, I = self.mask_features.search(mask_embedding, topk)
            D = torch.from_numpy(D).to(input_ids.device)
            knn_logits = torch.full((bsz, self.num_labels), 0.0).to(input_ids.device)
            for i in range(bsz):
                """'like knnlm"""
                soft_knn_i = torch.softmax(D[i], dim=-1)  # 1 x topk
                for j in range(topk):
                    knn_logits[i][self.maskid2labelid[I[i][j]]] += soft_knn_i[j]

            mask_logits = torch.softmax(logits, dim=-1)
            combine_logits = combine_knn_and_vocab_probs(knn_logits, mask_logits, coeff=self.model_args.knn_lambda)
            # For semi
            # self.semi_logits.extend(logits.cpu().detach().tolist())

        loss = None
        if labels is not None:
            # knn for train
            if self.model_args.train_with_knn and knn_logits is not None:
                loss = self.loss_fct(logits, knn_logits, labels.view(-1), self.model_args.beta)
            else:
                loss_fct = nn.CrossEntropyLoss(reduction=reduction)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if combine_logits is not None and not self.model_args.only_train_knn:
            logits = combine_logits

        if return_output:
            outputs = (logits, sequence_mask_output)
        else:
            outputs = (logits,)
        return ((loss,) + outputs) if loss is not None else outputs


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff=0.5):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob
