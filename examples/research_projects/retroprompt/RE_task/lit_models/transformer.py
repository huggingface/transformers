import json
from copy import deepcopy
from functools import partial

# Hide lines below until Lab 5
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import faiss
from transformers.optimization import get_linear_schedule_with_warmup

from .base import BaseLitModel
from .util import f1_eval, f1_score, knnLoss


# Hide lines above until Lab 5



def mask_hook(grad_input, st, ed):
    mask = torch.zeros((grad_input.shape[0], 1)).type_as(grad_input)
    mask[st:ed] += 1.0
    # for the speaker unused token12
    mask[1:3] += 1.0
    return grad_input * mask


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """

    def __init__(self, model, args, tokenizer, datamodule):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        self.datamodule = datamodule

        with open(f"{args.data_dir}/rel2id.json", "r") as file:
            rel2id = json.load(file)
        self.rel2id = rel2id
        self.task_name = None

        if "tacrev" in args.data_dir:
            self.na_label = "no_relation"
            self.task_name = "tacrev"
        elif "tacred" in args.data_dir:
            self.na_label = "NA"
            self.task_name = "tacred"
        elif "semeval" in args.data_dir:
            self.na_label = "Other"
            self.task_name = "semeval"

        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        if args.train_with_knn:
            self.loss_fn = knnLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = (
            f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        )
        self.best_f1 = 0
        self.t_lambda = args.t_lambda

        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)["input_ids"][0]
        self.tokenizer = tokenizer

        self._init_label_word()

        # For saving [MASK]
        self.cnt_batch = 0  # record current batch
        self.maskid2labelid = {}
        d, measure = self.model.config.hidden_size, faiss.METRIC_L2
        if self.task_name == "semeval":  # semeval: compute inner product
            self.mask_features = faiss.IndexFlatIP(d)
        else:  # other: compute L2 distance
            self.mask_features = faiss.IndexFlatL2(d)
        self.total_features = []

    def _init_label_word(self,):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[1]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"./dataset/{model_name_or_path}_{dataset_name}.pt"
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        elif "semeval" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)

        self.num_labels = len(label_word_idx)

        self.model.resize_token_embeddings(len(self.tokenizer))

        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            continous_label_word = [
                a[0]
                for a in self.tokenizer(
                    [f"[class{i}]" for i in range(1, self.num_labels + 1)], add_special_tokens=False
                )["input_ids"]
            ]

            # for abaltion study
            if self.args.init_answer_words:
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(
                            word_embeddings.weight[idx], dim=0
                        )

            if self.args.init_type_words:
                so_word = [a[0] for a in self.tokenizer(["[obj]", "[sub]"], add_special_tokens=False)["input_ids"]]
                meaning_word = [
                    a[0]
                    for a in self.tokenizer(
                        ["person", "organization", "location", "date", "country"], add_special_tokens=False
                    )["input_ids"]
                ]

                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)

        self.word2label = continous_label_word  # a continous list

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, save_mask=False):  # pylint: disable=unused-argument
        if len(batch) == 4:
            input_ids, attention_mask, labels, so = batch
            block_flag_for_demo, demo_mask_features = None, None
        else:
            input_ids, attention_mask, labels, so, block_flag_for_demo, demo_mask_features = batch
        result, lm_features = self.model(
            input_ids,
            attention_mask,
            return_dict=True,
            output_hidden_states=True,
            block_flag_for_demo=block_flag_for_demo,
            demo_mask_features=demo_mask_features,
        )
        if save_mask:  # save mask features
            bsz = input_ids.size(0)
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            lm_features = lm_features[torch.arange(bsz), mask_idx]
            self.total_features.append(lm_features.cpu().detach())
            for idx, label in zip(range(bsz), labels):
                self.maskid2labelid[idx + self.cnt_batch] = label.cpu().detach()
            self.cnt_batch += bsz
            return None

        logits = result.logits
        logits, knn_logits, combine_logits = self.pvp(logits, input_ids, lm_features=lm_features)

        if self.args.train_with_knn:
            loss = self.loss_fn(logits, knn_logits, labels, self.args.alpha)
        else:
            loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return loss

    def on_train_epoch_start(self):
        if self.args.use_demo:
            # for demo
            data_loader = DataLoader(
                self.datamodule.data_train,
                batch_size=self.datamodule.batch_size,
                shuffle=False,
                num_workers=self.datamodule.num_workers,
                pin_memory=True,
            )
            self.get_train_mask_features(data_loader)
        if self.args.train_with_knn:
            self.clear_mask_features()
            data_loader = DataLoader(
                self.datamodule.data_train,
                batch_size=self.datamodule.batch_size,
                shuffle=False,
                num_workers=self.datamodule.num_workers,
                pin_memory=True,
            )
            self.get_mask_features_for_knn(data_loader)

    def on_training_epoch_end(self):
        self.clear_mask_features()

    def training_epoch_end(self, outputs) -> None:
        self.clear_mask_features()

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if len(batch) == 4:
            input_ids, attention_mask, labels, so = batch
            block_flag_for_demo, demo_mask_features = None, None
        else:
            input_ids, attention_mask, labels, so, block_flag_for_demo, demo_mask_features = batch
        output, lm_features = self.model(
            input_ids,
            attention_mask,
            return_dict=True,
            output_hidden_states=True,
            block_flag_for_demo=block_flag_for_demo,
            demo_mask_features=demo_mask_features,
        )

        logits = output.logits
        logits, knn_logits, combine_logits = self.pvp(logits, input_ids, lm_features=lm_features)
        if combine_logits is not None:
            logits = combine_logits
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}

    def on_validation_epoch_start(self):
        if self.args.use_demo:
            train_dataloader = DataLoader(
                self.datamodule.data_train,
                batch_size=self.datamodule.batch_size,
                shuffle=False,
                num_workers=self.datamodule.num_workers,
                pin_memory=True,
            )
            query_dataloader = self.val_dataloader()
            self.get_eval_mask_features(train_dataloader, query_dataloader)
        if self.args.only_train_knn:
            self.args.knn_mode = False
        if self.args.knn_mode:
            self.clear_mask_features()
            dataloader = DataLoader(
                self.datamodule.data_train,
                batch_size=self.datamodule.batch_size,
                shuffle=False,
                num_workers=self.datamodule.num_workers,
                pin_memory=True,
            )
            self.get_mask_features_for_knn(dataloader)

    def on_validation_epoch_end(self):
        self.clear_mask_features()
        if self.args.only_train_knn:
            self.args.knn_mode = True

    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)["f1"]
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        self.clear_mask_features()

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        if len(batch) == 4:
            input_ids, attention_mask, labels, so = batch
            block_flag_for_demo, demo_mask_features = None, None
        else:
            input_ids, attention_mask, labels, so, block_flag_for_demo, demo_mask_features = batch
        output, lm_features = self.model(
            input_ids,
            attention_mask,
            return_dict=True,
            output_hidden_states=True,
            block_flag_for_demo=block_flag_for_demo,
            demo_mask_features=demo_mask_features,
        )
        logits = output.logits
        logits, knn_logits, combine_logits = self.pvp(logits, input_ids, lm_features=lm_features)
        if combine_logits is not None:
            logits = combine_logits
        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def on_test_epoch_start(self):
        if self.args.use_demo:
            train_dataloader = DataLoader(
                self.datamodule.data_train,
                batch_size=self.datamodule.batch_size,
                shuffle=False,
                num_workers=self.datamodule.num_workers,
                pin_memory=True,
            )
            query_dataloader = self.test_dataloader()
            self.get_eval_mask_features(train_dataloader, query_dataloader)
        if self.args.only_train_knn:
            self.args.knn_mode = False
        if self.args.knn_mode:
            self.clear_mask_features()
            dataloader = DataLoader(
                self.datamodule.data_train,
                batch_size=self.datamodule.batch_size,
                shuffle=False,
                num_workers=self.datamodule.num_workers,
                pin_memory=True,
            )
            self.get_mask_features_for_knn(dataloader)

    def on_test_epoch_end(self):
        self.clear_mask_features()
        if self.args.only_train_knn:
            self.args.knn_mode = True

    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)["f1"]
        self.log("Test/f1", f1)

    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser

    def pvp(self, logits, input_ids, lm_features=None):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        lm_features = lm_features[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, ("only one mask in sequence!", mask_idx.shape, bs)
        logits = mask_output[:, self.word2label]
        knn_logits, combine_logits = None, None

        # for knn
        if self.args.knn_mode:
            bsz = input_ids.size(0)
            mask_embedding = np.array(lm_features.cpu().detach(), dtype=np.float32)
            topk = self.args.knn_topk
            D, I = self.mask_features.search(mask_embedding, topk)
            D = torch.from_numpy(D).to(input_ids.device)
            # filter no_relation
            for i in range(bsz):
                for j in range(topk):
                    if (
                        self.maskid2labelid[I[i][j]] == self.rel2id[self.na_label]
                    ):  # semeval: Other; tacrev: no_relation; tacred: NA
                        D[i][j] = -1000
            knn_logits = torch.full((bsz, self.num_labels), 0.0).to(input_ids.device)
            for i in range(bsz):
                """'like knnlm"""
                if self.task_name == "semeval":
                    soft_knn_i = torch.softmax(D[i], dim=-1)  # 1 x topk
                else:
                    soft_knn_i = torch.softmax(-D[i] / self.args.temp, dim=-1)  # 1 x topk
                for j in range(topk):
                    knn_logits[i][self.maskid2labelid[I[i][j]]] += soft_knn_i[j]

            mask_logits = torch.softmax(logits, dim=-1)
            combine_logits = combine_knn_and_vocab_probs(knn_logits, mask_logits, coeff=self.args.knn_lambda)

        return logits, knn_logits, combine_logits

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps:
            parameters = self.model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {
                "params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0},
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1,},  # or 'epoch'
        }

    def get_mask_features_for_knn(self, dataloader):
        dataloader.dataset.demo_mode = "save"
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc="KNN"):
                inputs = tuple(input.to(self.device) for input in inputs)
                self.training_step(inputs, batch_idx=None, save_mask=True)
            self.total_features = np.concatenate(self.total_features, axis=0)
            self.mask_features.add(self.total_features)
        dataloader.dataset.demo_mode = "get"

    def get_train_mask_features(self, data_loader):
        data_loader.dataset.demo_mode = "save"
        with torch.no_grad():
            for inputs in tqdm(data_loader, desc="TRAIN MASK"):
                inputs = tuple(input.to(self.device) for input in inputs)
                self.training_step(inputs, batch_idx=None, save_mask=True)
            self.total_features = np.concatenate(self.total_features, axis=0)
            self.mask_features.add(self.total_features)
            data_loader.dataset.support_features = deepcopy(self.mask_features)
            data_loader.dataset.support_labelids = deepcopy(self.maskid2labelid)

            data_loader.dataset.query_features = deepcopy(self.mask_features)
            data_loader.dataset.query_labelids = deepcopy(self.maskid2labelid)
            data_loader.dataset.get_demos()

        self.clear_mask_features()
        data_loader.dataset.demo_mode = "get"

    def get_eval_mask_features(self, support_dataloader=None, query_dataloader=None):
        support_dataloader.dataset.demo_mode = "save"
        query_dataloader.dataset.demo_mode = "save"
        with torch.no_grad():
            # support
            for inputs in tqdm(support_dataloader, desc="train_support", total=len(support_dataloader)):
                inputs = tuple(input.to(self.device) for input in inputs)
                self.training_step(inputs, batch_idx=None, save_mask=True)

            self.total_features = np.concatenate(self.total_features, axis=0)
            self.mask_features.add(self.total_features)

            query_dataloader.dataset.support_features = deepcopy(self.mask_features)
            query_dataloader.dataset.support_labelids = deepcopy(self.maskid2labelid)
            self.clear_mask_features()
            # query
            for inputs in tqdm(query_dataloader, desc="train_query", total=len(query_dataloader)):
                inputs = tuple(input.to(self.device) for input in inputs)
                self.training_step(inputs, batch_idx=None, save_mask=True)

            self.total_features = np.concatenate(self.total_features, axis=0)
            self.mask_features.add(self.total_features)
            query_dataloader.dataset.query_features = deepcopy(self.mask_features)
            query_dataloader.dataset.query_labelids = deepcopy(self.maskid2labelid)
            query_dataloader.dataset.get_demos()
            self.clear_mask_features()
        support_dataloader.dataset.demo_mode = "get"
        query_dataloader.dataset.demo_mode = "get"

    def clear_mask_features(self):
        self.total_features = []
        self.mask_features = faiss.IndexFlatL2(self.model.config.hidden_size)
        self.maskid2labelid = {}
        self.cnt_batch = 0


def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff=0.5):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob
