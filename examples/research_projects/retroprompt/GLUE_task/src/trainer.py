import gc
import logging
import os
import pickle
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from tqdm import tqdm, trange

import faiss
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import TrainOutput


logger = logging.getLogger(__name__)


def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


class Trainer(transformers.Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {"params": [p for n, p in params.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def train(self, model_path=None, dev_objective=None):
        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        train_dataloader = self.get_train_dataloader()

        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        model = self.model

        # without regard to fp16, tpu or multi-gpu
        # in view of 'gradient_accumulation'
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
        if self.args.use_demo:
            dataloader = self.get_eval_dataloader(self.train_dataset)
            # self.get_train_mask_features(model, dataloader, mode='support')
            # self.get_train_mask_features(model, dataloader, mode='query')
            self.get_train_mask_features(model, dataloader)

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
            # init for knn saveing mask features
            if self.args.train_with_knn:
                self.clear_mask_features()
                knn_dataloader = self.get_eval_dataloader(self.train_dataset)
                self.get_mask_features_for_knn(self.model, knn_dataloader)

            for step, inputs in enumerate(epoch_iterator):
                tr_loss += self.training_step(model, inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps and (step + 1) == len(epoch_iterator)
                ):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["norm"] = norm.item()
                        logs["learning_rate"] = scheduler.get_last_lr()[0]
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    metrics = None
                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        output = self.evaluate()
                        metrics = output.metrics
                        objective = self.dev_objective(metrics)
                        if objective > self.objective:
                            logger.info("Best dev result: {}".format(objective))
                            self.objective = objective
                            self.save_model(self.args.output_dir)
                        if self.args.use_demo:
                            dataloader = self.get_eval_dataloader(self.train_dataset)
                            # self.get_train_mask_features(model, dataloader, mode='support')
                            # self.get_train_mask_features(model, dataloader, mode='query')
                            self.get_train_mask_features(model, dataloader)
                        if self.args.train_with_knn:
                            self.clear_mask_features()
                            knn_dataloader = self.get_eval_dataloader(self.train_dataset)
                            self.get_mask_features_for_knn(self.model, knn_dataloader)

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

        return TrainOutput(self.global_step, tr_loss / self.global_step, {"metric": self.objective})

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.args.only_train_knn:
            self.model.model_args.knn_mode = False
        # for getting mask features
        train_dataloader = self.get_train_dataloader()
        query_dataloader = self.get_eval_dataloader(eval_dataset)
        # transfer learning: use source domain datastore
        if self.args.use_source_datastore:
            logger.info("Loading source domain datastore from {}/".format(self.args.ckpt_dir))
            mask_features = faiss.read_index(os.path.join(self.args.ckpt_dir, "support_features.index"))
            maskid2labelid = pickle.load(open(os.path.join(self.args.ckpt_dir, "support_labelids.pkl"), "rb"))
            logger.info("Num of mask_features is {}".format(mask_features.ntotal))
            query_dataloader.dataset.support_features = deepcopy(mask_features)
            query_dataloader.dataset.support_labelids = deepcopy(maskid2labelid)
            self.get_eval_mask_features(
                self.model, support_dataloader=None, query_dataloader=query_dataloader, mode="query"
            )
            if self.model.model_args.knn_mode:
                self.model.mask_features = deepcopy(mask_features)
                self.model.maskid2labelid = deepcopy(maskid2labelid)
        else:
            self.clear_mask_features()
            if self.args.use_demo:
                # self.get_eval_mask_features(self.model, support_dataloader=train_dataloader, query_dataloader=query_dataloader, mode='support')
                # self.get_eval_mask_features(self.model, support_dataloader=None, query_dataloader=query_dataloader, mode='query')
                self.get_eval_mask_features(
                    self.model, support_dataloader=train_dataloader, query_dataloader=query_dataloader
                )
            if self.model.model_args.knn_mode:
                train_dataloader = self.get_train_dataloader()
                self.get_mask_features_for_knn(self.model, train_dataloader)
        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)

        self.model.model_args.knn_mode = self.args.knn_mode
        if self.model.model_args.knn_mode:
            self.clear_mask_features()

        return output

    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def training_step(self, model: nn.Module, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs, return_seq_output=True)

        if self.args.gradient_accumulation_steps > 1:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def compute_loss(
        self, model, inputs, inputs2=None, return_outputs=False, return_seq_output=False, target_model=None
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs, return_output=return_seq_output)

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """

        if "v2" in inputs:
            inputs1 = inputs["v1"]
            inputs2 = inputs["v2"]
        else:
            inputs1 = inputs
            inputs2 = None

        for k, v in inputs1.items():
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                inputs1[k] = v.to(**kwargs)

        if inputs2 is not None:
            for k, v in inputs2.items():
                if isinstance(v, torch.Tensor):
                    kwargs = dict(device=self.args.device)
                    inputs2[k] = v.to(**kwargs)

        if inputs2 is not None:
            return inputs1, inputs2
        else:
            return inputs1

    def get_train_mask_features(self, model, dataloader=None):
        dataloader.dataset.demo_mode = "save"
        with torch.no_grad():
            model.eval()
            for inputs in tqdm(dataloader, desc="MASK_TRAIN", total=len(dataloader)):
                inputs = self._prepare_inputs(inputs)
                model(**inputs, save_mask=True)
            model.total_features = np.concatenate(model.total_features, axis=0)
            model.mask_features.add(model.total_features)

            dataloader.dataset.support_features = deepcopy(model.mask_features)
            dataloader.dataset.support_labelids = deepcopy(model.maskid2labelid)

            dataloader.dataset.query_features = deepcopy(model.mask_features)
            dataloader.dataset.query_labelids = deepcopy(model.maskid2labelid)
            dataloader.dataset.get_demos()

        self.clear_mask_features()
        dataloader.dataset.demo_mode = "get"

    def get_train_mask_features_(self, model, dataloader=None, mode=None):
        dataloader.dataset.demo_mode = "save"
        with torch.no_grad():
            model.eval()
            for inputs in tqdm(dataloader, desc="MASK_TRAIN", total=len(dataloader)):
                inputs = self._prepare_inputs(inputs)
                model(**inputs, save_mask=True)
            model.total_features = np.concatenate(model.total_features, axis=0)
            model.mask_features.add(model.total_features)

            if mode == "support":
                dataloader.dataset.support_features = deepcopy(model.mask_features)
                dataloader.dataset.support_labelids = deepcopy(model.maskid2labelid)
            elif mode == "query":
                dataloader.dataset.query_features = deepcopy(model.mask_features)
                dataloader.dataset.query_labelids = deepcopy(model.maskid2labelid)
                dataloader.dataset.get_demos()
            self.clear_mask_features()
        dataloader.dataset.demo_mode = "get"

    def get_eval_mask_features(self, model, support_dataloader=None, query_dataloader=None):
        support_dataloader.dataset.demo_mode = "save"
        query_dataloader.dataset.demo_mode = "save"
        with torch.no_grad():
            # support
            for inputs in tqdm(support_dataloader, desc="eval_support", total=len(support_dataloader)):
                model.eval()
                inputs = self._prepare_inputs(inputs)
                model(**inputs, save_mask=True)

            model.total_features = np.concatenate(model.total_features, axis=0)
            model.mask_features.add(model.total_features)

            query_dataloader.dataset.support_features = deepcopy(model.mask_features)
            query_dataloader.dataset.support_labelids = deepcopy(model.maskid2labelid)
            self.clear_mask_features()
            # query
            for inputs in tqdm(query_dataloader, desc="eval_query", total=len(query_dataloader)):
                model.eval()
                inputs = self._prepare_inputs(inputs)
                model(**inputs, save_mask=True)

            model.total_features = np.concatenate(model.total_features, axis=0)
            model.mask_features.add(model.total_features)

            query_dataloader.dataset.query_features = deepcopy(model.mask_features)
            query_dataloader.dataset.query_labelids = deepcopy(model.maskid2labelid)
            query_dataloader.dataset.get_demos()
            self.clear_mask_features()
        support_dataloader.dataset.demo_mode = "get"
        query_dataloader.dataset.demo_mode = "get"

    def get_eval_mask_features_(self, model, support_dataloader=None, query_dataloader=None, mode=None):
        dataloader = support_dataloader if support_dataloader is not None else query_dataloader
        dataloader.dataset.demo_mode = "save"
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc="MASK_EVAL", total=len(dataloader)):
                model.eval()
                inputs = self._prepare_inputs(inputs)
                model(**inputs, save_mask=True)

            model.total_features = np.concatenate(model.total_features, axis=0)
            model.mask_features.add(model.total_features)

            if mode == "support":
                query_dataloader.dataset.support_features = deepcopy(model.mask_features)
                query_dataloader.dataset.support_labelids = deepcopy(model.maskid2labelid)
            elif mode == "query":
                query_dataloader.dataset.query_features = deepcopy(model.mask_features)
                query_dataloader.dataset.query_labelids = deepcopy(model.maskid2labelid)
                query_dataloader.dataset.get_demos()
            self.clear_mask_features()
        dataloader.dataset.demo_mode = "get"

    def get_mask_features_for_knn(self, model, dataloader):
        dataloader.dataset.demo_mode = "save"
        with torch.no_grad():
            for inputs in dataloader:
                model.eval()
                inputs = self._prepare_inputs(inputs)
                model(**inputs, save_mask=True)

            model.total_features = np.concatenate(model.total_features, axis=0)
            model.mask_features.add(model.total_features)
        dataloader.dataset.demo_mode = "get"

    def clear_mask_features(self):
        self.model.total_features = []
        self.model.mask_features = faiss.IndexFlatIP(self.model.config.hidden_size)
        self.model.maskid2labelid = {}
        self.model.cnt_batch = 0
