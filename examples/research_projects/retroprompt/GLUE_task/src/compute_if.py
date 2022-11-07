# part of codes from https://github.com/xszheng2020/memorization/blob/master/sst/compute_if.py
# -*- coding: utf-8 -*-
import enum
import os

import numpy as np
import pandas as pd


pd.set_option("max_colwidth", 256)
import pickle

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from contexttimer import Timer


def compute_s(trainer, model, v, train_data_loader, damp, scale, num_samples):

    last_estimate = list(v).copy()
    with tqdm(total=num_samples) as pbar:
        n = 0
        for i, inputs in enumerate(train_data_loader):
            inputs = trainer._prepare_inputs(inputs)
            labels = inputs["labels"].cuda()
            input_ids = inputs["input_ids"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            mask_pos = inputs["mask_pos"].cuda()

            n += len(input_ids)

            if trainer.args.use_demo:
                block_flag_for_demo = inputs["block_flag_for_demo"].cuda()
                demo_mask_features = inputs["demo_mask_features"].cuda()
            else:
                block_flag_for_demo, demo_mask_features = None, None
            ####
            this_estimate = compute_hessian_vector_products(
                model=model,
                vectors=last_estimate,
                labels=labels,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_pos=mask_pos,
                block_flag_for_demo=block_flag_for_demo,
                demo_mask_features=demo_mask_features,
            )
            # Recursively caclulate h_estimate
            # https://github.com/dedeswim/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_functions/hvp_grad.py#L118
            with torch.no_grad():
                new_estimate = [a + (1 - damp) * b - c / scale for a, b, c in zip(v, last_estimate, this_estimate)]
            ####
            pbar.update(len(input_ids))

            new_estimate_norm = new_estimate[0].norm().item()
            last_estimate_norm = last_estimate[0].norm().item()
            estimate_norm_diff = new_estimate_norm - last_estimate_norm
            pbar.set_description(f"{new_estimate_norm:.2f} | {estimate_norm_diff:.2f}")
            ####
            last_estimate = new_estimate

            if n > num_samples:  # should be i>=(num_samples-1) but does not matters
                break

    # References:
    # https://github.com/kohpangwei/influence-release/blob/master/influence/genericNeuralNet.py#L475
    # Do this for each iteration of estimation
    # Since we use one estimation, we put this at the end
    inverse_hvp = [X / scale for X in last_estimate]

    return inverse_hvp


def compute_hessian_vector_products(
    model, vectors, labels, input_ids, attention_mask, mask_pos, block_flag_for_demo, demo_mask_features,
):

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        mask_pos=mask_pos,
        block_flag_for_demo=block_flag_for_demo,
        demo_mask_features=demo_mask_features,
    )
    loss, logits = outputs

    model.zero_grad()
    grad_tuple = torch.autograd.grad(
        outputs=loss,
        inputs=[
            param
            for name, param in model.named_parameters()
            if param.requires_grad and "roberta.pooler.dense" not in name
        ],
        create_graph=True,
        allow_unused=True,
    )

    grad_grad_tuple = torch.autograd.grad(
        outputs=grad_tuple,
        inputs=[
            param
            for name, param in model.named_parameters()
            if param.requires_grad and "roberta.pooler.dense" not in name
        ],
        grad_outputs=vectors,
        only_inputs=True,
    )

    return grad_grad_tuple


def compute_memorize_score(trainer):
    # prepare works before eval
    train_dataloader = DataLoader(trainer.train_dataset, batch_size=1, shuffle=False, collate_fn=trainer.data_collator)
    dev_dataloader = trainer.get_eval_dataloader()

    model = trainer.model
    model.cuda()
    model.eval()

    # eval model performance on the train dataset
    predictions_list = []
    label_list = []
    # get demos
    if trainer.args.use_demo:
        train_dataloader_demo = trainer.get_train_dataloader()
        query_dataloader_demo = trainer.get_eval_dataloader()
        trainer.get_eval_mask_features(
            model, support_dataloader=train_dataloader_demo, query_dataloader=query_dataloader_demo, mode="support"
        )
        trainer.get_eval_mask_features(
            model, support_dataloader=None, query_dataloader=query_dataloader_demo, mode="query"
        )

    for inputs in dev_dataloader:
        loss, logits, labels = trainer.prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=None)
        predictions = logits.detach().cpu().numpy()
        predictions_list.append(predictions)
        label_list.extend(labels.detach().cpu().view(-1).tolist())

    predictions = np.vstack(predictions_list)
    predictions = np.argmax(predictions, axis=-1)

    label_list = np.array(label_list)

    print(classification_report(label_list, predictions, digits=4))

    # # cal mem scores

    start = trainer.args.START
    length = trainer.args.LENGTH

    output_collections = []

    # get demos
    if trainer.args.use_demo:
        dataloader_demo = trainer.get_eval_dataloader(trainer.train_dataset)
        trainer.get_train_mask_features(model, dataloader_demo, mode="support")
        trainer.get_train_mask_features(model, dataloader_demo, mode="query")

    for idx, inputs in enumerate(train_dataloader):
        inputs = trainer._prepare_inputs(inputs)

        raw_words = trainer.train_dataset.query_examples[idx].text_a
        raw_labels = trainer.train_dataset.query_examples[idx].label

        if idx < start:
            continue
        if idx >= start + length:
            break

        # raw input
        z_label = inputs["labels"].cuda()
        z_input_ids = inputs["input_ids"].cuda()
        z_attention_mask = inputs["attention_mask"].cuda()
        z_mask_pos = inputs["mask_pos"].cuda()
        if trainer.args.use_demo:
            z_block_flag_for_demo = inputs["block_flag_for_demo"].cuda()
            z_demo_mask_features = inputs["demo_mask_features"].cuda()
        else:
            z_block_flag_for_demo = None
            z_demo_mask_features = None
        tokens = model.tokenizer.convert_ids_to_tokens(z_input_ids[0].cpu().numpy())

        # raw

        outputs = model(
            input_ids=z_input_ids,
            attention_mask=z_attention_mask,
            mask_pos=z_mask_pos,
            block_flag_for_demo=z_block_flag_for_demo,
            demo_mask_features=z_demo_mask_features,
            return_output=False,
        )[0]

        prob = F.softmax(outputs, dim=-1)

        prediction = torch.argmax(prob, dim=1)

        prob_gt = torch.gather(prob, 1, z_label.unsqueeze(1))

        model.zero_grad()

        v = torch.autograd.grad(
            outputs=prob_gt,
            inputs=[
                param
                for name, param in model.named_parameters()
                if param.requires_grad and "roberta.pooler.dense" not in name
            ],
            create_graph=False,
        )
        ####

        for repetition in range(4):
            with Timer() as timer:
                train_dataloader = DataLoader(
                    trainer.train_dataset, batch_size=4, num_workers=0, shuffle=True, collate_fn=trainer.data_collator,
                )
                s = compute_s(
                    trainer=trainer,
                    model=model,
                    v=v,
                    train_data_loader=train_dataloader,
                    damp=5e-3,
                    scale=1e4,
                    num_samples=500,
                )

                time_elapsed = timer.elapsed

            outputs = model(
                input_ids=z_input_ids,
                labels=z_label,
                attention_mask=z_attention_mask,
                mask_pos=z_mask_pos,
                block_flag_for_demo=z_block_flag_for_demo,
                demo_mask_features=z_demo_mask_features,
                return_output=False,
            )
            loss = outputs[0]

            model.zero_grad()

            grad_tuple_ = torch.autograd.grad(
                outputs=loss,
                inputs=[
                    param
                    for name, param in model.named_parameters()
                    if param.requires_grad and "roberta.pooler.dense" not in name
                ],
                create_graph=True,
            )
            ####
            ####
            influence = [-torch.sum(x * y) for x, y in zip(s, grad_tuple_)]
            influence = sum(influence).item()

            outputs = {
                "index": idx,
                "sentence": raw_words,
                "label": raw_labels,
                "prob": prob.detach().cpu().numpy()[0],
                "prediction": prediction.detach().cpu().numpy()[0],
                "influence": influence,
                "tokens": tokens,
                "repetition": repetition,
                "time_elapsed": time_elapsed,
            }

            print(outputs["index"])
            print(outputs["sentence"])
            print(outputs["label"], outputs["prob"], outputs["prediction"])
            print("influence: ", outputs["influence"])
            print("tokens: ", outputs["tokens"])
            print("repetition", repetition)  #

            output_collections.append(outputs)
            ####
            break
        del v, s, grad_tuple_
        torch.cuda.empty_cache()
    # +
    filename = os.path.join("ckpt", "score_mem/{}.pkl".format(start))
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as handle:
        pickle.dump(output_collections, handle)
    # -
