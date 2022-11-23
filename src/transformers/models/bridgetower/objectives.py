# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TODO"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
from tqdm import tqdm
from tqdm.contrib import tzip
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .dist_utils import all_gather


def compute_mlm(pl_module, batch, split):
    infer = pl_module.infer_simple(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    loss_name = 'mlm'

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    
    return ret


def compute_itm(pl_module, batch, split):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer_simple(batch)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }
    
    loss_name = 'itm'

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/accuracy", acc)

    return ret

def compute_snli(pl_module, batch, split):
    infer = pl_module.infer_simple(batch) 
    snli_logits = pl_module.snli_classifier(infer["cls_feats"])

    snli_labels = batch["labels"]
    snli_labels = torch.tensor(snli_labels).to(pl_module.device).long()
    snli_loss = F.cross_entropy(snli_logits, snli_labels.view(-1))

    ret = {
        "snli_loss": snli_loss,
        "snli_logits": snli_logits,
        "snli_labels": snli_labels,
    }

    loss_name = 'snli'

    if split == "train":
        loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["snli_loss"])
        acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
            ret["snli_logits"], ret["snli_labels"]
        )
        pl_module.log(f"{split}/{loss_name}/loss", loss)
        pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    else:
        val_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if val_batches:
            val_loss = getattr(pl_module, f"val_{loss_name}_loss")(
                F.cross_entropy(
                    ret["snli_logits"][val_batches], ret["snli_labels"][val_batches]
                )
            )
            val_acc = getattr(pl_module, f"val_{loss_name}_accuracy")(
                ret["snli_logits"][val_batches], ret["snli_labels"][val_batches]
            )
            pl_module.log(f"val/snli/loss", val_loss)
            pl_module.log(f"val/snli/accuracy", val_acc)

        if test_batches:
            test_loss = getattr(pl_module, f"test_{loss_name}_loss")(
                F.cross_entropy(
                    ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_{loss_name}_accuracy")(
                ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
            )
            pl_module.log(f"test/snli/loss", test_loss)
            pl_module.log(f"test/snli/accuracy", test_acc)

    return ret


def compute_vqa(pl_module, batch, split):
    infer = pl_module.infer_simple(batch)
    
    if pl_module.hparams.config["downstream_fusion_method"] == 'ensemble':
        vqa_logits = []
        for feats, classifier in zip(infer["cls_feats"], pl_module.vqa_classifier):
            vqa_logits.append(classifier(feats))
        vqa_logits = torch.stack(vqa_logits, dim=0)
        vqa_logits = torch.mean(vqa_logits, dim=0)
    else:
        vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
        # vqa_logits = vqa_logits.to('cpu')
    # vqa_targets = torch.zeros(
    #     len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    # ).to('cpu') # .to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]
    vqa_targets = batch['vqa_targets']

    # for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
    #     for l, s in zip(_label, _score):
    #         vqa_targets[i, l] = s
    # vqa_targets = vqa_targets.to(pl_module.device)

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    loss_name = 'vqa'

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{split}_{loss_name}_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/score", score)

    return ret


def compute_nlvr2(pl_module, batch, split):
    if pl_module.hparams.config["nlvr2_head_format"] == 'triplet':
        img1 = pl_module.infer_image_simple(batch[f"image_0"][0])
        img2 = pl_module.infer_image_simple(batch[f"image_1"][0])
        image_embedss = torch.cat((img1, img2), dim=-2)
        text_embedss, extend_text_masks = pl_module.infer_text_simple(batch)
        cls_feats = pl_module.infer_fusion_simple(image_embedss, text_embedss, extend_text_masks, image_token_type_idx=3)["cls_feats"]
    else:
        infer1 = pl_module.infer_simple(batch, image_token_type_idx=1)
        infer2 = pl_module.infer_simple(batch, image_token_type_idx=2)
        
        if pl_module.hparams.config["nlvr2_head_format"] == 'pair':
            cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
        elif pl_module.hparams.config["nlvr2_head_format"] == 'pair-biatten':
            left_out, right_out = infer1["image_feats"], infer2["image_feats"]
            l2r_attn, _ = pl_module.nlvr2_biatten_head_attn1(left_out, right_out, right_out)
            r2l_attn, _ = pl_module.nlvr2_biatten_head_attn2(right_out, left_out, left_out)
            left_out = pl_module.nlvr2_biatten_head_fc(torch.cat([l2r_attn, left_out], dim=-1))
            right_out = pl_module.nlvr2_biatten_head_fc(torch.cat([r2l_attn, right_out], dim=-1))
            left_out = pl_module.nlvr2_biatten_head_attn_pool(left_out)
            right_out = pl_module.nlvr2_biatten_head_attn_pool(right_out)
            cls_feats = torch.cat([left_out, right_out], dim=-1)
        else:
            raise ValueError(f"Unknown nlvr2_head_format: {pl_module.hparams.config['nlvr2_head_format']}")

    cls_feats = pl_module.nlvr2_classifier_dropout(cls_feats)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    loss_name = 'nlvr2'

    if split == "train":
        loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"{split}/{loss_name}/loss", loss)
        pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    else:
        val_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if val_batches:
            val_loss = getattr(pl_module, f"val_{loss_name}_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][val_batches], ret["nlvr2_labels"][val_batches]
                )
            )
            val_acc = getattr(pl_module, f"val_{loss_name}_accuracy")(
                ret["nlvr2_logits"][val_batches], ret["nlvr2_labels"][val_batches]
            )
            pl_module.log(f"val/nlvr2/loss", val_loss)
            pl_module.log(f"val/nlvr2/accuracy", val_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_{loss_name}_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_{loss_name}_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"test/nlvr2/loss", test_loss)
            pl_module.log(f"test/nlvr2/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch, split):
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)

    infer = pl_module.infer_simple(
        {
            "image": batch["image"],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        },
        irtr_len=false_len+1,
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    loss_name = 'irtr'

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["irtr_loss"])
    pl_module.log(f"{split}/{loss_name}/loss", loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module, split):
    print("load irtr dataset for text features caching")
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_text"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=text_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )
    print("load irtr dataset for image features caching")
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split, image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_image"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=image_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )
    print("start to cache the text features")
    text_embedss_cache, extend_text_masks_cache, tiids = list(), list(), list()
    for _b in tqdm(text_loader, desc="text prefetch loop"):
        text_embedss, extend_text_masks = pl_module.infer_text_simple(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
            },
        )
        text_embedss_cache.append(text_embedss)
        extend_text_masks_cache.append(extend_text_masks)
        tiids += _b["img_index"]

    text_embedss_cache = torch.cat(text_embedss_cache, dim=1)
    extend_text_masks_cache = torch.cat(extend_text_masks_cache, dim=0)
    tiids = torch.LongTensor(tiids)

    # gather all text features
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    text_embedss_cache = all_gather(text_embedss_cache)
    extend_text_masks_cache = all_gather(extend_text_masks_cache)
    
    text_embedss_cache = torch.cat([i.to(pl_module.device) for i in text_embedss_cache], dim=1)
    extend_text_masks_cache = torch.cat([i.to(pl_module.device) for i in extend_text_masks_cache], dim=0)
    tiids = torch.cat(all_gather(tiids), dim=0).view(-1)
    
    print("start to cache the image features")
    image_embedss_cache, iids_cache = list(), list()
    for _b in tqdm(image_loader, desc="image prefetch loop"):
        image_embedss = pl_module.infer_image_simple(img=_b["image"][0].to(pl_module.device))
        image_embedss_cache.append(image_embedss)
        iids_cache += _b["img_index"]
    image_embedss_cache = torch.cat(image_embedss_cache, dim=1)
    
    image_index, rank_scores, rank_iids = 0, list(), list()
    
    text_chunk_size = pl_module.hparams.config["per_gpu_eval_batchsize_fusion_text"]
    if text_embedss_cache.size(1) % text_chunk_size == 0:
        text_chunk_num = text_embedss_cache.size(1) // text_chunk_size
    else:
        text_chunk_num = text_embedss_cache.size(1) // text_chunk_size + 1

    print("start to compute the irtr recall")
    for _iid in tqdm(iids_cache, desc="rank loop"):
        image_embedss = image_embedss_cache[:, image_index]
        image_index += 1
        
        img_batch_score = list()
        for _i in range(text_chunk_num):
            text_embedss = text_embedss_cache[:, _i*text_chunk_size:(_i+1)*text_chunk_size]
            extend_text_masks = extend_text_masks_cache[_i*text_chunk_size:(_i+1)*text_chunk_size]
            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer_fusion_simple(
                        image_embedss, 
                        text_embedss, 
                        extend_text_masks, 
                        irtr_len=text_embedss.size(1),
                    )["cls_feats"]
                )[:, 0]            

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.LongTensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    try:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
            if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        )
    except:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["gqa_test"].id2answer
            if "gqa_test" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["gqa"].id2answer
        )
        vqa_logits = output["vqa_logits"]
        if pl_module.hparams.config["downstream_fusion_method"] == 'ensemble':
            vqa_logits = torch.mean(vqa_logits, dim=0)
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        questions = batch["text"]
        qids = batch["qid"]
        return {"qids": qids, "preds": vqa_preds, "gqa": True}
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": False}

def vqa_test_wrapup(outs, model_name, log_dir, num_nodes):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    gqa = False
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out['gqa']

    rets = list()
    for qid, pred in zip(qids, preds):
        if gqa:
            rets.append({"questionId": qid, "prediction": pred})
        else:
            rets.append({"question_id": qid, "answer": pred})
    # if num_nodes == 1:

    with open(f"{log_dir}/vqa_submit_{model_name}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"{log_dir}/vqa_submit_{model_name}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        with open(f"{log_dir}/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)
        
    #     if torch.distributed.is_initialized():
    #         torch.distributed.barrier()
    #     os.remove(f"{log_dir}/vqa_submit_{model_name}_{rank}.json")

    # else:
    # if torch.distributed.is_initialized():
    #     torch.distributed.barrier()   
    
    # gather_rets = all_gather(rets)
    # print(f'length of gather_rest: {len(gather_rets)}')
    
    # if rank == 0:
    #     jsons = list()
    #     for rets_ in gather_rets:
    #         jsons += rets_
    #     with open(f"{log_dir}/vqa_submit_{model_name}.json", "w") as fp:
    #         json.dump(jsons, fp, indent=4)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def set_metrics(pl_module):
    for split in ["train", "val", "test"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v <= pl_module.hparams.config["task_threshold"]:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_{k}_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2" or k == "snli" or k == "itm" or k == "mlm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "irtr":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            else:
                raise ValueError(f"Unknown loss name: {k}")

    if pl_module.hparams.config["test_only"]:
        split = "test"
    else:
        split = "val"
    setattr(pl_module, "best_metric_log", {f"{split}/the_metric": 0})
    
    for k, v in pl_module.hparams.config["loss_names"].items():
        if v <= pl_module.hparams.config["task_threshold"]:
            continue
        if k == "vqa" and split == "val":
            getattr(pl_module, f"best_metric_log").update({
                f"{split}/{k}_score": -1e9,
                f"{split}/{k}_loss": 1e9,
            })
        if k == "nlvr2" or k == "snli":
             getattr(pl_module, f"best_metric_log").update({
                f"val/{k}_accuracy": -1e9,
                f"test/{k}_accuracy": -1e9,
                f"val/{k}_loss": 1e9,
                f"test/{k}_loss": 1e9,
            })     
        if k == "mlm" and split == "val":
            getattr(pl_module, f"best_metric_log").update({
                f"{split}/{k}_accuracy": -1e9,
                f"{split}/{k}_loss": 1e9,
            })
        if k == "itm":
            getattr(pl_module, f"best_metric_log").update({
                f"{split}/{k}_accuracy": -1e9,
                f"{split}/{k}_loss": 1e9,
            })
        
        if k == "irtr":
            getattr(pl_module, f"best_metric_log").update({
                f"{split}/{k}_loss": 1e9,
                f"{split}/ir_r1": -1e9,
                f"{split}/ir_r5": -1e9,
                f"{split}/ir_r10": -1e9,
                f"{split}/tr_r1": -1e9,
                f"{split}/tr_r5": -1e9,
                f"{split}/tr_r10": -1e9,
            })
