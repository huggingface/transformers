import os
from typing import Dict

import nlp
import pytorch_lightning as pl
import torch
from tokenizers import AddedToken, CharBPETokenizer, Tokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import BartConfig, BartForConditionalGeneration
from utils import label_smoothed_nll_loss


class Translate(pl.LightningModule):
    def __init__(self, model, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, x, **kwargs):
        return self.model.forward(x, **kwargs)

    def step(self, batch):
        source_ids, source_mask, target_ids = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["decoder_input_ids"],
        )

        decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
        lm_labels = target_ids[:, 1:].clone()  # why clone?

        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False,)
        lprobs = F.log_softmax(outputs[0], dim=-1)
        loss = label_smoothed_nll_loss(lprobs, lm_labels, epsilon=0.1, ignore_index=self.tokenizer.pad_token_id)
        return loss[0]

    def training_step(self, batch, batch_nb):
        loss = self.step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss = self.step(batch)
        tensorboard_logs = {"val_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_acc_mean = 0
        for output in outputs:
            val_acc_mean += output["loss"]

        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_acc_mean.item()}

        # show val_acc in progress bar but only log val_loss
        results = {"progress_bar": tqdm_dict, "log": {"val_loss": val_acc_mean}}
        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-7, betas=(0.9, 0.98))


def encode_line(
    tokenizer, line, max_length, pad_token_id, pad_to_max_length=True, return_tensors="pt",
):
    encoded = tokenizer.encode(line)
    input_ids = torch.zeros(max_length).long() + pad_token_id
    attention_mask = torch.zeros(max_length).long()
    n = len(encoded.ids)
    input_ids[:n] = torch.Tensor(encoded.ids).long()
    attention_mask[:n] = torch.Tensor(encoded.attention_mask).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        raw_dataset,
        max_source_length,
        max_target_length,
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        self.dataset = raw_dataset
        self.src_lens = len(raw_dataset)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        source_line = self.dataset[index]["source"]
        target_line = self.dataset[index]["target"]
        assert source_line, f"empty source line for index {index}"
        assert target_line, f"empty tgt line for index {index}"
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length, self.pad_token_id)
        target_inputs = encode_line(self.tokenizer, target_line, self.max_target_length, self.pad_token_id)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
        }
        return batch


def main():
    batch_size = 32
    vocab_size = 16384
    max_source_length = 1024
    max_target_length = 1024
    num_workers = 3

    dataset = nlp.load_dataset("iwslt2017", "nl-en")

    # Train tokenizer
    tokenizer_filename = "tokenizer.json"
    if os.path.exists(tokenizer_filename):
        tokenizer = Tokenizer.from_file(tokenizer_filename)
    else:
        data_filename = "whole_data.txt"
        with open(data_filename, "w") as f:
            for item in dataset["train"]:
                f.write(item["source"] + "\n")
                f.write(item["target"] + "\n\n")

        tokenizer = CharBPETokenizer()
        tokenizer.train([data_filename], vocab_size=vocab_size)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        tokenizer.add_tokens([pad_token])
        tokenizer.save(tokenizer_filename)

    tokenizer.pad_token_id = vocab_size

    # Loaders
    train_dataset = Seq2SeqDataset(tokenizer, dataset["train"], max_source_length, max_target_length)
    val_dataset = Seq2SeqDataset(tokenizer, dataset["validation"], max_source_length, max_target_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn, num_workers=num_workers,
    )

    # Train model
    config = BartConfig(
        vocab_size=vocab_size + 1,  # Pad
        d_model=1024,
        encoder_ffn_dim=1024,
        encoder_layers=6,
        encoder_attention_heads=4,
        decoder_ffn_dim=1024,
        decoder_layers=6,
        decoder_attention_heads=4,
    )
    model = BartForConditionalGeneration(config)
    translator = Translate(model, tokenizer)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(translator, train_loader, val_loader)


if __name__ == "__main__":
    main()
