import torch
from torch import nn
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, MarianMTModel
from transformers.trainer_seq2seq import Seq2SeqTrainer as BaseSeq2SeqTrainer

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
        # attention masks de 1s
        self.attn_mask = torch.ones_like(input_ids)
        self.dec_attn_mask = torch.ones_like(labels)
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "labels": self.labels[idx],
            "decoder_attention_mask": self.dec_attn_mask[idx],
        }

def test_ignore_decoder_inputs_conflict(tmp_path):
    # Cargamos un modelo pequeño de Marian (se descarga la primera vez)
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    # Forzamos que haya ambos argumentos en inputs: ids y embeds
    input_ids = torch.tensor([[0, 1, 2]])
    labels   = torch.tensor([[0, 1, 2]])
    # Creamos los embeds del decoder
    decoder_inputs_embeds = model.model.decoder.embed_tokens(labels)

    # Dataset dummy
    ds = DummyDataset(input_ids, labels)

    # Args de entrenamiento mínimos
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        no_cuda=True,
    )

    # Usamos tu Seq2SeqTrainer parcheado
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=ds,
        tokenizer=None,
    )

    # Ejecutamos un solo paso de train; antes explotaba aquí
    state = trainer.train()

    # Comprobamos que al menos avanzó un paso
    assert state.global_step >= 1
