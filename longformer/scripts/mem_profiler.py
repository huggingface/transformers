from longformer.longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration
from longformer.longformer_encoder_decoder import LongformerEncoderDecoderConfig

from longformer.longformer import LongformerForMaskedLM
from longformer.longformer import LongformerConfig

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
import pytorch_lightning as pl

seqlen = 1024 * 2
global_size = seqlen // 100
attention_window = 256  # one sided


class CoolDataset(Dataset):
    def __len__(self):
        return 1024  # number of examples

    def __getitem__(self, idx):
        tokne_ids = torch.tensor([5] * seqlen)
        mask = torch.tensor([1] * seqlen)
        mask[:global_size] = 2
        return tokne_ids, mask


class MemoryProfiler(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()
        self.hparams = hparams

        config = LongformerEncoderDecoderConfig.from_pretrained('bart-long-4096')
        # config = LongformerConfig.from_pretrained('roberta-large')
        config.max_position_embeddings = seqlen + 2
        config.gradient_checkpointing = True
        config.attention_mode = 'sliding_chunks'
        # config.attention_mode = 'n2'
        config.attention_window = [attention_window] * config.num_hidden_layers
        config.attention_dilation = [1] * config.num_hidden_layers
        self.model = LongformerEncoderDecoderForConditionalGeneration(config)
        # self.model = LongformerForMaskedLM(config)

    def forward(self, x, y):
        print(seqlen, global_size, attention_window, torch.cuda.max_memory_allocated(x.device) / 1024 ** 3)
        # import ipdb; ipdb.set_trace()
        # return self.model(x, attention_mask=y, decoder_input_ids=x[:, :attention_window * 2], use_cache=False)
        return self.model(x, attention_mask=y)

    def training_step(self, batch, batch_idx):
        # import ipdb; ipdb.set_trace()
        x, y = batch
        y_hat = self(x, y)
        loss = y_hat[0].sum()
        # import ipdb; ipdb.set_trace()
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return DataLoader(CoolDataset(), batch_size=2, num_workers=0)


if __name__ == '__main__':
    model = MemoryProfiler(hparams={})
    trainer = Trainer(gpus=[0], progress_bar_refresh_rate=1, max_epochs=1, amp_level='O2', use_amp=True)
    trainer.fit(model)
