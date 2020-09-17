from pathlib import Path

import fire
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
try:
    from .utils import Seq2SeqDataset, pickle_save
except ImportError:
    from utils import Seq2SeqDataset, pickle_save

def save_len_file(tokenizer_name, data_dir, max_source_length=1024, max_target_length=1024, **kwargs):
    """Save max(src_len, tgt_len) for each example to allow dynamic batching."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    train_ds = Seq2SeqDataset(tok, data_dir, max_source_length, max_target_length, type_path='train', **kwargs)
    def get_lens(ds):
        dl = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False, collate_fn=ds.collate_fn)
        return [max(batch['input_ids'][0].shape[0], batch['labels'][0].shape[0]) for batch in dl]
    train_lens = get_lens(train_ds)
    val_ds = Seq2SeqDataset(tok, data_dir, max_source_length, max_target_length, type_path='val', **kwargs)
    val_lens = get_lens(val_ds)
    pickle_save(train_lens, train_ds.len_file)
    pickle_save(val_lens, val_ds.len_file)



if __name__ == "__main__":
    fire.Fire(save_len_file)
