import os

from torch.utils.data import Dataset


def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    with open(data_path, "r") as f:
        for text in f.readlines():
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
    return examples


class SummarizationDataset(Dataset):
    def __init__(
        self, tokenizer, data_dir="./cnn-dailymail/cnn_dm/", type_path="train", block_size=1024, max_target_length=56
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.source = encode_file(tokenizer, os.path.join(data_dir, type_path + ".source"), block_size)
        self.target = encode_file(tokenizer, os.path.join(data_dir, type_path + ".target"), max_target_length)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}
