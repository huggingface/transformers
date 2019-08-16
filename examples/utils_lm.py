from torch.utils.data import Dataset, DataLoader
import os
import random
import torch
import torch.nn.functional as F


class WikiTextDataset(Dataset):
	def __init__(self, tokenizer, file='train', directory='wikitext', max_context_length=1024):
		self.max_context_length = max_context_length

		self.examples = []

		with open(os.path.join(directory, f"wiki.{file}.raw"), encoding="utf-8") as f:
			text = f.read()
			tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

			while len(tokenized_text) > max_context_length:
				self.examples.append(tokenized_text[:max_context_length])
				tokenized_text = tokenized_text[max_context_length:]

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, item):
		return torch.tensor(self.examples[item])
