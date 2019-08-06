from torch.utils.data import Dataset, DataLoader
import os
import random
import torch
import torch.nn.functional as F


class WikiTextDataset(Dataset):
	def __init__(self, tokenizer, file='train', directory='wikitext', max_context_length=512, device='cpu'):
		self.device = device
		self.max_context_length = max_context_length

		self.examples = []

		with open(os.path.join(directory, f"wiki.{file}.raw"), encoding="utf-8") as f:
			text = f.read()
			spans = list(filter(lambda item: len(item) > 120, text.split("\n")[:20]))

			for span in spans:
				span = tokenizer.encode(span)
				while len(span) > 0:
					self.examples.append(span[:max_context_length])
					span = span[max_context_length:]

		# Randomly shuffle the examples array
		random.shuffle(self.examples)

		# Sort the array by example length.
		self.examples.sort(key=len)

		print("nice")

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, item):
		return torch.tensor(self.examples[item], device=self.device)

	@staticmethod
	def collate(values):
		stack = torch.stack([F.pad(value, (len(values[-1]) - value.size(0), 0), "constant", 0) for value in values])
		return stack
