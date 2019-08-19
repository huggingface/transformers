from torch.utils.data import Dataset, DataLoader
import os
import random
import torch
import torch.nn.functional as F
import logging
import pickle

logger = logging.getLogger(__name__)


class WikiTextDataset(Dataset):
	def __init__(self, args, tokenizer, file='train', directory='wikitext', max_context_length=512, cache=None):
		if args.local_rank not in [-1, 0]:
			torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
			
			
		cached_features_file = os.path.join(args.data_dir, f'cached_lm_{file}_{args.max_seq_length}')
		
		if os.path.exists(cached_features_file):
			logger.info("Loading features from cached file %s", cached_features_file)
			with open(cached_features_file, 'rb') as handle:
				self.examples = pickle.load(handle)
		else:
			logger.info("Creating features from dataset file at %s", args.data_dir)	
		
		self.max_context_length = max_context_length

		self.examples = []

		with open(os.path.join(directory, f"wiki.{file}.raw"), encoding="utf-8") as f:
			text = f.read()
			tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

			while len(tokenized_text) > max_context_length:
				self.examples.append(tokenized_text[:max_context_length])
				tokenized_text = tokenized_text[max_context_length:]
			
		if args.local_rank in [-1, 0]:
			logger.info("Saving features into cached file %s", cached_features_file)
			with open(cached_features_file, 'wb') as handle:
				pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		if args.local_rank == 0:
			torch.distributed.barrier()

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, item):
		return torch.tensor(self.examples[item])
