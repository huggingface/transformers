import json
import os
import mmap
import pickle
import random
import time
import torch

from threading import Lock
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler
from itertools import accumulate

class LazyDataset(Dataset):
    """A dataset for lazy loading from disk.
    """

    def __init__(self, path, use_mmap=True):
        self.path = path
        self.use_mmap = use_mmap

        if not os.path.exists(path):
            raise ValueError('File `%s` does not exist.' % path)

        self.lazy_dir = self.get_lazy_dir()
        self.lazy_file = os.path.join(self.lazy_dir, 'data')
        self.lazy_index = os.path.join(self.lazy_dir, 'index.pkl')
        if not os.path.exists(self.lazy_dir):
            self.create_index()

        self.file = open(self.lazy_file, 'rb')
        if self.use_mmap:
            print('Memory map `%s`' % path)
            # In distributed setting, it's useful to cache file in ramdisk
            self.file = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)

        self.read_lock = Lock()

        with open(self.lazy_index, 'rb') as f_index:
            self.offsets = pickle.load(f_index)

    def get_lazy_dir(self):
        return os.path.splitext(self.path)[0]+'_lazy'

    def create_index(self):
        # In distributed setting, only first process creates the index..
        # Note that we assume that the file is stored on a shared file system
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            os.makedirs(self.lazy_dir)
            print("Creating index for lazy loading of `%s`" % self.path)
            offsets = list()
            cnt = 0
            with open(self.path, 'rb') as f_in, open(self.lazy_file, 'wb') as f_out:
                for line in f_in:
                    f_out.write(line)
                    cnt += len(line)
                    offsets.append(cnt)
            with open(self.lazy_index, 'wb') as f_index:
                pickle.dump(offsets, f_index)
        else:
            while not os.path.exists(self.lazy_index):
                time.sleep(1)


    def __getitem__(self, index):
        start = 0 if index == 0 else self.offsets[index-1]
        end = self.offsets[index] - 1

        self.read_lock.acquire()
        self.file.seek(start)
        txt = self.file.read(end - start)
        self.read_lock.release()
        return txt.decode('utf-8')

    def __len__(self):
        return len(self.offsets)


class JSONDataset(Dataset):

    def __init__(self, dataset, key=None):
        self.dataset = dataset
        self.key = key

    def __getitem__(self, index):
        item = json.loads(self.dataset[index])
        if self.key:
            item = item[self.key]
        return item

    def __len__(self):
        return len(self.dataset)

class BertDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_seq_length=512, short_seq_prob=0.1,
                 masked_lm_prob=0.15, max_predictions_per_seq=80):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.short_seq_prob = short_seq_prob
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq


    def __getitem__(self, index):
        # get rng state corresponding to index (allows deterministic random pair)
        rng = random.Random(index)
        # get seq length, subtract [CLS], .., [SEP], .. [SEP] tokens
        target_seq_length = self.max_seq_length - 3

        if rng.random() < self.short_seq_prob:
            target_seq_length = rng.randint(2, target_seq_length)

        # get sentence pair and label
        tokens_a, tokens_b, is_random_next = self.create_random_instance(target_seq_length, rng)
        self.truncate_seq_pair(tokens_a, tokens_b, target_seq_length, rng)

        # [CLS], tokens_a.. [SEP] tokens_b... [SEP]
        tokens = list()
        segment_ids = list()
        tokens.append('[CLS]')
        for tok in tokens_a:
            tokens.append(tok)
            segment_ids.append(0)

        tokens.append('[SEP]')
        segment_ids.append(0)

        for tok in tokens_b:
            tokens.append(tok)
            segment_ids.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)

        # pad sequences to max length
        attention_mask = [1]*512
        attention_mask[len(tokens):] = [0]*(512-len(tokens))
        self.pad_seq(tokens, self.max_seq_length, '[PAD]')
        self.pad_seq(segment_ids, self.max_seq_length, 0)

        # randomly mask inputs
        input_tokens, lm_labels = self.create_masked_lm_predictions(tokens, rng)

        input_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)

        sample = dict()
        sample['input_tokens'] = torch.LongTensor(input_tokens)
        sample['segment_ids'] = torch.LongTensor(segment_ids)
        sample['attention_mask'] = torch.LongTensor(attention_mask)
        sample['lm_labels'] = torch.LongTensor(lm_labels)
        sample['is_random_next'] = torch.LongTensor([int(is_random_next)])
        return sample

    def pad_seq(self, seq, max_len, val):
        while len(seq) < max_len:
            seq.append(val)

    def sentence_split(self, document):
        """split document into sentences"""
        return [line for line in document.split('\n') if line]

    def get_random_doc(self, rng):
        doc_idx = rng.randint(0, len(self.dataset) - 1)
        doc = self.sentence_split(self.dataset[doc_idx])
        doc = [self.tokenizer.tokenize(sentence) for sentence in doc]
        return doc, doc_idx

    def remaining_doc_length(self, doc):
        segment_lengths = [len(sentence) for sentence in doc]
        return list(accumulate(reversed(segment_lengths)))[::-1]

    def create_random_instance(self, target_seq_length, rng):
        """Fetches a random sentence pair corresponding to rng state similar to
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L248-L294"""

        # sample random doc, make sure it contains more than `target_seq_length` tokens
        doc_a, doc_a_idx = self.get_random_doc(rng)
        remaining_doc_length = self.remaining_doc_length(doc_a)
        len_doc_a = remaining_doc_length[0]
        while len_doc_a < target_seq_length:
            doc_a, doc_a_idx = self.get_random_doc(rng)
            remaining_doc_length = self.remaining_doc_length(doc_a)
            len_doc_a = remaining_doc_length[0]

        # start at random segment (but ensure that we keep more than `target_seq_length` tokens)
        segment_id = rng.choice([i for i, l in enumerate(remaining_doc_length) if l >= target_seq_length])
        segments = list()
        num_tokens = 0
        for segment in doc_a[segment_id:]:
            segments.append(segment)
            num_tokens += len(segment)
            if num_tokens >= target_seq_length:
                break

        a_end = 1
        if len(segments) > 1:
            a_end = rng.randint(1, len(segments) - 1)

        a_tokens = list()
        for seg in segments[:a_end]:
            a_tokens.extend(seg)

        if len(segments) == 1 or random.random() > 0.5:
            is_random_next = True
            target_b_length = target_seq_length - len(a_tokens)

            # ensure that doc_a != doc_b
            doc_b, doc_b_idx = self.get_random_doc(rng)
            remaining_doc_b_length = self.remaining_doc_length(doc_b)
            while doc_a_idx == doc_b_idx or remaining_doc_b_length[0] < target_b_length:
                doc_b, doc_b_idx = self.get_random_doc(rng)
                remaining_doc_b_length = self.remaining_doc_length(doc_b)

            remaining_doc_b_length = self.remaining_doc_length(doc_b)
            segment_id = rng.choice([i for i, l in enumerate(remaining_doc_b_length) if l >= target_b_length])

            b_tokens = list()
            for seg in doc_b[segment_id:]:
                b_tokens.extend(seg)
                if len(b_tokens) >= target_b_length:
                    break
        else:
            is_random_next = False
            b_tokens = list()
            for seg in segments[a_end:]:
                b_tokens.extend(seg)

        return a_tokens, b_tokens, is_random_next

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens, rng):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if rng.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def create_masked_lm_predictions(self, tokens, rng):
        cand_indexes = [i for i, tok in enumerate(tokens) if tok not in ('[SEP]', '[CLS]', '[PAD]')]
        rng.shuffle(cand_indexes)
        output_tokens = list(tokens)
        num_to_predict = min(self.max_predictions_per_seq,
                             max(1, int(round(len(cand_indexes) * self.masked_lm_prob))))

        lm_labels = [-1]*len(tokens)
        for index in cand_indexes[:num_to_predict]:
            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    random_tok_ind = rng.randint(0, len(self.tokenizer.vocab) - 1)
                    masked_token = self.tokenizer.ids_to_tokens[random_tok_ind]

            output_tokens[index] = masked_token
            lm_labels[index] = self.tokenizer.vocab[tokens[index]]

        return output_tokens, lm_labels

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir='./data')

    wiki_dataset = LazyDataset('/home/nathan/data/wiki/enwiki.txt', use_mmap=True)
    wiki_dataset = JSONDataset(wiki_dataset, key='text')

    dataset = BertDataset(wiki_dataset, tokenizer)
    sampler = BatchSampler(RandomSampler(dataset), 4, True)

    iterator = DataLoader(dataset, batch_sampler=sampler, num_workers=1, pin_memory=True)

    for batch in iterator:
        print(batch.keys())