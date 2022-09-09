import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right


@dataclass
class DataCollatorForBartDenoisingLM:
    """
    Data collator used for BART denoising language modeling.

    The code is modified from the original
    <Denoising Dataset https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/denoising_dataset.py>`_.
    This implies some minute differences. As a data collator, we now work on the batch level rather than sample level.
    This mainly has a consequence if/when we insert noise (the default BART training did not add noise, however). In
    this event, all sequences in a batch will get the same number of added noise (num_noise). In fairseq, this is
    implemented on the dataset-level, meaning that every sequence in a batch may have a different amount of noise.

    The defaults here are based on the defaults for training BART, as indicated
    `here <https://github.com/facebookresearch/fairseq/issues/1899#issuecomment-1069429320>`_.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
        permute_sentence_ratio (:obj:`float`):
            Ratio of sentences to be permuted in each document
        mask_ratio (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input
        random_ratio (:obj:`float`):
            The probability with which to (randomly) replace a token by a random token
        insert_ratio (:obj:`float`):
            The probability with which to (randomly) insert noise. Will add `insert_ratio * input.numel()` noise
        rotate_ratio (:obj:`float`):
                The probability with which to (randomly) add rolling noise (i.e., shifted tokens in order)
        poisson_lambda (:obj:`float`):
            Mean parameter of Poisson distribution used to generate span-lengths to be masked
        mask_length (:obj:`str`):
            Whether to add span, word or subword masks ("span-poisson", "word", "subword")
        mask_whole_word (:obj:`torch.Tensor`):
            A tensor of the size of the vocabulary to indicate which of the tokens are the start of a word.
            Also see the
            `fairseq implementation https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/tasks/multilingual_masked_lm.py#L118`_
            of this.
    """

    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int
    permute_sentence_ratio: float = 1.0
    mask_ratio: float = 0.3
    random_ratio: float = 0.1
    insert_ratio: float = 0.0
    rotate_ratio: float = 0.0
    poisson_lambda: float = 3.5
    replace_length: int = 1
    mask_length: Optional[str] = "span-poisson"
    mask_whole_word: Optional[torch.LongTensor] = None

    def __post_init__(self):
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={self.mask_length}")
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError("if using subwords, use replace-length=1 or 0")

        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            _lambda = self.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        batch["labels"] = batch["input_ids"].clone()
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )

        if self.permute_sentence_ratio > 0.0:
            batch["input_ids"] = self.permute_sentences(batch["input_ids"])

        if self.mask_ratio > 0:
            batch["input_ids"] = self.add_whole_word_mask(batch["input_ids"])

        if self.insert_ratio > 0:
            batch["input_ids"] = self.add_insertion_noise(batch["input_ids"], self.insert_ratio)

        if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
            batch["input_ids"] = self.add_rolling_noise(batch["input_ids"])

        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).long()

        return batch

    def permute_sentences(self, input_ids):
        """Different implementation than in fairseq. See
        this issue https://github.com/facebookresearch/fairseq/issues/4695`_"""
        all_results = input_ids.clone()
        for seq_idx, sequence in enumerate(input_ids):
            full_stops = sequence == self.tokenizer.pad_token_id

            # Find the position of </s> EOS tokens, and mark the position before that as a full stop
            # so that the last sentence can also be extracted as a span
            # This approach is needed when our batches have padding (and we cannot simply target the one but last item)
            eos_positions = (sequence == self.tokenizer.eos_token_id).roll(-1)
            full_stops[eos_positions] = 1

            # Mark sentence ends: those cases where the token is a full_stop (pad)
            # but the previous and next ones are not
            next_token_is_full_stop = torch.cat((full_stops[2:], torch.BoolTensor([0])))
            sentence_ends = (full_stops[1:] * ~full_stops[:-1] * ~next_token_is_full_stop).nonzero(as_tuple=False) + 2
            result = sequence.clone()

            num_sentences = sentence_ends.size(0)
            num_to_permute = math.ceil((num_sentences * 2 * self.permute_sentence_ratio) / 2.0)
            substitutions = torch.randperm(num_sentences)[:num_to_permute]
            ordering = torch.arange(0, num_sentences)
            ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

            # Ignore <bos> at start
            index = 1
            for order_idx, orig_sent_idx in enumerate(ordering):
                is_last_orig = orig_sent_idx == num_sentences - 1
                is_last_in_loop = order_idx == num_sentences - 1
                start_idx = sentence_ends[orig_sent_idx - 1] if orig_sent_idx > 0 else 1
                # remove last idx (pad) from last sentence of this loop but only if it is not the orig last sentence
                end_idx = sentence_ends[orig_sent_idx] - (int(is_last_in_loop) if not is_last_orig else 0)
                sentence = sequence[start_idx:end_idx]

                # add padding token if this was the original last sentence and now it isn't anymore
                if is_last_orig and not is_last_in_loop:
                    sentence = torch.cat((sentence, torch.LongTensor([self.tokenizer.pad_token_id])))

                result[index : index + sentence.size(0)] = sentence
                index += sentence.size(0)

            all_results[seq_idx] = result

        return all_results

    def get_word_starts(self, input_ids):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, input_ids.view(-1)).reshape(input_ids.size())
        else:
            is_word_start = (
                ~torch.BoolTensor(
                    [self.tokenizer.get_special_tokens_mask(seq, already_has_special_tokens=True) for seq in input_ids]
                )
            ).long()
        is_word_start[:, 0] = 0
        is_word_start[:, -1] = 0
        return is_word_start

    def add_whole_word_mask(self, input_ids):
        # Note that is_word_start cannot be a booltensor but has to be an int tensor as we use it to subtract
        # from the span lengths later on
        is_word_start = self.get_word_starts(input_ids)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * self.mask_ratio))

        num_inserts = 0
        if num_to_mask == 0:
            return input_ids

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))
            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)

            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            # For every 0-length span, we instead insert noise
            # So we decrease the required `num_to_mask` and instead add to `num_inserts`
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(input_ids, num_inserts / input_ids.numel())

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,), dtype=torch.long)

        assert not is_word_start[:, 0].any()
        assert not is_word_start[:, -1].any()

        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]]

        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = input_ids.size(1)
        assert source_length - 1 not in indices[:, 1]

        to_keep = torch.ones_like(input_ids, dtype=torch.bool)
        is_word_start[:, -1] = 255  # acts as a long length, so spans don't go over the end of doc

        if self.replace_length == 0:
            to_keep[indices[:, 0], indices[:, 1]] = 0
        else:
            # Mask some tokens with a mask token
            for idxs in indices:
                input_ids[tuple(idxs)] = self.tokenizer.mask_token_id

            # Replace a fraction (random_ratio) with a random token
            rand_tokens = torch.randint(1, len(self.tokenizer), size=(mask_random.sum(),))
            for idxs, tok in zip(indices[mask_random], rand_tokens):
                input_ids[tuple(idxs)] = tok

        if self.mask_span_distribution is not None:
            lengths -= 1

            while indices.size(0) > 0:
                lengths -= is_word_start[indices[:, 0], indices[:, 1] + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted, :]
                indices[:, 1] += 1  # increment to keep masking the next positions

                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]

                if self.replace_length != -1:
                    to_keep[indices[:, 0], indices[:, 1]] = 0
                else:
                    # Mask some tokens with a mask token
                    for idxs in indices:
                        input_ids[tuple(idxs)] = self.tokenizer.mask_token_id

                    # Replace a fraction (random_ratio) with a random token
                    rand_tokens = torch.randint(1, len(self.tokenizer), size=(mask_random.sum(),))
                    for idxs, tok in zip(indices[mask_random], rand_tokens):
                        input_ids[tuple(idxs)] = tok

                assert source_length - 1 not in indices[:, 1]
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices[:, 0], indices[:, 1] + 1] == 0
                indices = indices[uncompleted, :]
                indices[:, 1] += 1  # increment to keep masking the next positions

                mask_random = mask_random[uncompleted]

                if self.replace_length != -1:
                    to_keep[indices[:, 0], indices[:, 1]] = 0
                else:
                    # Mask some tokens with a mask token
                    for idxs in indices:
                        input_ids[tuple(idxs)] = self.tokenizer.mask_token_id

                    # Replace a fraction (random_ratio) with a random token
                    rand_tokens = torch.randint(1, len(self.tokenizer), size=(mask_random.sum(),))
                    for idxs, tok in zip(indices[mask_random], rand_tokens):
                        input_ids[tuple(idxs)] = tok

                assert source_length - 1 not in indices[:, 1]

        # Remove some items (e.g. consecutive masks)
        final_ids = torch.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
        for keeper_idx, keeper in enumerate(to_keep):
            seq = input_ids[keeper_idx, keeper]
            final_ids[keeper_idx, : seq.size(0)] = seq
        input_ids = final_ids

        if num_inserts > 0:
            input_ids = self.add_insertion_noise(input_ids, num_inserts / input_ids.numel())

        return input_ids

    def add_rolling_noise(self, input_ids):
        offset = torch.randint(1, max(1, input_ids.size(-1) - 1) + 1, (input_ids.size(0),))

        output_ids = torch.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
        for seq_idx in range(output_ids.size(0)):
            output_ids[seq_idx] = torch.cat(
                (
                    input_ids[seq_idx, 0:1],
                    input_ids[seq_idx, offset[seq_idx] : -1],
                    input_ids[seq_idx, 1 : offset[seq_idx]],
                    input_ids[seq_idx, -1:],
                ),
                dim=0,
            )
        return output_ids

    def add_insertion_noise(self, input_ids, p):
        """As currently implemented, all sequences in this batch will get the same number of added noise (num_noise).
        In fairseq, this is implemented on the dataset-level, meaning that every sequence in a batch may have a
        different number of added noise.

        In addition, because we are now in the data collator, we have to already account for sequence length. In
        Fairseq, the dataset can output any longer length, which can be truncated later. Here, we have to truncate
        directly at the end, after inserting noise. This means, however, that it is possible that a sequence does not
        end with </s>.
        """
        if p == 0.0:
            return input_ids

        seq_num_tokens = input_ids.size(1)
        num_noise = int(math.ceil(seq_num_tokens * p))
        all_results = torch.full(
            (input_ids.size(0), seq_num_tokens + num_noise), fill_value=self.tokenizer.pad_token_id
        )
        for seq_id, sequence in enumerate(input_ids):
            # -2 and + 1 to avoid targetting first and last item?
            noise_indices = torch.randperm(seq_num_tokens + num_noise - 2)[:num_noise] + 1
            noise_mask = torch.zeros(size=(seq_num_tokens + num_noise,), dtype=torch.bool)
            noise_mask[noise_indices] = 1

            result = torch.LongTensor(seq_num_tokens + num_noise).fill_(-1)

            num_random = int(math.ceil(num_noise * self.random_ratio))
            result[noise_indices[num_random:]] = self.tokenizer.mask_token_id
            result[noise_indices[:num_random]] = torch.randint(low=1, high=len(self.tokenizer), size=(num_random,))

            result[~noise_mask] = sequence

            assert (result >= 0).all()
            all_results[seq_id] = result

        all_results = all_results[:, :seq_num_tokens]
        return all_results


def main():
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return len(self.data)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("facebook/bart-base")
    # Two sequences, containing a padtoken to separate sentences
    text = [
        "A cookie is a baked or cooked snack or dessert that is typically small, flat and sweet."
        f"{tokenizer.pad_token}It usually contains flour, sugar, egg, and some type of oil, fat, or butter."
        f"{tokenizer.pad_token}It may include other ingredients such as raisins, oats, chocolate chips, nuts, etc.",
        "Biscuit or cookie variants include sandwich biscuits, such as custard creams."
        f"{tokenizer.pad_token}Chewier biscuits are sometimes called cookies",
    ]
    max_length = max([len(tokenizer(s)["input_ids"]) for s in text])
    encoded = [tokenizer(s, padding="max_length", truncation=True, max_length=max_length) for s in text]
    dataset = DummyDataset(data=encoded)
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    dl = DataLoader(
        dataset,
        collate_fn=DataCollatorForBartDenoisingLM(tokenizer, model.config.decoder_start_token_id),
        batch_size=2,
    )

    for b in dl:
        print(tokenizer.batch_decode(b["labels"]))
        print(tokenizer.batch_decode(b["input_ids"]))


if __name__ == "__main__":
    main()
