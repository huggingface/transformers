from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from ..tokenization_utils import PreTrainedTokenizer
from ..tokenization_utils_base import BatchEncoding, PaddingStrategy
from ..tokenization_utils_fast import PreTrainedTokenizerFast


InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset
and collate them into a batch, as a dictionary of Tensors.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])


def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding
            index) among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
              single sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
            >= 7.5 (Volta).
    """

    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


@dataclass
class DataCollatorForSOP(DataCollatorForLanguageModeling):
    """
    Data collator used for sentence order prediction task.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and sentence order prediction
    """

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [example["input_ids"] for example in examples]
        input_ids = self._tensorize_batch(input_ids)
        input_ids, labels, attention_mask = self.mask_tokens(input_ids)

        token_type_ids = [example["token_type_ids"] for example in examples]
        # size of segment_ids varied because randomness, padding zero to the end as the orignal implementation
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        sop_label_list = [example["sentence_order_label"] for example in examples]
        sentence_order_label = torch.stack(sop_label_list)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sentence_order_label": sentence_order_label,
        }

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels/attention_mask for masked language modeling: 80% MASK, 10% random, 10% original.
        N-gram not applied yet.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # probability be `1` (masked), however in albert model attention mask `0` means masked, revert the value
        attention_mask = (~masked_indices).float()
        if self.tokenizer._pad_token is not None:
            attention_padding_mask = labels.eq(self.tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=1.0)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens, -100 is default for CE compute

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, attention_mask


@dataclass
class DataCollatorForPermutationLanguageModeling:
    """
    Data collator used for permutation language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for permutation language modeling with procedures specific to XLNet
    """

    tokenizer: PreTrainedTokenizer
    plm_probability: float = 1 / 6
    max_span_length: int = 5  # maximum length of a span of masked tokens

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        inputs, perm_mask, target_mapping, labels = self.mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.Tensor(e) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The masked tokens to be predicted for a particular sequence are determined by the following algorithm:
            0. Start from the beginning of the sequence by setting ``cur_len = 0`` (number of tokens processed so far).
            1. Sample a ``span_length`` from the interval ``[1, max_span_length]`` (length of span of tokens to be masked)
            2. Reserve a context of length ``context_length = span_length / plm_probability`` to surround span to be masked
            3. Sample a starting point ``start_index`` from the interval ``[cur_len, cur_len + context_length - span_length]`` and mask tokens ``start_index:start_index + span_length``
            4. Set ``cur_len = cur_len + context_length``. If ``cur_len < max_len`` (i.e. there are tokens remaining in the sequence to be processed), repeat from Step 1.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer."
            )

        if inputs.size(1) % 2 != 0:
            raise ValueError(
                "This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details."
            )

        labels = inputs.clone()
        # Creating the mask and target_mapping tensors
        masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)
        target_mapping = torch.zeros((labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32)

        for i in range(labels.size(0)):
            # Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
            cur_len = 0
            max_len = labels.size(1)

            while cur_len < max_len:
                # Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
                span_length = torch.randint(1, self.max_span_length + 1, (1,)).item()
                # Reserve a context of length `context_length = span_length / plm_probability` to surround the span to be masked
                context_length = int(span_length / self.plm_probability)
                # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
                start_index = cur_len + torch.randint(context_length - span_length + 1, (1,)).item()
                masked_indices[i, start_index : start_index + span_length] = 1
                # Set `cur_len = cur_len + context_length`
                cur_len += context_length

            # Since we're replacing non-masked tokens with -100 in the labels tensor instead of skipping them altogether,
            # the i-th predict corresponds to the i-th token.
            target_mapping[i] = torch.eye(labels.size(1))

        special_tokens_mask = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()],
            dtype=torch.bool,
        )
        masked_indices.masked_fill_(special_tokens_mask, value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            masked_indices.masked_fill_(padding_mask, value=0.0)

        # Mask indicating non-functional tokens, where functional tokens are [SEP], [CLS], padding, etc.
        non_func_mask = ~(padding_mask & special_tokens_mask)

        inputs[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        perm_mask = torch.zeros((labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32)

        for i in range(labels.size(0)):
            # Generate permutation indices i.e. sample a random factorisation order for the sequence. This will
            # determine which tokens a given token can attend to (encoded in `perm_mask`).
            # Note: Length of token sequence being permuted has to be less than or equal to reused sequence length
            # (see documentation for `mems`), otherwise information may leak through due to reuse. In this implementation,
            # we assume that reused length is half of sequence length and permutation length is equal to reused length.
            # This requires that the sequence length be even.

            # Create a linear factorisation order
            perm_index = torch.arange(labels.size(1))
            # Split this into two halves, assuming that half the sequence is reused each time
            perm_index = perm_index.reshape((-1, labels.size(1) // 2)).transpose(0, 1)
            # Permute the two halves such that they do not cross over
            perm_index = perm_index[torch.randperm(labels.size(1) // 2)]
            # Flatten this out into the desired permuted factorisation order
            perm_index = torch.flatten(perm_index.transpose(0, 1))
            # Set the permutation indices of non-masked (non-functional) tokens to the
            # smallest index (-1) so that:
            # (1) They can be seen by all other positions
            # (2) They cannot see masked positions, so there won't be information leak
            perm_index.masked_fill_(~masked_indices[i] & non_func_mask[i], -1)
            # The logic for whether the i-th token can attend on the j-th token based on the factorisation order:
            # 0 (can attend): If perm_index[i] > perm_index[j] or j is neither masked nor a functional token
            # 1 (cannot attend): If perm_index[i] <= perm_index[j] and j is either masked or a functional token
            perm_mask[i] = (
                perm_index.reshape((labels.size(1), 1)) <= perm_index.reshape((1, labels.size(1)))
            ) & masked_indices[i]

        return inputs, perm_mask, target_mapping, labels


@dataclass
class DataCollatorForNextSentencePrediction:
    """
    Data collator used for next sentence prediction.
    - collates examples which contains pre-generated negative examples
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    block_size: int = 512
    short_seq_probability: float = 0.1
    nsp_probability: float = 0.5
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        The input should contain negative examples, :class:`~transformers.DataCollatorForNextSentencePrediction` will not generate any negative examples.
        Args:
            examples (:obj:`List[Dict]`): Each dictionary should have the following keys:
                  - ``tokens_a``: A sequence of tokens, which should appear before ``tokens_b`` in the text.
                  - ``tokens_b``: A sequence of tokens, which should appear after ``tokens_a`` in the text.
                  - ``is_random_next``: 1 if this pair is generated randomly, else 0.
        """

        tokens_a = [e["tokens_a"] for e in examples]
        tokens_b = [e["tokens_b"] for e in examples]
        nsp_labels = [1 if e["is_random_next"] else 0 for e in examples]

        input_ids = []
        segment_ids = []
        attention_masks = []

        assert len(tokens_a) == len(tokens_b)
        for i in range(len(tokens_a)):
            input_id, attention_mask, segment_id = self.create_features_from_example(tokens_a[i], tokens_b[i])
            input_ids.append(input_id)
            segment_ids.append(segment_id)
            attention_masks.append(attention_mask)
        if self.mlm:
            input_ids, mlm_labels = self.mask_tokens(self._tensorize_batch(input_ids))
        else:
            input_ids = self._tensorize_batch(input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": self._tensorize_batch(attention_masks),
            "token_type_ids": self._tensorize_batch(segment_ids),
            "masked_lm_labels": mlm_labels if self.mlm else None,
            "next_sentence_label": torch.tensor(nsp_labels),
        }
        if self.mlm:
            result["masked_lm_labels"] = mlm_labels
        return result

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def create_features_from_example(self, tokens_a, tokens_b):
        """Creates examples for a single document."""

        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        tokens_a, tokens_b, _ = self.tokenizer.truncate_sequences(
            tokens_a,
            tokens_b,
            num_tokens_to_remove=len(tokens_a) + len(tokens_b) - max_num_tokens,
            truncation_strategy="longest_first",
        )

        input_id = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        attention_mask = [1] * len(input_id)
        segment_id = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        assert len(input_id) <= self.block_size

        # pad
        while len(input_id) < self.block_size:
            input_id.append(0)
            attention_mask.append(0)
            segment_id.append(0)

        input_id = torch.tensor(input_id)
        attention_mask = torch.tensor(attention_mask)
        segment_id = torch.tensor(segment_id)

        return input_id, attention_mask, segment_id

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
