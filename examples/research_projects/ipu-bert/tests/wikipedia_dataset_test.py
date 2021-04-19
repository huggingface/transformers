# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Read wikipedia datafiles and see that they conform to the schema
# Schema:
#         input_ids: tokens after masking
#         attention_mask:  0 if is a padding token, 1 otherwise
#         token_type_ids: sentence 0/1
#         labels: 0 if corresponding token not masked, original value otherwise
#         nsp: 1 if is next sentence, 0 otherwise
#
# Here is an example data point:
#
# input_ids - the encoded tokens after mlm maskng and padding
# tensor([  101,  1996,  2536,   103, 21670,  1998,  2385,  3461,  8333,  2915,
#          2011,  1996,  3626,  1997,  9348,  2620,  2020,   103,  1998,  2207,
#           103,  9274,  1999,  1996,  3440,  4516,  1000,   103, 23736, 12879,
#          1024,  9348,  2620,  1000,  1010,   103,  2011, 17754, 10635,  1012,
#           102,   103,  2804,  1010,   103,  3152,  2207,  1010,  1999,  2494,
#          1010,  1037,  2093,  1011,  5860,   103,  2275,  4820,  2035,  1997,
#          9274,  1005,  1055,  2694,   103,  2385,  2050,   103,  2143,  8333,
#          3141,   103,  1996,  3260,  1010,  2164,  2035,  2694, 21670,  2013,
#          2686,  1010,  2731,  1998,  4888,   103,  1010,  1998,  4367,  4620,
#          2579,  1999,  3462,  1012,   102,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0])
#
# attention_mask - 1 is valid input, 0 is padding.
# tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0])
#
# Type IDs. First 0s are first sentence. Then the 1s are the second sentence. The remaining 0s are padding.
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0])
#
# Mask LM positions
# tensor([ 3, 17, 20, 27, 35, 41, 44, 55, 64, 67, 71, 78, 85, 93,  0,  0,  0,  0,
#          0,  0])
#
# Mask LM Labels: Ground truth of the mlm masked tokens.
#                 0 is not Masked. Any other value is the ground truth token before masking.
# tensor([ 2547,  9227,  2011,  2139,  4354,  1999, 12076,  4966,  1998,  3461,
#          2000, 21670,  8333,  1012,     0,     0,     0,     0,     0,     0])

# NSP. 1 means the two sentences came after each other in the original text.
#      0 means they didn't
# tensor([0])

import numpy as np
from collections import Counter
from tqdm import tqdm
from utils import parse_bert_args
from bert_data import get_dataset
from bert_ipu import get_options
from poptorch import DataLoader
import transformers
import pytest


def check_dimensions(config, tokens, attn_mask, token_types, mask_lm_positions, labels, nsp):
    assert (tokens.shape == (config.sequence_length, ))
    assert (attn_mask.shape == (config.sequence_length, ))
    assert (token_types.shape == (config.sequence_length, ))
    assert (mask_lm_positions.shape == (config.mask_tokens, ))
    assert (labels.shape == (config.mask_tokens, ))
    assert (nsp.shape == (1, ))


def check_tokens(config, tokens, mask_lm_position, labels):
    """
    First sentence starts with 101 ([CLS])
    Second sentence starts and ends with 102 ([SEP])
    Padding at end of sequence is 0s, this won't be masked.
    Mask token is 101 ([MASK])
    """
    # Remove masked tokens for tests
    tokens_nomask = np.delete(tokens, mask_lm_position[labels != 0])
    assert tokens_nomask[0] == 101  # sequence starts with [CLS]
    # Only 1 [CLS]
    assert tokens_nomask[tokens_nomask == 101].shape[0] == 1
    # Sentence 1 and 2 finish by [SEP]
    assert tokens_nomask[tokens_nomask == 102].shape[0] == 2

    # After second 102 [SEP] it should be all padding
    indices_102 = np.where(tokens_nomask == 102)[0]
    assert np.all(tokens_nomask[indices_102[1] + 1:] == 0)

    # The unmasked tokens in the sentences minus special characters
    # have values between 999 and vocab_size
    slice_0 = slice(1, indices_102[0])
    slice_1 = slice(indices_102[0] + 1, indices_102[1])
    sentence_0 = tokens_nomask[slice_0]
    sentence_1 = tokens_nomask[slice_1]

    special_tokens = np.array([100, 101, 102, 103, 0])
    normal_tokens_0 = sentence_0[~np.in1d(sentence_0, special_tokens)]
    normal_tokens_1 = sentence_1[~np.in1d(sentence_1, special_tokens)]

    assert np.all((normal_tokens_0 >= 999) &
                  (normal_tokens_0 < config.vocab_size))
    assert np.all((normal_tokens_1 >= 999) &
                  (normal_tokens_1 < config.vocab_size))


def check_attention_mask(attn_mask, tokens):
    """
    Starts with sequence of 1s.
    If padded will continue with sequence of 0s
    """
    # Can only be 0 or 1
    assert np.all((attn_mask == 0) | (attn_mask == 1))

    idx_zeros = np.where(attn_mask == 0)[0]
    if idx_zeros.size > 0:
        idx = idx_zeros[0]
        assert np.all(attn_mask[:idx] == 1)
        assert np.all(attn_mask[idx:] == 0)

        # Check that the token padding (0) matches attn_mask
        assert tokens[idx - 1] == 102
        assert np.all(tokens[idx:] == 0)
    else:
        assert np.all(attn_mask[:] == 1)


def check_token_type(token_types):
    """
    This is a mask for the two different sentences plus padding
    Sentence 0: 0,0,0,0,0
    Sentence 1: 1,1,1,1,1,1,1
    Padding: 0,0,0,0,0,0
    """
    # Can only be 0 or 1
    assert np.all((token_types == 0) | (token_types == 1))

    # Find sentence boundary
    idx_ones = np.where(token_types == 1)[0]

    # Should be two sentences
    assert idx_ones.size > 0

    idx = idx_ones[0]
    assert np.all(token_types[:idx] == 0)

    idx_zeros = np.where(token_types[idx:] == 0)[0]
    if idx_zeros.size > 0:
        # There is padding
        idx_pad = idx_zeros[0] + idx
        assert np.all(token_types[idx:idx_pad] == 1)
        assert np.all(token_types[idx_pad:] == 0)
    else:
        assert np.all(token_types[idx:] == 1)


def check_mask_lm_positions(config, mask_lm_positions):
    """
    Position of MLM tokens.
    Must be less then the sequence_length of tokens.
    """
    assert np.all(mask_lm_positions < config.sequence_length)


def check_labels(config, tokens, mask_lm_positions, labels):
    """
    MLM Labels

    value should be 0, 103, or in [0, vocab_size)?
    80% it will be 103
    10% it will be the same
    10% it will be random int between [0, vocab_size)
    """
    masked_tokens = tokens[mask_lm_positions[labels != 0]]
    assert np.all((masked_tokens >= 0) & (masked_tokens < config.vocab_size))


def mask_type_count(tokens, mask_lm_positions, labels):
    masked_tokens = tokens[mask_lm_positions[labels != 0]]
    true_tokens = labels[labels != 0]
    mask_types = Counter({"103": 0, "same": 0, "random": 0})
    mask_types["103"] += np.where(masked_tokens == 103)[0].shape[0]
    mask_types["same"] += np.where(masked_tokens == true_tokens)[0].shape[0]
    mask_types["random"] += np.where(
        (masked_tokens != 103) & (masked_tokens != true_tokens))[0].shape[0]

    assert sum(mask_types.values()) == masked_tokens.shape[0]
    return mask_types


def check_nsp(nsp):
    assert (nsp == 0) or (nsp == 1)


@pytest.mark.skip_longtest_needs_dataset
def test_wikipedia_dataset():
    args = "--config demo_tiny_128".split()
    config = transformers.BertConfig(**(vars(parse_bert_args(args))))
    config.vocab_size = 30522
    config.input_files = ["data/wikipedia/128/wiki_000.tfrecord"]

    num_tokens = 0
    replacement_counts = Counter({"103": 0, "same": 0, "random": 0})

    dataset = get_dataset(config)
    opts = get_options(config)
    loader = DataLoader(opts,
                        dataset,
                        batch_size=config.batch_size,
                        num_workers=config.dataloader_workers)

    for datum in tqdm(loader):
        tokens, attn_mask, types, mask_lm_pos, labels, nsp = datum
        tokens = tokens.numpy()
        attn_mask = attn_mask.numpy()
        types = types.numpy()
        mask_lm_pos = mask_lm_pos.numpy()
        labels = labels.numpy()
        nsp = nsp.numpy()
        for b in range(config.batch_size):
            check_dimensions(config, tokens[b], attn_mask[b], types[b], mask_lm_pos[b], labels[b], nsp[b])
            check_tokens(config, tokens[b], mask_lm_pos[b], labels[b])
            check_attention_mask(attn_mask[b], tokens[b])
            check_mask_lm_positions(config, mask_lm_pos[b])
            check_labels(config, tokens[b], mask_lm_pos[b], labels[b])
            check_token_type(types[b])
            check_nsp(nsp[b])

            replacement_counts += mask_type_count(tokens[b], mask_lm_pos[b], labels[b])

            # Number of tokens, not including padding
            num_tokens += attn_mask[b, attn_mask[b] == 1].shape[0]

    # Test masked token proportions
    total = sum(replacement_counts.values())
    for k in replacement_counts:
        replacement_counts[k] /= total

    assert (0.79 < replacement_counts["103"] < 0.81)
    assert (0.09 < replacement_counts["same"] < 0.11)
    assert (0.09 < replacement_counts["random"] < 0.11)
    assert (0.14 < total / num_tokens < 0.16)  # should be ~0.15
