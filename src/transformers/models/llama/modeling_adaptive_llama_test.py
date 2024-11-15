import pytest

import torch

from .configuration_llama import LlamaConfig
from .modeling_adaptive_llama import AdaptiveFanIn, AdaptiveFanOut, AdaptiveFanInOutput, AdaptiveFanOutOutput

def test_adaptive_fan_in_no_merge():
    config = LlamaConfig(hidden_size=256)

    afin = AdaptiveFanIn(config)

    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ])
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merging_map = torch.zeros([batch_size, seq_len - 1])

    afin_output = afin.forward(hidden_states, attention_mask, special_embeddings_mask, merging_map=merging_map)

    assert afin_output.attention_mask.shape[1] == seq_len
    assert afin_output.hidden_state.shape[1] == seq_len
    assert afin_output.merged_embeddings_counts.shape[1] == seq_len


def test_adaptive_fan_in_all_merge():
    config = LlamaConfig(hidden_size=256)

    afin = AdaptiveFanIn(config)

    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ])
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merge_map = torch.ones([batch_size, seq_len - 1])

    afin_output = afin.forward(hidden_states, attention_mask, special_embeddings_mask, merging_map=merge_map)

    assert afin_output.attention_mask.shape[1] == 3 # bos + merged_embedding + eos
    assert afin_output.hidden_state.shape[1] == 3 # bos + merged_embedding + eos
    assert afin_output.merged_embeddings_counts.shape[1] == 3 # bos + merged_embedding + eos
    assert afin_output.special_embeddings_mask.shape[1] == 3 # bos + merged_embedding + eos

    return

def test_adaptive_fan_in_all_but_first_merge():
    config = LlamaConfig(hidden_size=256)

    afin = AdaptiveFanIn(config)

    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ])
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merge_map = torch.ones([batch_size, seq_len - 1])
    merge_map[:, 1] = 0

    afin_output = afin.forward(hidden_states, attention_mask, special_embeddings_mask, merging_map=merge_map)

    assert afin_output.attention_mask.shape[1] == 4 # bos + original_embedding + merged_embedding + eos
    assert afin_output.hidden_state.shape[1] == 4 # bos + original_embedding + merged_embedding + eos
    assert afin_output.merged_embeddings_counts.shape[1] == 4 # bos + original_embedding + merged_embedding + eos
    assert afin_output.special_embeddings_mask.shape[1] == 4 # bos + original_embedding + merged_embedding + eos

    assert (afin_output.hidden_state[:, 1] == hidden_states[:, 1]).all()
    assert (afin_output.hidden_state[:, 0] == hidden_states[:, 0]).all()
    assert (afin_output.hidden_state[:, -1] == hidden_states[:, -1]).all()

    return


def test_adaptive_fan_out():
    config = LlamaConfig(hidden_size=256)
    afout = AdaptiveFanOut(config)


def test_adaptive_fan_in_fan_out():
    pass


def test_backward():
    pass