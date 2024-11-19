import pytest

import torch

from .configuration_llama import LlamaConfig
from .modeling_adaptive_llama import AdaptiveFanIn, AdaptiveFanOut, AdaptiveFanInOutput, AdaptiveFanOutOutput, AdaptiveLlamaModel

def test_adaptive_fan_in_no_merge():
    config = LlamaConfig(hidden_size=256)

    afin = AdaptiveFanIn(config)

    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ])
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    merging_log_probas = torch.ones([batch_size, seq_len - 1, 2])
    merging_log_probas[:, :, 1] = 0
    merging_log_probas += 1e-4
    merging_log_probas = merging_log_probas.log()

    afin_output = afin.forward(hidden_states, attention_mask, special_embeddings_mask, merging_log_probas=merging_log_probas)

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

    merging_log_probas = torch.ones([batch_size, seq_len - 1, 2])
    merging_log_probas[:, :, 0] = 0
    merging_log_probas += 1e-4
    merging_log_probas = merging_log_probas.log()

    afin_output = afin.forward(hidden_states, attention_mask, special_embeddings_mask, merging_log_probas=merging_log_probas)

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

    merging_log_probas = torch.ones([batch_size, seq_len - 1, 2])
    merging_log_probas[:, :, 0] = 0
    merging_log_probas[:, 1] = torch.tensor([1, 0])
    merging_log_probas += 1e-4
    merging_log_probas = merging_log_probas.log()

    afin_output = afin.forward(hidden_states, attention_mask, special_embeddings_mask, merging_log_probas=merging_log_probas)

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

    batch_size, seq_len, residual_seq_len = 3, 7, 12

    hidden_states = torch.rand([batch_size, seq_len, config.hidden_size])
    attention_mask = torch.tensor([
        [ 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 0, 0],
        [ 1, 1, 1, 1, 0, 0, 0],
    ], dtype=torch.float32)
    merged_embeddings_counts = torch.tensor([
        [ 1, 3, 1, 2, 1, 3, 1],
        [ 1, 2, 2, 2, 1, 0, 0],
        [ 1, 5, 3, 1, 0, 0, 0],
    ], dtype=torch.long)
    residual_hidden_states = torch.rand([batch_size, residual_seq_len, config.hidden_size])
    residual_attention_mask = torch.tensor([
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    ], dtype=torch.float32)

    restored_hidden_states = afout.forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        merged_embeddings_counts=merged_embeddings_counts,
        residual_hidden_states=residual_hidden_states,
        residual_attention_mask=residual_attention_mask,
    )

    assert restored_hidden_states.hidden_state.shape == residual_hidden_states.shape

    restored_hidden_states


def test_adaptive_fan_in_fan_out():
    config = LlamaConfig(hidden_size=256)

    afin = AdaptiveFanIn(config)
    afout = AdaptiveFanOut(config)

    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ], requires_grad=True)
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    afin_output = afin.forward(hidden_states, attention_mask, special_embeddings_mask)
    assert afin_output.hidden_state.grad_fn is not None

    residual_hidden_states = hidden_states
    residual_attention_mask = attention_mask

    afout_output = afout.forward(
        hidden_states=afin_output.hidden_state,
        attention_mask=afin_output.attention_mask,
        merged_embeddings_counts=afin_output.merged_embeddings_counts,
        residual_hidden_states=residual_hidden_states,
        residual_attention_mask=residual_attention_mask,
    )
    restored_hidden_states = afout_output.hidden_state

    assert restored_hidden_states.shape == residual_hidden_states.shape

    restored_hidden_states.backward(torch.rand_like(restored_hidden_states))

    for name, p in afin.named_parameters():
        assert p.grad is not None, f"afin param grad is none: {name}"


def test_adaptive_llama_e2e():
    config = LlamaConfig(hidden_size=256, attn_implementation='eager')

    allama_model = AdaptiveLlamaModel(config)
    batch_size, seq_len = 3, 6
    hidden_states = torch.rand([ batch_size, seq_len, config.hidden_size ], requires_grad=True)
    attention_mask = torch.ones([batch_size, seq_len])
    special_embeddings_mask = torch.zeros([batch_size, seq_len])
    special_embeddings_mask[:, 0] = 1
    special_embeddings_mask[:, -1] = 1

    llama_output = allama_model.forward(inputs_embeds=hidden_states, attention_mask=attention_mask, special_embeddings_mask=special_embeddings_mask, use_cache=False)

    assert llama_output.last_hidden_state.shape == hidden_states.shape
