import torch


def mask_inputs(input_ids: torch.Tensor, mask_token_id, mask_probability, tokens_to_ignore, max_predictions_per_seq, proposal_distribution=1.0):
    inputs_which_can_be_masked = torch.ones_like(input_ids)
    for token in tokens_to_ignore:
        inputs_which_can_be_masked -= torch.eq(input_ids, token).long()

    total_number_of_tokens = input_ids.shape[-1]

    # Identify the number of tokens to be masked, which should be: 1 < num < max_predictions per seq.
    # It is set to be: n_tokens * mask_probability, but is truncated if it goes beyond bounds.
    number_of_tokens_to_be_masked = torch.max(
        torch.tensor(1),
        torch.min(
            torch.tensor(max_predictions_per_seq),
            torch.tensor(total_number_of_tokens * mask_probability, dtype=torch.long)
        )
    )

    # The probability of each token being masked
    sample_prob = proposal_distribution * inputs_which_can_be_masked
    sample_prob /= torch.sum(sample_prob)
    # Should be passed through a log function here

    # Weight of each position: 1 the position will be masked, 0 the position won't be masked
    masked_lm_weights = torch.tensor([0] * max_predictions_per_seq, dtype=torch.bool)
    masked_lm_weights[:number_of_tokens_to_be_masked] = True

    # Sample from the probabilities
    masked_lm_positions = sample_prob.multinomial(max_predictions_per_seq)

    # Apply the weights to the positions
    masked_lm_positions *= masked_lm_weights.long()

    # Gather the IDs from the positions
    masked_lm_ids = input_ids.gather(-1, masked_lm_positions)

    # Apply weights to the IDs
    masked_lm_ids *= masked_lm_weights.long()

    replace_with_mask_positions = masked_lm_positions * (torch.rand(masked_lm_positions.shape) < 0.85)

    # Replace the input IDs with masks on given positions
    masked_input_ids = input_ids.scatter(-1, replace_with_mask_positions, mask_token_id)

    # Updates to index 0 should be ignored
    masked_input_ids[..., 0] = input_ids[..., 0]

    return masked_input_ids, masked_lm_positions


def gather_positions(sequence, positions):
    batch_size, sequence_length, dimension = sequence.shape
    position_shift = (sequence_length * torch.arange(batch_size)).unsqueeze(-1)
    flat_positions = torch.reshape(positions + position_shift, [-1]).long()
    flat_sequence = torch.reshape(sequence, [batch_size * sequence_length, dimension])
    gathered = flat_sequence.index_select(0, flat_positions)
    return torch.reshape(gathered, [batch_size, -1, dimension])
