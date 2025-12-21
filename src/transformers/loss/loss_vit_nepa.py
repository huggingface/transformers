import torch.nn.functional as F


def ViTNepaPreTrainingLoss(hidden_states_in, hidden_states_out, shift: bool = True):
    """
    similarity loss between two hidden states.

    Args:
        hidden_states_in:  [B, T, D]  input hidden states
        hidden_states_out: [B, T, D]  output hidden states (prediction)
        shift: if True, compare h_out[:, :-1] with h_in[:, 1:]
               else, compare h_out with h_in (position-wise)

    Returns:
        scalar loss (negative cosine similarity)
    """
    # detach target
    hidden_states_in = hidden_states_in.detach()

    if shift:
        # shift one step forward
        p = hidden_states_out[:, :-1, :]  # predict next
        z = hidden_states_in[:, 1:, :]  # target is next hidden state
    else:
        # same-position matching
        p = hidden_states_out
        z = hidden_states_in

    # normalize
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    # negative cosine similarity
    loss = -(p * z).sum(dim=-1).mean()
    return loss
