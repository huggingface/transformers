import os

import torch

from ...utils import logging


logger = logging.get_logger(__name__)


def sequential_gemm(input, weight, tokens_per_expert):
    """
    Compute the matrix multiplication (GEMM) for each expert sequentially. This approach is computationally inefficient, especially when dealing with a large number of experts.

    Args:
        input (torch.Tensor): Input tensor of shape (num_tokens, in_features).
        weight (torch.Tensor): Weight tensor of shape (num_experts, in_features, out_features).
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

    Returns:
        torch.Tensor: Output tensor of shape (num_tokens, out_features).
    """
    num_tokens = input.shape[0]
    out_features = weight.shape[-1]
    output = torch.zeros(num_tokens, out_features, dtype=input.dtype, device=input.device)

    cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
    # Insert zero at the begining for offset index's convenience
    zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
    cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

    for expert_num in range(weight.shape[0]):
        start = cumsum_num_tokens[expert_num]
        end = cumsum_num_tokens[expert_num + 1]
        tokens = input[start:end]

        out = torch.matmul(tokens, weight[expert_num])
        output[start:end] = out
    return output


try:
    from grouped_gemm.ops import gmm as experts_gemm

    if os.environ.get("USE_GROUPED_GEMM", "1") == "0":
        logger.warning("environment variable USE_GROUPED_GEMM is set to 0, using sequential GEMM instead.")
        experts_gemm = sequential_gemm
except ImportError:
    logger.warning("`grouped_gemm` is not installed, using sequential GEMM, which is slower.")
    experts_gemm = sequential_gemm
