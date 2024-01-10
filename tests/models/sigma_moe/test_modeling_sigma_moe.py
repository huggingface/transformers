import torch
from transformers.models.sigma_moe.moe_layer import SigmaMoELayer
from transformers.testing_utils import slow


# @slow
def test_equivalence_cpu_cuda():
    torch.manual_seed(0)

    d_model = 128
    n_experts = 4
    expert_size = 64
    k = 2

    cpu_moe = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    )

    cuda_moe = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=k,
    ).cuda()

    cuda_moe.load_state_dict(cpu_moe.state_dict())

    inp = torch.randn(5, 10, 128)
    cpu_x, _ = cpu_moe(inp)
    cuda_x, _ = cuda_moe(inp.cuda())

    assert torch.allclose(cpu_x, cuda_x.cpu(), atol=1e-4)

if __name__ == "__main__":
    test_equivalence_cpu_cuda()
