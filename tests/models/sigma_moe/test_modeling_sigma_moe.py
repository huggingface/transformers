from transformers.models.sigma_moe.moe_layer import SigmaMoELayer
from transformers import SigmaMoEForCausalLM, AutoTokenizer
from transformers.testing_utils import (
    slow,
    require_torch,
    require_torch_gpu,
    torch_device,
)
from transformers.utils import is_torch_available

if is_torch_available():
    import torch


@slow
@require_torch
@require_torch_gpu
def test_equivalence_cpu_cuda():
    """
    Test whether the CPU and GPU implementations of the MoE layer are equivalent.
    """
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
    ).to(torch_device)

    cuda_moe.load_state_dict(cpu_moe.state_dict())

    inp = torch.randn(5, 10, 128)
    cpu_x, _ = cpu_moe(inp)
    cuda_x, _ = cuda_moe(inp.to(torch_device))

    assert torch.allclose(cpu_x.to(torch_device), cuda_x, atol=1e-4)


@slow
@require_torch
def test_model_correctness():
    """
    Test whether the model outputs are correct compared to the pretrained model.
    """
    torch.manual_seed(0)
    model = SigmaMoEForCausalLM.from_pretrained("ibm-aimc/sigma-moe-small")
    tokenizer = AutoTokenizer.from_pretrained("ibm-aimc/sigma-moe-small")
    model.to(torch_device)
    model.eval()
    test_data = tokenizer("this is a test")
    model_outputs = model(
        input_ids=torch.tensor(test_data.input_ids).unsqueeze(0).to(torch_device),
        attention_mask=torch.tensor(test_data.attention_mask)
        .unsqueeze(0)
        .to(torch_device),
        output_hidden_states=False,
        output_attentions=False,
    )
    assert torch.allclose(
        model_outputs.logits[0, 0, :3],
        torch.tensor([-10.3830, -10.3886, -10.4600]).to(torch_device),
        atol=1e-4,
    )
    assert torch.allclose(
        model_outputs.all_reg_losses[0],
        torch.tensor(-2.5466).to(torch_device),
        atol=1e-4,
    )
