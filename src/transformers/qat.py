import torch


class QATMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # behaves like normal torch.matmul unless a SparseML QuantizationModifier
        # is initialized
        self.wrap_qat = True
        self.qat_wrapper_kwargs = {
            "num_inputs": 2,
            "input_qconfigs": ["asymmetric", "symmetric"],
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return torch.matmul(a, b)
