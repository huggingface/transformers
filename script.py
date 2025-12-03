import torch


class DummyModule(torch.nn.Module):
    def forward(self, x):
        return x * 2


if __name__ == "__main__":
    model = DummyModule()
    input_tensor = torch.tensor([-1.0, -2.0, -3.0])
    output = model(input_tensor)
