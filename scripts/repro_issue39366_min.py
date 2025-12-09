import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


class DummyConfig(PretrainedConfig):
    """
    Minimal configuration for reproducing the init dtype issue; provides required fields for initialization.
    """

    model_type = "dummy"

    def __init__(self, initializer_range: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.initializer_range = initializer_range


class DummyModel(PreTrainedModel):
    """
    Minimal model to reproduce the issue: create a Linear layer with int8 weights, then trigger
    transformers' generic initialization to demonstrate that calling normal_ on non-floating dtypes fails.
    """

    config_class = DummyConfig

    def __init__(self, config: DummyConfig):
        super().__init__(config)
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight = nn.Parameter(
                torch.empty_like(self.linear.weight, dtype=torch.int8), requires_grad=False
            )

        # Trigger library's generic initialization logic
        self._init_weights(self.linear)


def main():
    """
    Run minimal reproduction: previously raised RuntimeError when initializing int8 weights; with the fix,
    initialization is safely skipped for non-floating dtypes.
    """

    _ = DummyModel(DummyConfig())
    print("dtype-safe init: OK (no crash)")


if __name__ == "__main__":
    main()

