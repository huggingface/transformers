from typing import Union

import torch

from ...utils.import_utils import requires


@requires(backends=("torch",))
class MambaCache:
    """
    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a smaller batch size is used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, MambaForCausalLM, MambaCache

        >>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

        >>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = MambaCache(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values
        MambaCache()
        ```
    """

    is_compileable = True

    # TODO (joao): add layer_device_map arg and update code in `generate` accordingly
    def __init__(
        self,
        config,
        max_batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: Union[torch.device, str, None] = None,
    ):
        self.max_batch_size = max_batch_size
        self._dtype = dtype
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel

        self.conv_states: list[torch.Tensor] = []
        self.ssm_states: list[torch.Tensor] = []
        device = torch.device(device) if device is not None else None
        for _ in range(config.num_hidden_layers):
            conv_state: torch.Tensor = torch.zeros(
                self.max_batch_size,
                self.intermediate_size,
                self.conv_kernel_size,
                device=device,
                dtype=self._dtype,
            )
            ssm_state: torch.Tensor = torch.zeros(
                self.max_batch_size,
                self.intermediate_size,
                self.ssm_state_size,
                device=device,
                dtype=self._dtype,
            )

            torch._dynamo.mark_static_address(conv_state)
            torch._dynamo.mark_static_address(ssm_state)
            self.conv_states.append(conv_state)
            self.ssm_states.append(ssm_state)

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        # This `if` blocks is only reached in multigpu and if `layer_device_map` is not passed. It is used
        # when the cache is initialized in the forward pass (e.g. Mamba)
        if self.conv_states[layer_idx].device != new_conv_state.device:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].to(new_conv_state.device)

        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(device=conv_state.device, dtype=conv_state.dtype)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]

    def reset(self):
        for layer_idx in range(len(self.conv_states)):
            # In-place ops prevent breaking the static address
            self.conv_states[layer_idx].zero_()
            self.ssm_states[layer_idx].zero_()


__all__ = ["MambaCache"]
