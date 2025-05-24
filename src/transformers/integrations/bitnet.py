from ..utils import is_accelerate_available, is_torch_available, logging


if is_accelerate_available():
    from accelerate import init_empty_weights

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

logger = logging.get_logger(__name__)


# the weights are ternary so can be represented with 2 bits, and they are packed in uint8 tensors, hence the number of values per item is 4
VALUES_PER_ITEM = 4


def pack_weights(quantized_weights: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor of quantized weights into a compact format using 2 bits per value.

    Parameters:
    -----------
    quantized_weights : torch.Tensor
        A tensor containing ternary quantized weights with values in {-1, 0, 1}. These values are adjusted to
        {0, 1, 2} before being packed.

    Returns:
    --------
    torch.Tensor
        A packed tensor where each element stores 4 quantized values (each using 2 bits) in an 8-bit format.
    """

    original_shape = quantized_weights.shape

    row_dim = (original_shape[0] + VALUES_PER_ITEM - 1) // VALUES_PER_ITEM

    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])

    quantized_weights += 1
    packed = torch.zeros(packed_tensor_shape, device=quantized_weights.device, dtype=torch.uint8)
    unpacked = quantized_weights.to(torch.uint8)

    it = min(VALUES_PER_ITEM, (original_shape[0] // row_dim) + 1)
    for i in range(it):
        start = i * row_dim
        end = min(start + row_dim, original_shape[0])
        packed[: (end - start)] |= unpacked[start:end] << 2 * i

    return packed


@torch.compile
def unpack_weights(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Unpacks a tensor of quantized weights that were stored in a packed format using 2 bits per value.

    Parameters:
    -----------
    packed : torch.Tensor
        A tensor containing packed weights where each element represents 4 quantized values (using 2 bits per value).
    dtype : torch.dtype
        The dtype of the returned Tensor
    Returns:
    --------
    torch.Tensor
        A tensor of unpacked weights, where each value is converted from its packed 2-bit representation.

    Example:
    --------
    packed = torch.tensor([[0b10100001, 0b00011000],
                           [0b10010000, 0b00001010]], dtype=torch.uint8)

    # Unpack the values
    unpacked = unpack_weights(packed)

    # Resulting unpacked tensor
    print(unpacked)
    # Output: tensor([[ 0, -1],
                      [-1,  1],
                      [-1,  1],
                      [-1,  1],
                      [ 1,  0],
                      [ 0, -1],
                      [ 1, -1],
                      [ 1, -1]])

    Explanation of the example:
    ---------------------------
    Let's take the first value for example 0b10100001, we we will only focus on the first column,
    because every element is unpacked across the first dimension
    - First 2 bits: `01` → 0 at [0][0]
    - Second 2 bits: `00` → -1 at [0][2]
    - Third 2 bits: `10` → 1 at [0][4]
    - Fourth 2 bits: `10` → 1 at [0][6]
    the second value of the same row (0b10010000) will give the values for [0][1], [0][3], [0][5], [0][7]

    We subtract 1 because during the packing process, it's easier to work with values like 0, 1, and 2. To make this possible,
    we add 1 to the original ternary weights (which are typically -1, 0, and 1) when packing them. When unpacking, we reverse
    this by subtracting 1 to restore the original ternary values.
    """
    packed_shape = packed.shape

    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = torch.zeros(unpacked_shape, device=packed.device, dtype=torch.uint8)

    for i in range(VALUES_PER_ITEM):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)
        unpacked[start:end] = (packed & mask) >> (2 * i)

    return unpacked.to(dtype) - 1


class BitLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device=None,
        dtype=None,
        use_rms_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.zeros(
                (out_features // VALUES_PER_ITEM, in_features),
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(
                (1),
                dtype=dtype,
                device=device,
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=dtype, device=device))
        else:
            self.bias = None

        # Optional RMSNorm (applied on the activations before quantization).
        self.rms_norm = None
        if use_rms_norm:
            from ..models.llama.modeling_llama import LlamaRMSNorm

            self.rms_norm = LlamaRMSNorm(in_features, eps=rms_norm_eps)

    @torch.compile
    def activation_quant(self, input, num_bits=8):
        """
        Activation function : Performs symmetric, per-token quantization on the input activations.
        Parameters:
        -----------
        x : torch.Tensor
            Input activations to be quantized.
        num_bits : int, optional (default=8)
            Number of bits to use for quantization, determining the quantization range.

        Returns:
        --------
        result : torch.Tensor
            Quantized activation tensor, with values mapped to an `int8` range.
        scale : torch.Tensor
            The per-channel scaling factors used to quantize the tensor.
        """
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1
        scale = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * scale).round().clamp(Qn, Qp)
        return result.to(torch.int8), scale

    @torch.compile
    def post_quant_process(self, input, input_scale, weight_scale):
        out = input / (input_scale * weight_scale)
        return out

    def forward(self, input):
        # Apply RMSNorm on the input if requested.
        if self.rms_norm is not None:
            input = self.rms_norm(input)

        w = self.weight
        w_quant = unpack_weights(w, dtype=self.dtype)
        input_quant, input_scale = self.activation_quant(input)
        y = F.linear(input_quant.to(self.dtype), w_quant)
        y = self.post_quant_process(y, self.weight_scale, input_scale)
        if self.bias is not None:
            y += self.bias.view(1, -1).expand_as(y)
        return y


class WeightQuant(torch.autograd.Function):
    """
    Implements a custom autograd function for weight quantization.
    This performs ternary quantization (-1, 0, 1) based on scaling by the
    mean absolute value of the weights. It uses the Straight-Through Estimator
    (STE) for the backward pass.
    """

    @staticmethod
    @torch.compile
    def forward(ctx, weight):
        dtype = weight.dtype
        weight = weight.float()
        scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        weight = (weight * scale).round().clamp(-1, 1) / scale
        return weight.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ActQuant(torch.autograd.Function):
    """
    Implements a custom autograd function for activation quantization.
    This performs symmetric 8-bit quantization (to the range [-128, 127])
    based on the maximum absolute value along the last dimension (per-token/row scaling).
    It uses the Straight-Through Estimator (STE) for the backward pass.
    """

    @staticmethod
    @torch.compile
    def forward(ctx, activation):
        dtype = activation.dtype
        activation = activation.float()
        scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        activation = (activation * scale).round().clamp(-128, 127) / scale
        return activation.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class AutoBitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        online_quant: bool = False,
        use_rms_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__(in_features, out_features, bias)
        self.online_quant = online_quant
        # Optional RMSNorm
        self.rms_norm = None
        if use_rms_norm:
            from ..models.llama.modeling_llama import LlamaRMSNorm

            self.rms_norm = LlamaRMSNorm(in_features, eps=rms_norm_eps)
        if not online_quant:
            self.register_buffer(
                "weight_scale",
                torch.ones(
                    (1),
                    dtype=dtype,
                    device=device,
                ),
            )
            self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict,
        prefix,
        *args,
        **kwargs,
    ):
        if (prefix + "weight") in state_dict and state_dict[prefix + "weight"].dtype != self.weight.dtype:
            state_dict[prefix + "weight"] = unpack_weights(state_dict[prefix + "weight"], dtype=self.weight.dtype)
        return state_dict

    def forward(self, input):
        # Optional RMSNorm on activations prior to quantization.
        if self.rms_norm is not None:
            input = self.rms_norm(input)

        if self.online_quant:
            weight = WeightQuant.apply(self.weight)
        else:
            weight = self.weight
        input = ActQuant.apply(input)
        output = F.linear(input, weight, self.bias)
        if not self.online_quant:
            output = output * self.weight_scale
        return output


def _replace_with_bitnet_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    pre_quantized=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successful or not.
    """

    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        # Check if the current key is not in the `modules_to_not_convert`
        if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
            with init_empty_weights():
                if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
                    in_features = module.in_features
                    out_features = module.out_features
                    if quantization_config and quantization_config.linear_class == "autobitlinear":
                        model._modules[name] = AutoBitLinear(
                            in_features=in_features,
                            out_features=out_features,
                            bias=module.bias is not None,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            online_quant=(quantization_config.quantization_mode == "online"),
                            use_rms_norm=quantization_config.use_rms_norm,
                            rms_norm_eps=quantization_config.rms_norm_eps,
                        )
                        if quantization_config.quantization_mode == "offline":
                            model._modules[name].requires_grad_(False)
                    else:
                        model._modules[name] = BitLinear(
                            in_features=in_features,
                            out_features=out_features,
                            bias=module.bias is not None,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            use_rms_norm=quantization_config.use_rms_norm if quantization_config else False,
                            rms_norm_eps=quantization_config.rms_norm_eps if quantization_config else 1e-6,
                        )
                        model._modules[name].requires_grad_(False)
                    has_been_replaced = True

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bitnet_linear(
                module,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_bitnet_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    pre_quantized=False,
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `BitLinear158` modules`.

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Each weight will be quantized along the channel.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `BitLinear`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    if quantization_config and quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_bitnet_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        quantization_config,
        pre_quantized=pre_quantized,
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using bitnet but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
