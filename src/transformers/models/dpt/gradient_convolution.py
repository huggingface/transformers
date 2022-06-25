from typing import (Any, Callable, Optional, Sequence, Tuple,
                    Union)
import flax.linen as nn
from jax import lax
from typing import (Any, Callable, NamedTuple, Optional, Sequence, Tuple,
                    Union)
from jax.lax import conv_general_dilated
from flax.linen.initializers import lecun_normal
from flax.linen.initializers import zeros
from flax.linen.module import compact
from jax import lax
import jax.numpy as jnp
import numpy as np

default_kernel_init = lecun_normal()

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]]

PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]]
PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]


class ConvDimensionNumbers(NamedTuple):
  """
  Describes batch, spatial, and feature dimensions of a convolution.

  Args:
    lhs_spec: a tuple of nonnegative integer dimension numbers containing
      `(batch dimension, feature dimension, spatial dimensions...)`.
    rhs_spec: a tuple of nonnegative integer dimension numbers containing
      `(out feature dimension, in feature dimension, spatial dimensions...)`.
    out_spec: a tuple of nonnegative integer dimension numbers containing
      `(batch dimension, feature dimension, spatial dimensions...)`.
  """
  lhs_spec: Sequence[int]
  rhs_spec: Sequence[int]
  out_spec: Sequence[int]

ConvGeneralDilatedDimensionNumbers = Union[
  None, ConvDimensionNumbers, Tuple[str, str, str]]

def _flip_axes(x, axes):
  """Flip ndarray 'x' along each axis specified in axes tuple."""
  for axis in axes:
    x = np.flip(x, axis)
  return x

def conv_general_permutations(dimension_numbers):
  """Utility for convolution dimension permutations relative to Conv HLO."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  lhs_char, rhs_char, out_char = charpairs = ("N", "C"), ("O", "I"), ("N", "C")
  for i, (a, b) in enumerate(charpairs):
    if not dimension_numbers[i].count(a) == dimension_numbers[i].count(b) == 1:
      msg = ("convolution dimension_numbers[{}] must contain the characters "
             "'{}' and '{}' exactly once, got {}.")
      raise TypeError(msg.format(i, a, b, dimension_numbers[i]))
    if len(dimension_numbers[i]) != len(set(dimension_numbers[i])):
      msg = ("convolution dimension_numbers[{}] cannot have duplicate "
             "characters, got {}.")
      raise TypeError(msg.format(i, dimension_numbers[i]))
  if not (set(lhs_spec) - set(lhs_char) == set(rhs_spec) - set(rhs_char) ==
          set(out_spec) - set(out_char)):
    msg = ("convolution dimension_numbers elements must each have the same "
           "set of spatial characters, got {}.")
    raise TypeError(msg.format(dimension_numbers))

  def getperm(spec, charpair):
    spatial = (i for i, c in enumerate(spec) if c not in charpair)
    if spec is not rhs_spec:
      spatial = sorted(spatial, key=lambda i: rhs_spec.index(spec[i]))
    return (spec.index(charpair[0]), spec.index(charpair[1])) + tuple(spatial)

  lhs_perm, rhs_perm, out_perm = map(getperm, dimension_numbers, charpairs)
  return lhs_perm, rhs_perm, out_perm

def conv_dimension_numbers_(lhs_shape, rhs_shape, dimension_numbers
                           ) -> ConvDimensionNumbers:
  """Converts convolution `dimension_numbers` to a `ConvDimensionNumbers`.

  Args:
    lhs_shape: tuple of nonnegative integers, shape of the convolution input.
    rhs_shape: tuple of nonnegative integers, shape of the convolution kernel.
    dimension_numbers: None or a tuple/list of strings or a ConvDimensionNumbers
      object following the convolution dimension number specification format in xla_client.py.

  Returns:
    A `ConvDimensionNumbers` object that represents `dimension_numbers` in the canonical form used by lax functions.
  """
  if isinstance(dimension_numbers, ConvDimensionNumbers):
    return dimension_numbers
  if len(lhs_shape) != len(rhs_shape):
    msg = "convolution requires lhs and rhs ndim to be equal, got {} and {}."
    raise TypeError(msg.format(len(lhs_shape), len(rhs_shape)))

  if dimension_numbers is None:
    iota = tuple(range(len(lhs_shape)))
    return ConvDimensionNumbers(iota, iota, iota)
  elif isinstance(dimension_numbers, (list, tuple)):
    if len(dimension_numbers) != 3:
      msg = "convolution dimension_numbers list/tuple must be length 3, got {}."
      raise TypeError(msg.format(len(dimension_numbers)))
    if not all(isinstance(elt, str) for elt in dimension_numbers):
      msg = "convolution dimension_numbers elements must be strings, got {}."
      raise TypeError(msg.format(tuple(map(type, dimension_numbers))))
    msg = ("convolution dimension_numbers[{}] must have len equal to the ndim "
           "of lhs and rhs, got {} for lhs and rhs shapes {} and {}.")
    for i, elt in enumerate(dimension_numbers):
      if len(elt) != len(lhs_shape):
        raise TypeError(msg.format(i, len(elt), lhs_shape, rhs_shape))

    lhs_spec, rhs_spec, out_spec = conv_general_permutations(dimension_numbers)
    return ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)
  else:
    msg = "convolution dimension_numbers must be tuple/list or None, got {}."
    raise TypeError(msg.format(type(dimension_numbers)))

def _deconv_output_length(input_length,
                          filter_size,
                          padding,
                          output_padding=None,
                          stride=0,
                          dilation=1):
  """Determines the output length of a transposed convolution given the input length.
  Arguments:
  Function modified from Keras.
      input_length: Integer. filter_size: Integer. padding: one of `"SAME"`, `"VALID"`, or a 2-integer tuple.
      output_padding: Integer, amount of padding along the output dimension. Can
        be set to `None` in which case the output length is inferred.
      stride: Integer. dilation: Integer.
  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None

  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'VALID':
      length = input_length * stride + max(filter_size - stride, 0)
    elif padding == 'SAME':
      length = input_length * stride
    else:
      length = ((input_length - 1) * stride + filter_size
                - padding[0] - padding[1])

  else:
    if padding == 'SAME':
      pad = filter_size // 2
      total_pad = pad * 2
    elif padding == 'VALID':
      total_pad = 0
    else:
      total_pad = padding[0] + padding[1]

    length = ((input_length - 1) * stride + filter_size - total_pad +
              output_padding)

  return length

def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank


    if isinstance(padding, Sequence) and len(padding) == rank:
        new_pad = []
        for p in padding:
            if isinstance(p, int):
                new_pad.append((p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                new_pad.append(p)
            else:
                break
        if len(new_pad) == rank:
            return new_pad
    raise ValueError(
        f'Invalid padding format: {padding}, should be str, int,'
        f' or a sequence of len {rank} where each element is an'
        f' int or pair of ints.')

def _compute_adjusted_padding(
    input_size: int,
    output_size: int,
    kernel_size: int,
    stride: int,
    padding: Union[str, Tuple[int, int]],
    dilation: int = 1,
) -> Tuple[int, int]:
  """Computes adjusted padding for desired ConvTranspose `output_size`.
  Ported from DeepMind Haiku.
  """
  kernel_size = (kernel_size - 1) * dilation + 1
  if padding == "VALID":
    expected_input_size = (output_size - kernel_size + stride) // stride
    if input_size != expected_input_size:
      raise ValueError(f"The expected input size with the current set of input "
                       f"parameters is {expected_input_size} which doesn't "
                       f"match the actual input size {input_size}.")
    padding_before = 0
  elif padding == "SAME":
    expected_input_size = (output_size + stride - 1) // stride
    if input_size != expected_input_size:
      raise ValueError(f"The expected input size with the current set of input "
                       f"parameters is {expected_input_size} which doesn't "
                       f"match the actual input size {input_size}.")
    padding_needed = max(0,
                         (input_size - 1) * stride + kernel_size - output_size)
    padding_before = padding_needed // 2
  else:
    padding_before = padding[0]  # type: ignore[assignment]

  expanded_input_size = (input_size - 1) * stride + 1
  padded_out_size = output_size + kernel_size - 1
  pad_before = kernel_size - 1 - padding_before
  pad_after = padded_out_size - expanded_input_size - pad_before
  return (pad_before, pad_after)

def gradient_based_conv_transpose(lhs: Array, rhs: Array, strides: Sequence[int],
                                  padding: Union[str, Sequence[Tuple[int, int]]],
                                  output_padding: Optional[Sequence[int]] = None,
                                  output_shape: Optional[Sequence[int]] = None,
                                  dilation: Optional[Sequence[int]] = None,
                                  dimension_numbers: ConvGeneralDilatedDimensionNumbers = None,
                                  transpose_kernel: bool = True,
                                  precision: PrecisionLike = None) -> Array:
  """Convenience wrapper for calculating the N-d transposed convolution.
  Args:
  Much like *conv_transpose*, this function calculates transposed convolutions via fractionally strided convolution
  rather than calculating the gradient (transpose) of a forward convolution. However, the latter is more common among
  deep learning frameworks, such as TensorFlow, PyTorch, and Keras. This function provides the same set of APIs to help:
  reproduce results in these frameworks.
    lhs: a rank *n+2* dimensional input array. rhs: a rank *n+2* dimensional array of kernel weights. strides: sequence
    of *n* integers, amounts to strides of the corresponding forward convolution. padding: *"SAME"*, *"VALID"*, or a
    sequence of *n* integer 2-tuples that controls
      the before-and-after padding for each *n* spatial dimension of the corresponding forward convolution.
    output_padding: A sequence of integers specifying the amount of padding along
      each spacial dimension of the output tensor, used to disambiguate the output shape of transposed convolutions
      when the stride is larger than 1. (see a detailed description at
      1https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) The amount of output padding along a
      given dimension must be lower than the stride along that same dimension. If set to *None* (default), the output
      shape is inferred. If both *output_padding* and *output_shape* are specified, they have to be mutually
      compatible.
    output_shape: Output shape of the spatial dimensions of a transpose
      convolution. Can be *None* or an iterable of *n* integers. If a *None* value is given (default), the shape is
      automatically calculated. Similar to *output_padding*, *output_shape* is also for disambiguating the output shape
      when stride > 1 (see also https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose) If both
      *output_padding* and *output_shape* are specified, they have to be mutually compatible.
    dilation: *None*, or a sequence of *n* integers, giving the
      dilation factor to apply in each spatial dimension of *rhs*. Dilated convolution is also known as atrous
      convolution.
    dimension_numbers: tuple of dimension descriptors as in
      lax.conv_general_dilated. Defaults to tensorflow convention.
    transpose_kernel: if *True* flips spatial axes and swaps the input/output
      channel axes of the kernel. This makes the output of this function identical to the gradient-derived functions
      like keras.layers.Conv2DTranspose and torch.nn.ConvTranspose2d applied to the same kernel. Although for typical
      use in neural nets this is unnecessary and makes input/output channel specification confusing, you need to set
      this to *True* in order to match the behavior in many deep learning frameworks, such as TensorFlow, Keras, and
      PyTorch.
    precision: Optional. Either `None`, which means the default precision for
      the backend, a `lax.Precision` enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a
      tuple of two `lax.Precision` enums indicating precision of ``lhs``` and `rhs`.
  Returns:
    Transposed N-d convolution.
  """
  assert len(lhs.shape) == len(rhs.shape) and len(lhs.shape) >= 2
  ndims = len(lhs.shape)
  one = (1,) * (ndims - 2)
  # Set dimensional layout defaults if not specified.
  if dimension_numbers is None:
    if ndims == 2:
      dimension_numbers = ('NC', 'IO', 'NC')
    elif ndims == 3:
      dimension_numbers = ('NHC', 'HIO', 'NHC')
    elif ndims == 4:
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    elif ndims == 5:
      dimension_numbers = ('NHWDC', 'HWDIO', 'NHWDC')
    else:
      raise ValueError('No 4+ dimensional dimension_number defaults.')
  dn = conv_dimension_numbers_(lhs.shape, rhs.shape, dimension_numbers)
  # dn = dimension_numbers
  k_shape = np.take(rhs.shape, dn.rhs_spec)
  k_sdims = k_shape[2:]  # type: ignore[index]
  i_shape = np.take(lhs.shape, dn.lhs_spec)
  i_sdims = i_shape[2:]  # type: ignore[index]

  # Calculate correct output shape given padding and strides.
  if dilation is None:
    dilation = (1,) * (rhs.ndim - 2)

  if output_padding is None:
    output_padding = [None] * (rhs.ndim - 2)  # type: ignore[list-item]

  if isinstance(padding, str):
    if padding in {'SAME', 'VALID'}:
      padding = [padding] * (rhs.ndim - 2)  # type: ignore[list-item]
    else:
      raise ValueError(f"`padding` must be 'VALID' or 'SAME'. Passed: {padding}.")

  inferred_output_shape = tuple(map(_deconv_output_length, i_sdims, k_sdims,
                              padding, output_padding, strides, dilation))
  if output_shape is None:
    output_shape = inferred_output_shape  # type: ignore[assignment]
  else:
    if not output_shape == inferred_output_shape:
      raise ValueError(f"`output_padding` and `output_shape` are not compatible."
                       f"Inferred output shape from `output_padding`: {inferred_output_shape}, "
                       f"but got `output_shape` {output_shape}")

  pads = tuple(map(_compute_adjusted_padding, i_sdims, output_shape,
                   k_sdims, strides, padding, dilation))

  if transpose_kernel:
    # flip spatial dims and swap input / output channel axes
    rhs = _flip_axes(rhs, np.array(dn.rhs_spec)[2:])
    rhs = np.swapaxes(rhs, dn.rhs_spec[0], dn.rhs_spec[1])
  return conv_general_dilated(lhs, rhs, one, pads, strides, dilation, dimension_numbers,
                              precision=precision)


class ConvTransposeGradient(nn.Module):
    """Convolution Module wrapping lax.conv_transpose.

    Attributes:
        features: number of convolution filters.
        kernel_size: shape of the convolutional kernel. For 1D convolution,
        the kernel size can be passed as an integer. For all other cases, it must
        be a sequence of integers.
        strides: a sequence of `n` integers, representing the inter-window strides.
        padding: either the string `'SAME'`, the string `'VALID'`, the string
        `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpeted as applying the same padding
        in all dims and passign a single int in a sequence causes the same padding
        to be used on both sides.
        kernel_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel. Convolution with kernel dilation is also known as 'atrous
        convolution'.
        use_bias: whether to add a bias to the output (default: True).
        dtype: the dtype of the computation (default: float32).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
        kernel_init: initializer for the convolutional kernel.
        bias_init: initializer for the bias.
    """
    features: int
    kernel_size: Union[int, Tuple[int, ...]]
    strides: Optional[Tuple[int, ...]] = None
    padding: PaddingLike = 'SAME'
    kernel_dilation: Optional[Sequence[int]] = None
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a transposed convolution to the inputs.

        Behaviour mirrors of `jax.lax.conv_transpose`.

        Args:
        inputs: input data with dimensions (batch, spatial_dims..., features).
            This is the channels-last convention, i.e. NHWC for a 2d convolution and NDHWC for a 3D convolution. Note:
            this is different from the input convention used by `lax.conv_general_dilated`, which puts the spatial
            dimensions last.

        Returns:
        The convolved data.
        """
        inputs = jnp.asarray(inputs, self.dtype)

        kernel_size: Tuple[int, ...]
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size,)
        else:
            kernel_size = self.kernel_size

        is_single_input = False
        if inputs.ndim == len(kernel_size) + 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        strides: Tuple[int, ...]
        strides = self.strides or (1,) * (inputs.ndim - 2)

        in_features = inputs.shape[-1]
        kernel_shape = kernel_size + (in_features, self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape,
                            self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)

        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax == 'CIRCULAR':
            padding_lax = 'VALID'

        y = gradient_based_conv_transpose(
            inputs,
            kernel,
            strides,
            padding_lax,
            dilation=self.kernel_dilation,
            precision=self.precision)

        if self.padding == 'CIRCULAR':
            # For circular padding, we need to identify the size of the final output
            # ("period") along each spatial dimension, pad each dimension to an
            # integer number of periods, and wrap the array periodically around each
            # dimension. Padding should be done in such a way that the start of the
            # original input data inside the padded array is located at integer
            # number of periods - otherwise the result would be circularly shifted.

            # Compute period along each spatial dimension - it's input size scaled
            # by the stride.
            scaled_x_dims = [
                x_dim * stride for x_dim, stride in zip(inputs.shape[1:-1], strides)
            ]
            # Compute difference between the current size of y and the final output
            # size, and complement this difference to 2 * period - that gives how
            # much we need to pad.
            size_diffs = [
                -(y_dim - x_dim) % (2 * x_dim)
                for y_dim, x_dim in zip(y.shape[1:-1], scaled_x_dims)
            ]
            # Divide the padding equaly between left and right. The choice to put
            # "+1" on the left (and not on the right) represents a convention for
            # aligning even-sized kernels.
            total_pad = [
                ((size_diff + 1) // 2, size_diff // 2) for size_diff in size_diffs
            ]
            y = np.pad(y, [(0, 0)] + total_pad + [(0, 0)])
            # Wrap the result periodically around each spatial dimension,
            # one by one.
            for i in range(1, y.ndim - 1):
                y = y.reshape(y.shape[:i] + (-1, scaled_x_dims[i - 1]) +
                            y.shape[i + 1:])
                y = y.sum(axis=i)

        if is_single_input:
            y = jnp.squeeze(y, axis=0)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,),
                                self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y