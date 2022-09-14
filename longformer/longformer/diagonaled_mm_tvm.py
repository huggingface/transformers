from typing import Union
from functools import lru_cache

import torch
import os.path


class DiagonaledMM(torch.autograd.Function):
    '''Class to encapsulate tvm code for compiling a diagonal_mm function, in addition to calling
    this function from PyTorch
    '''

    function_dict = {}  # save a list of functions, each has a different set of parameters

    @staticmethod
    def _compile_function(dtype: str, device: str, b0: int = 4, b1: int = 4, b2: int = 16):
        '''Compiles a tvm function that computes diagonal_mm
        args:
        dtype: str in ['float64', 'float32', 'float16']
        device: str in ['cpu' or 'cuda']
        b0, b1, b2: size of tensor tiles. Very important for good performance

        '''
        import tvm  # import the full tvm library here for compilation. Don't import at the top of the file in case we don't need to compile
        from tvm.contrib import nvcc
        @tvm.register_func
        def tvm_callback_cuda_compile(code):
            """Use nvcc compiler for better perf."""
            ptx = nvcc.compile_cuda(code, target="ptx", arch='sm_52')  # use old arch for this to work on old GPUs
            return ptx

        assert dtype in ['float16', 'float32', 'float64']
        assert device in ['cpu', 'cuda']
        device = None if device == 'cpu' else device
        tgt_host="llvm"

        b = tvm.var('b')  # batch size
        n = tvm.var('n')  # sequence length
        h = tvm.var('h')  # number of heads
        m = tvm.var('m')  # hidden dimension
        w = tvm.var('w')  # window size
        w_upper = tvm.var('w_upper')  # window size to the right of the word. Should be `0` or `w`
        padding = tvm.var('padding')  # padding
        transpose_t1 = tvm.var('transpose_t1')  # t1 should be transposed
        t1d3 = tvm.var('t1d3')  # last dimension of t1
        t3d3 = tvm.var('t3d3')  # last dimension of t3 (the result tensor)
        X = tvm.placeholder((b, n, h, t1d3), name='X', dtype=dtype)  # first tensor
        Y = tvm.placeholder((b, n, h, m), name='Y', dtype=dtype)  # second tensor
        k = tvm.reduce_axis((0, t1d3), name='k')  # dimension to sum over
        D = tvm.placeholder((h), name='D', dtype='int')  # dilation per head
        output_shape = (b, n, h, t3d3)  # shape of the result tensor
        algorithm = lambda l, i, q, j: tvm.sum(
            tvm.if_then_else(
                t3d3 == m,  # if output dimension == m, then t1 is diagonaled (FIXME: This breaks if t3d3 == m == t1d3)
                tvm.if_then_else(
                    transpose_t1 == 0,
                    tvm.if_then_else(
                        tvm.all(
                            i + D[q] * (k - w) >= 0,
                            i + D[q] * (k - w) < n,
                        ),
                        X[l, i, q, k] * Y[l, i + D[q] * (k - w), q, j],  # t1 is diagonaled
                        padding
                    ),
                    tvm.if_then_else(
                        tvm.all(
                            i + D[q] * (k - w_upper) >= 0,  # `w_upper` to handle the case `autoregressive=True`
                            i + D[q] * (k - w_upper) < n,
                        ),
                        X[l, i + D[q] * (k - w_upper), q, (w_upper + w) - k] * Y[l, i + D[q] * (k - w_upper), q, j],  # # t1 is diagonaled and should be transposed
                        padding
                    ),
                ),
                tvm.if_then_else(
                    tvm.all(
                        i + D[q] * (j - w) >= 0,
                        i + D[q] * (j - w) < n,
                    ),
                    X[l, i, q, k] * Y[l, i + D[q] * (j - w), q, k],  # t1 is not diagonaled, but the output tensor is going to be
                    padding
                )
            ), axis=k)

        Z = tvm.compute(output_shape, algorithm, name='Z')  # automatically generate cuda code
        s = tvm.create_schedule(Z.op)

        print('Lowering: \n ===================== \n{}'.format(tvm.lower(s, [X, Y, D], simple_mode=True)))

        # split long axis into smaller chunks and assing each one to a separate GPU thread/block
        ko, ki = s[Z].split(Z.op.reduce_axis[0], factor=b0)
        ZF = s.rfactor(Z, ki)

        j_outer, j_inner = s[Z].split(s[Z].op.axis[-1], factor=b1)
        i_outer, i_inner = s[Z].split(s[Z].op.axis[1], factor=b2)

        s[Z].bind(j_outer, tvm.thread_axis("blockIdx.x"))
        s[Z].bind(j_inner, tvm.thread_axis("threadIdx.y"))

        s[Z].bind(i_outer, tvm.thread_axis("blockIdx.y"))
        s[Z].bind(i_inner, tvm.thread_axis("threadIdx.z"))

        tx = tvm.thread_axis("threadIdx.x")
        s[Z].bind(s[Z].op.reduce_axis[0], tx)
        s[ZF].compute_at(s[Z], s[Z].op.reduce_axis[0])
        s[Z].set_store_predicate(tx.var.equal(0))

        print('Lowering with GPU splits: \n ===================== \n{}'.format(tvm.lower(s, [X, Y, D], simple_mode=True)))

        # compiling the automatically generated cuda code
        diagonaled_mm = tvm.build(s, [X, Y, Z, D, w, w_upper, padding, transpose_t1, t3d3], target=device, target_host=tgt_host, name='diagonaled_mm')
        return diagonaled_mm

    @staticmethod
    def _get_lib_filename(dtype: str, device: str):
        base_filename = 'longformer/lib/lib_diagonaled_mm'
        return '{}_{}_{}.so'.format(base_filename, dtype, device)

    @staticmethod
    def _save_compiled_function(f, dtype: str, device: str):
        if not os.path.exists('longformer/lib/'):
            os.makedirs('longformer/lib/')
        f.export_library(DiagonaledMM._get_lib_filename(dtype, device))

    @staticmethod
    def _load_compiled_function(dtype: str, device: str):
        from tvm.module import load  # this can be the small runtime python library, and doesn't need to be the whole thing
        filename = DiagonaledMM._get_lib_filename(dtype, device)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_dirs = ['../../', '../', './', f'{current_dir}/', f'{current_dir}/../']
        for potential_dir in  potential_dirs:
            filepath = '{}{}'.format(potential_dir, filename)
            if os.path.isfile(filepath):
                print('Loading tvm binary from: {}'.format(filepath))
                return load(filepath)
        return None

    @staticmethod
    def _get_function(dtype: str, device: str):
        '''Loads the function from the disk or compile it'''
        # A list of arguments that define the function
        args = (dtype, device)
        if args not in DiagonaledMM.function_dict:
            diagonaled_mm = DiagonaledMM._load_compiled_function(dtype, device)  # try to load from disk
            if not diagonaled_mm:
                print('Tvm binary not found. Compiling ...')
                diagonaled_mm = DiagonaledMM._compile_function(dtype, device)  # compile
                DiagonaledMM._save_compiled_function(diagonaled_mm, dtype, device)  # save to disk
            # convert the tvm function into a pytorch function
            from tvm.contrib import dlpack
            diagonaled_mm_pytorch = dlpack.to_pytorch_func(diagonaled_mm)  # wrap it as a pytorch function
            # save the function into a dictionary to be reused
            DiagonaledMM.function_dict[args] = diagonaled_mm_pytorch  # save it in a dictionary for next time
        return DiagonaledMM.function_dict[args]

    @staticmethod
    def _diagonaled_mm(t1: torch.Tensor, t2: torch.Tensor, w: int, d: Union[torch.Tensor,int],
                       is_t1_diagonaled: bool = False, transpose_t1: bool = False, padding: int = 0,
                       autoregressive: bool = False):
        '''Calls the compiled function after checking the input format. This function is called in three different modes.
        t1 x t2 = r ==> t1 and t2 are not diagonaled, but r is. Useful for query x key = attention_scores
        t1 x t2 = r ==> t1 is diagonaled, but t2 and r are not. Useful to compuate attantion_scores x value = context
        t1 x t2 = r ==> t1 is diagonaled and it should be transposed, but t2 and r are not diagonaled. Useful in some of
                            the calculations in the backward pass.
        '''
        dtype = str(t1.dtype).split('.')[1]
        device = t1.device.type
        assert len(t1.shape) == 4
        assert len(t1.shape) == len(t2.shape)
        assert t1.shape[:3] == t2.shape[:3]
        if isinstance(d, int):  # if d is an integer, replace it with a tensor of the same length
                                # as number of heads, and it is filled with the same dilation value
            d = t1.new_full(size=(t1.shape[2],), fill_value=d, dtype=torch.int, requires_grad=False)

        assert len(d.shape) == 1
        assert d.shape[0] == t1.shape[2]  # number of dilation scores should match number of heads
        b = t1.shape[0]  # batch size
        n = t1.shape[1]  # sequence length
        h = t1.shape[2]  # number of heads
        m = t2.shape[3]  # hidden dimension
        w_upper = 0 if autoregressive else w
        c = w_upper + w + 1  # number of diagonals
        if is_t1_diagonaled:
            assert t1.shape[3] == c
            r = t1.new_empty(b, n, h, m)  # allocate spase for the result tensor
        else:
            assert not transpose_t1
            assert t1.shape[3] == m
            r = t1.new_empty(b, n, h, c)  # allocate spase for the result tensor

        # gets function from memory, from disk or compiles it from scratch
        _diagonaled_mm_function = DiagonaledMM._get_function(dtype=dtype, device=device)

        # The last argument to this function is a little hacky. It is the size of the last dimension of the result tensor
        # We use it as a proxy to tell if t1_is_diagonaled or not (if t1 is diagonaled, result is not, and vice versa).
        # The second reason is that the lambda expression in `_compile_function` is easier to express when the shape
        # of the output is known
        # This functions computes diagonal_mm then saves the result in `r`
        if m == c:
            # FIXME
            print('Error: the hidden dimension {m} shouldn\'t match number of diagonals {c}')
            assert False
        _diagonaled_mm_function(t1, t2, r, d, w, w_upper, padding, transpose_t1, m if is_t1_diagonaled else c)
        return r

    @staticmethod
    def _prepare_tensors(t):
        '''Fix `stride()` information of input tensor. This addresses some inconsistency in stride information in PyTorch.
        For a tensor t, if t.size(0) == 1, then the value of t.stride()[0] doesn't matter.
        TVM expects this value to be the `product(t.size()[1:])` but PyTorch some times sets it to `t.stride()[1]`.
        Here's an example to reporduce this issue:
            import torch
            print(torch.randn(1, 10).stride())
            > (10, 1)
            print(torch.randn(10, 1).t().contiguous().stride())
            > (1, 1)  # expected it to be (10, 1) as above
            print(torch.randn(10, 2).t().contiguous().stride())
            > (10, 1) # but gets the expected stride if the first dimension is > 1
        '''
        assert t.is_contiguous()
        t_stride = list(t.stride())
        t_size = list(t.size())
        # Fix wrong stride information for the first dimension. This occures when batch_size=1
        if t_size[0] == 1 and t_stride[0] == t_stride[1]:
            # In this case, the stride of the first dimension should be the product
            # of the sizes  of all other dimensions
            t_stride[0] = t_size[1] * t_size[2] * t_size[3]
            t = t.as_strided(size=t_size, stride=t_stride)
        return t

    min_seq_len = 16  # unexpected output if seq_len < 16

    @staticmethod
    def forward(ctx, t1: torch.Tensor, t2: torch.Tensor, w: int, d: Union[torch.Tensor,int], is_t1_diagonaled: bool = False, padding: int = 0, autoregressive: bool = False) -> torch.Tensor:
        '''Compuates diagonal_mm of t1 and t2.
        args: 
        t1: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size|number_of_diagonals).
            t1 can be a regular tensor (e.g. `query_layer`) or a diagonaled one (e.g. `attention_scores`)
        t2: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size). This is always a non-diagonaled
            tensor, e.g. `key_layer` or `value_layer`
        w: int = window size; number of attentions on each side of the word
        d: torch.Tensor or int = dilation of attentions per attention head. If int, the same dilation value will be used for all
            heads. If torch.Tensor, it should be 1D of lenth=number of attention heads
        is_t1_diagonaled: is t1 a diagonaled or a regular tensor
        padding: the padding value to use when accessing invalid locations. This is mainly useful when the padding
            needs to be a very large negative value (to compute softmax of attentions). For other usecases,
            please use zero padding.
        autoregressive: if true, return only the lower triangle
        returns: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size|number_of_diagonals)
            if t1 is diagonaed, result is non-diagonaled, and vice versa
        '''
        batch_size, seq_len, num_attention_heads, hidden_size = t1.size()
        assert seq_len >= DiagonaledMM.min_seq_len, 'avoid splitting errors by using seq_len >= {}'.format(DiagonaledMM.min_seq_len)  # FIXME
        ctx.save_for_backward(t1, t2)
        ctx.w = w
        ctx.d = d
        ctx.is_t1_diagonaled = is_t1_diagonaled
        ctx.autoregressive = autoregressive
        t1 = DiagonaledMM._prepare_tensors(t1)
        t2 = DiagonaledMM._prepare_tensors(t2)
        # output = t1.mm(t2)  # what would have been called if this was a regular matmul
        output = DiagonaledMM._diagonaled_mm(t1, t2, w, d, is_t1_diagonaled=is_t1_diagonaled, padding=padding, autoregressive=autoregressive)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        t1, t2 = ctx.saved_tensors
        w = ctx.w
        d = ctx.d
        is_t1_diagonaled = ctx.is_t1_diagonaled
        autoregressive = ctx.autoregressive
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()  # tvm requires all input tensors to be contiguous
        grad_output = DiagonaledMM._prepare_tensors(grad_output)
        t1 = DiagonaledMM._prepare_tensors(t1)
        t2 = DiagonaledMM._prepare_tensors(t2)
        # http://cs231n.github.io/optimization-2/
        # https://pytorch.org/docs/master/notes/extending.html
        # grad_t1 = grad_output.mm(t2)  # what would have been called if this was a regular matmul
        grad_t1 = DiagonaledMM._diagonaled_mm(grad_output, t2, w, d, is_t1_diagonaled=not is_t1_diagonaled, autoregressive=autoregressive)
        # grad_t2 = grad_output.t().mm(t1)  # or `grad_t2 = t1.t().mm(grad_output).t()` because `(AB)^T = B^TA^T`
        if is_t1_diagonaled:
            grad_t2 = DiagonaledMM._diagonaled_mm(t1, grad_output, w, d, is_t1_diagonaled=True, transpose_t1=True, autoregressive=autoregressive)
        else:
            grad_t2 = DiagonaledMM._diagonaled_mm(grad_output, t1, w, d, is_t1_diagonaled=True, transpose_t1=True, autoregressive=autoregressive)
        return grad_t1, grad_t2, None, None, None, None, None


def _get_invalid_locations_mask_fixed_dilation(seq_len: int, w: int, d: int):
    diagonals_list = []
    for j in range(-d * w, d, d):
        diagonal_mask = torch.zeros(seq_len, device='cpu', dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)
    return torch.stack(diagonals_list, dim=-1)

@lru_cache()
def _get_invalid_locations_mask(w: int, d: Union[torch.Tensor,int], autoregressive: bool, device: str):
    if isinstance(d, int):
        affected_seq_len = w * d
        mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
        mask = mask[None, :, None, :]
    else:
        affected_seq_len = w * d.max()
        head_masks = []
        d_list = d.cpu().numpy().tolist()
        for d in d_list:
            one_head_mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
            head_masks.append(one_head_mask)
        mask = torch.stack(head_masks, dim=-2)
        mask = mask[None, :, :, :]

    ending_mask = None if autoregressive else mask.flip(dims=(1, 3)).bool().to(device)
    return affected_seq_len, mask.bool().to(device), ending_mask

def mask_invalid_locations(input_tensor: torch.Tensor, w: int, d: Union[torch.Tensor, int], autoregressive: bool) -> torch.Tensor:
    affected_seq_len, beginning_mask, ending_mask = _get_invalid_locations_mask(w, d, autoregressive, input_tensor.device)
    seq_len = input_tensor.size(1)
    beginning_input = input_tensor[:, :affected_seq_len, :, :w+1]
    beginning_mask = beginning_mask[:, :seq_len].expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask, -float('inf'))
    if not autoregressive:
        ending_input = input_tensor[:, -affected_seq_len:, :, -(w+1):]
        ending_mask = ending_mask[:, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float('inf'))


diagonaled_mm = DiagonaledMM.apply

# The non-tvm implementation is the default, we don't need to load the kernel at loading time.
# DiagonaledMM._get_function('float32', 'cuda')
