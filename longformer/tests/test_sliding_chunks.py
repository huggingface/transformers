import time
import unittest
import torch
import numpy as np
import random
from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from longformer.sliding_chunks import sliding_chunks_matmul_pv, sliding_chunks_matmul_qk


def same_storage(x, y):
    '''Tests if two tensors share the same underlying storage (for memory optimizations)'''
    return x.storage().data_ptr() == y.storage().data_ptr()


class TestSlidingChunksMM(unittest.TestCase):
    def test_tvm_equal_sliding_chunks(self):
        np.random.seed(3)
        random.seed(3)
        torch.manual_seed(3)
        torch.cuda.manual_seed(3)
        torch.cuda.manual_seed_all(3)

        torch.set_printoptions(sci_mode=False)
        N = 4096  # * 16
        M = 64  # hidden size
        W = 256  # one sided. Actual window size = 2w+1
        B = 3
        D = 1  # no dilation
        H = 12  # number of heads
        autoregressive = False  # not autoregressive
        device = 'cuda'
        dtype = torch.float32

        failed_tests = 0
        time1 = time2 = 0
        for i in range(50):
            if i < 5:
                time1 = time2 = 0  # don't include the first few iterations because of high variance

            query = torch.randn(B * N * H * M, requires_grad=True, device=device, dtype=dtype).view(B, N, H, M)
            key = torch.randn(B * N * H * M, requires_grad=True, device=device, dtype=dtype).flip(dims=(0,)).view(B, N, H, M)
            value = torch.randn(B * N * H * M, requires_grad=True, device=device, dtype=dtype).view(B, N, H, M)

            # TVM MM
            torch.cuda.synchronize()
            start = time.time()
            attention1 = diagonaled_mm_tvm(query, key, W, D, False, 0, autoregressive)
            mask_invalid_locations(attention1, W, D, autoregressive)
            attention_probs1 = torch.nn.functional.softmax(attention1, dim=-1)
            context1 = diagonaled_mm_tvm(attention_probs1, value, W, D, True, 0, autoregressive)
            context1.sum().backward()
            torch.cuda.synchronize()
            time1 += time.time() - start
            torch.cuda.empty_cache()

            # query = query.half()  # uncomment to profile the fp16 performance
            # key = key.half()
            # value = value.half()
            assert D == 1
            assert not autoregressive
            torch.cuda.synchronize()
            start = time.time()
            attention2 = sliding_chunks_matmul_qk(query, key, W, float('-inf'))
            attention_probs2 = torch.nn.functional.softmax(attention2, dim=-1)
            context2 = sliding_chunks_matmul_pv(attention_probs2, value, W)
            context2.sum().backward()
            torch.cuda.synchronize()
            time2 += time.time() - start
            torch.cuda.empty_cache()

            try:
                assert torch.allclose(attention1, attention2.float(), atol=1e-4, rtol=1e-5)
                assert torch.allclose(context1, context2.float(), atol=1e-4, rtol=1e-5)
            except AssertionError:
                failed_tests += 1

        print('Time tvm: {0:.5f} s'.format(time1))
        print('Time pytorch sliding chunks: {0:.5f} s'.format(time2))
        print('Sliding chunks vs. TVM speedup: {0:.5f}x'.format(time1/time2))
        print(f'Failed tests: {failed_tests}/{i+1}')
        assert failed_tests == 0


if __name__ == '__main__':
    unittest.main()
