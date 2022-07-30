import datetime

import torch
from torch import nn



"""Numba implementation of resampler"""
from numba import (
    guvectorize,
    float32,
    float64,
    jit,
    prange,
    int16,
    int32,
    int64,
    complex64,
    complex128,
)


def _resample_loop(x, t_out, interp_win, interp_delta, num_table, scale, y):

    index_step = int(scale * num_table)
    time_register = 0.0

    n = 0
    frac = 0.0
    index_frac = 0.0
    offset = 0
    eta = 0.0
    weight = 0.0

    nwin = interp_win.shape[0]
    n_orig = x.shape[0]
    n_out = t_out.shape[0]

    for t in prange(n_out):
        time_register = t_out[t]

        # Grab the top bits as an index to the input buffer
        n = int(time_register)

        # Grab the fractional component of the time index
        frac = scale * (time_register - n)

        # Offset into the filter
        index_frac = frac * num_table
        offset = int(index_frac)

        # Interpolation factor
        eta = index_frac - offset

        # Compute the left wing of the filter response
        i_max = min(n + 1, (nwin - offset) // index_step)
        for i in range(i_max):

            weight = (
                interp_win[offset + i * index_step]
                + eta * interp_delta[offset + i * index_step]
            )
            y[t] += weight * x[n - i]

        # Invert P
        frac = scale - frac

        # Offset into the filter
        index_frac = frac * num_table
        offset = int(index_frac)

        # Interpolation factor
        eta = index_frac - offset

        # Compute the right wing of the filter response
        k_max = min(n_orig - n - 1, (nwin - offset) // index_step)
        for k in range(k_max):
            weight = (
                interp_win[offset + k * index_step]
                + eta * interp_delta[offset + k * index_step]
            )
            y[t] += weight * x[n + k + 1]


_resample_loop_p = jit(nopython=True, nogil=True, parallel=True)(_resample_loop)


@guvectorize(
    [
        (int16[:], float64[:], float64[:], float64[:], int32, float32, int16[:]),
        (int32[:], float64[:], float64[:], float64[:], int32, float32, int32[:]),
        (int64[:], float64[:], float64[:], float64[:], int32, float32, int64[:]),
        (float32[:], float64[:], float64[:], float64[:], int32, float32, float32[:]),
        (float64[:], float64[:], float64[:], float64[:], int32, float32, float64[:]),
        (
            complex64[:],
            float64[:],
            float64[:],
            float64[:],
            int32,
            float32,
            complex64[:],
        ),
        (
            complex128[:],
            float64[:],
            float64[:],
            float64[:],
            int32,
            float32,
            complex128[:],
        ),
    ],
    "(n),(m),(p),(p),(),()->(m)",
    nopython=True,
)
def resample_f_p(x, t_out, interp_win, interp_delta, num_table, scale, y):
    _resample_loop_p(x, t_out, interp_win, interp_delta, num_table, scale, y)


class DummyModel(nn.Module):
    def __init__(self, n_layers, dim):
        super().__init__()
        self.layers = [nn.Linear(dim, dim) for _ in range(n_layers)]
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            hidden_states = self.activation(hidden_states)

        return hidden_states


if __name__ == "__main__":

    batch_size = 64
    seq_len = 128

    vocab_size = 1024
    hidden_size = 32
    layer = nn.Embedding(vocab_size, hidden_size)

    inputs = torch.ones(batch_size, seq_len, dtype=torch.int32)

    for i in range(10):
        output = layer(inputs)

    for i in range(10):
        s = datetime.datetime.now()
        for j in range(300):
            output = layer(inputs)
        e = datetime.datetime.now()
        print(f"nn.Embedding ({i}): {(e - s).total_seconds() / 300} seconds")
    print("=" * 40)

    x = torch.ones(64, 128, 32, dtype=torch.float32)
    y = torch.ones(64, 128, 32, dtype=torch.float32)

    for i in range(10):
        z = x + y

    for i in range(10):
        s = datetime.datetime.now()
        for j in range(300):
            z = x + y
        e = datetime.datetime.now()
        print(f"z = x + y ({i}): {(e - s).total_seconds() / 300} seconds")
    print("=" * 40)

    q = torch.ones(64, 4, 128, 8, dtype=torch.float32)
    k = torch.ones(64, 4, 8, 128, dtype=torch.float32)

    for i in range(10):
        attention_scores = torch.matmul(q, k)

    for i in range(10):
        s = datetime.datetime.now()
        for j in range(300):
            attention_scores = torch.matmul(q, k)
        e = datetime.datetime.now()
        print(f"torch.matmul ({i}): {(e - s).total_seconds() / 300} seconds")
    print("=" * 40)
