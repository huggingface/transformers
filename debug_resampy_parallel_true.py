import datetime

import torch
from torch import nn
import resampy


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
