import random

import numpy as np
import torch


def set_seed(x=42):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(x)
