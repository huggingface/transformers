from datetime import datetime

import matplotlib.pyplot as plt
import torch


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    if device == "mps":
        print(
            "WARNING: MPS currently doesn't seem to work, and messes up backpropagation without any visible torch"
            " errors. I recommend using CUDA on a colab notebook or CPU instead if you're facing inexplicable issues"
            " with generations."
        )
    return device


def show_pil(img):
    fig = plt.imshow(img)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


def get_timestamp():
    current_time = datetime.now()
    timestamp = current_time.strftime("%H:%M:%S")
    return timestamp
