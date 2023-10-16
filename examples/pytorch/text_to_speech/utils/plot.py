import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm


matplotlib.use("Agg")


def plot_alignment(alignment, info=None, fig_size=(16, 10), title=None, output_fig=False, plot_log=False):
    if isinstance(alignment, torch.Tensor):
        alignment_ = alignment.detach().cpu().numpy().squeeze()
    else:
        alignment_ = alignment
    alignment_ = alignment_.astype(np.float32) if alignment_.dtype == np.float16 else alignment_
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(
        alignment_.T, aspect="auto", origin="lower", interpolation="none", norm=LogNorm() if plot_log else None
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    # plt.yticks(range(len(text)), list(text))
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    if not output_fig:
        plt.close()
    return fig
