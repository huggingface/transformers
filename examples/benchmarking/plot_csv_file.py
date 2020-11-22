import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from transformers import HfArgumentParser


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class PlotArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    csv_file: str = field(
        metadata={"help": "The csv file to plot."},
    )
    plot_along_batch: bool = field(
        default=False,
        metadata={"help": "Whether to plot along batch size or sequence length. Defaults to sequence length."},
    )
    is_time: bool = field(
        default=False,
        metadata={"help": "Whether the csv file has time results or memory results. Defaults to memory results."},
    )
    no_log_scale: bool = field(
        default=False,
        metadata={"help": "Disable logarithmic scale when plotting"},
    )
    is_train: bool = field(
        default=False,
        metadata={
            "help": "Whether the csv file has training results or inference results. Defaults to inference results."
        },
    )
    figure_png_file: Optional[str] = field(
        default=None,
        metadata={"help": "Filename under which the plot will be saved. If unused no plot is saved."},
    )
    short_model_names: Optional[List[str]] = list_field(
        default=None, metadata={"help": "List of model names that are used instead of the ones in the csv file."}
    )


def can_convert_to_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def can_convert_to_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


class Plot:
    def __init__(self, args):
        self.args = args
        self.result_dict = defaultdict(lambda: dict(bsz=[], seq_len=[], result={}))

        with open(self.args.csv_file, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                model_name = row["model"]
                self.result_dict[model_name]["bsz"].append(int(row["batch_size"]))
                self.result_dict[model_name]["seq_len"].append(int(row["sequence_length"]))
                if can_convert_to_int(row["result"]):
                    # value is not None
                    self.result_dict[model_name]["result"][
                        (int(row["batch_size"]), int(row["sequence_length"]))
                    ] = int(row["result"])
                elif can_convert_to_float(row["result"]):
                    # value is not None
                    self.result_dict[model_name]["result"][
                        (int(row["batch_size"]), int(row["sequence_length"]))
                    ] = float(row["result"])

    def plot(self):
        fig, ax = plt.subplots()
        title_str = "Time usage" if self.args.is_time else "Memory usage"
        title_str = title_str + " for training" if self.args.is_train else title_str + " for inference"

        if not self.args.no_log_scale:
            # set logarithm scales
            ax.set_xscale("log")
            ax.set_yscale("log")

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())

        for model_name_idx, model_name in enumerate(self.result_dict.keys()):
            batch_sizes = sorted(list(set(self.result_dict[model_name]["bsz"])))
            sequence_lengths = sorted(list(set(self.result_dict[model_name]["seq_len"])))
            results = self.result_dict[model_name]["result"]

            (x_axis_array, inner_loop_array) = (
                (batch_sizes, sequence_lengths) if self.args.plot_along_batch else (sequence_lengths, batch_sizes)
            )

            label_model_name = (
                model_name if self.args.short_model_names is None else self.args.short_model_names[model_name_idx]
            )

            for inner_loop_value in inner_loop_array:
                if self.args.plot_along_batch:
                    y_axis_array = np.asarray(
                        [results[(x, inner_loop_value)] for x in x_axis_array if (x, inner_loop_value) in results],
                        dtype=np.int,
                    )
                else:
                    y_axis_array = np.asarray(
                        [results[(inner_loop_value, x)] for x in x_axis_array if (inner_loop_value, x) in results],
                        dtype=np.float32,
                    )

                (x_axis_label, inner_loop_label) = (
                    ("batch_size", "len") if self.args.plot_along_batch else ("in #tokens", "bsz")
                )

                x_axis_array = np.asarray(x_axis_array, np.int)[: len(y_axis_array)]
                plt.scatter(
                    x_axis_array, y_axis_array, label=f"{label_model_name} - {inner_loop_label}: {inner_loop_value}"
                )
                plt.plot(x_axis_array, y_axis_array, "--")

            title_str += f" {label_model_name} vs."

        title_str = title_str[:-4]
        y_axis_label = "Time in s" if self.args.is_time else "Memory in MB"

        # plot
        plt.title(title_str)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.legend()

        if self.args.figure_png_file is not None:
            plt.savefig(self.args.figure_png_file)
        else:
            plt.show()


def main():
    parser = HfArgumentParser(PlotArguments)
    plot_args = parser.parse_args_into_dataclasses()[0]
    plot = Plot(args=plot_args)
    plot.plot()


if __name__ == "__main__":
    main()
